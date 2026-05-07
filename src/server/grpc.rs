//! gRPC API for phonex transcription.

use std::pin::Pin;
use std::sync::Arc;

use tonic::{Request, Response, Status, Streaming};
use tokio_stream::Stream;

use crate::inference::Engine;
use crate::streaming_pipeline::StreamingPipeline;
use crate::model_config::ModelInfo;

pub mod pb {
    tonic::include_proto!("phonex");
}

use pb::transcription_service_server::{TranscriptionService, TranscriptionServiceServer};
use pb::*;

pub struct PhonexGrpcService {
    engine: Arc<Engine>,
    model_info: ModelInfo,
    model_dir: String,
    stream_semaphore: Option<Arc<tokio::sync::Semaphore>>,
    api_key: Option<String>,
}

impl PhonexGrpcService {
    pub fn new(
        engine: Arc<Engine>,
        model_info: ModelInfo,
        model_dir: String,
        max_streaming_connections: usize,
        api_key: Option<String>,
    ) -> Self {
        let stream_semaphore = if max_streaming_connections > 0 {
            Some(Arc::new(tokio::sync::Semaphore::new(max_streaming_connections)))
        } else {
            None
        };
        Self { engine, model_info, model_dir, stream_semaphore, api_key }
    }

    fn check_auth<T>(&self, request: &Request<T>) -> Result<(), Status> {
        if let Some(ref expected) = self.api_key {
            let valid = request.metadata()
                .get("authorization")
                .and_then(|v| v.to_str().ok())
                .is_some_and(|s| s.strip_prefix("Bearer ").is_some_and(|t| t == expected));
            if !valid {
                return Err(Status::unauthenticated("Invalid API key"));
            }
        }
        Ok(())
    }

    pub fn into_server(self) -> TranscriptionServiceServer<Self> {
        TranscriptionServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl TranscriptionService for PhonexGrpcService {
    #[tracing::instrument(skip(self, request))]
    async fn transcribe(
        &self,
        request: Request<TranscribeRequest>,
    ) -> Result<Response<TranscribeResponse>, Status> {
        self.check_auth(&request)?;
        let req = request.into_inner();
        
        const MAX_AUDIO_BYTES: usize = 500 * 1024 * 1024; // 500 MB
        if req.audio_data.len() > MAX_AUDIO_BYTES {
            return Err(Status::invalid_argument(format!(
                "Audio data exceeds maximum size of {} bytes",
                MAX_AUDIO_BYTES
            )));
        }
        
        let mut guard = self.engine.pool.try_checkout()
            .ok_or_else(|| Status::resource_exhausted("No available inference sessions"))?;
        
        let samples = if req.sample_rate == 0 || req.sample_rate == self.engine.info.sample_rate as i32 {
            crate::inference::audio::bytes_to_f32_samples(&req.audio_data)
        } else {
            let raw = crate::inference::audio::bytes_to_f32_samples(&req.audio_data);
            crate::inference::resample(&raw, req.sample_rate as u32, self.engine.info.sample_rate)
                .map_err(|e| Status::internal(format!("Resample error: {e}")))?
        };
        
        let result = if req.enable_vad {
            self.engine.transcribe_samples_with_vad(&samples, &mut guard)
        } else {
            self.engine.transcribe_samples(&samples, &mut guard)
        }.map_err(|e| Status::internal(format!("Inference error: {e}")))?;
        
        let words = result.words.into_iter().map(|w| WordInfo {
            word: w.word,
            start: w.start,
            end: w.end,
            confidence: w.confidence,
        }).collect();
        
        Ok(Response::new(TranscribeResponse {
            text: result.text,
            words,
            duration_seconds: result.duration_s,
            model_id: self.model_info.model_id.clone(),
        }))
    }

    type StreamTranscribeStream = Pin<Box<dyn Stream<Item = Result<StreamTranscriptResponse, Status>> + Send>>;

    #[tracing::instrument(skip(self, request))]
    async fn stream_transcribe(
        &self,
        request: Request<Streaming<StreamAudioRequest>>,
    ) -> Result<Response<Self::StreamTranscribeStream>, Status> {
        self.check_auth(&request)?;
        let _permit = match &self.stream_semaphore {
            Some(sem) => {
                match tokio::time::timeout(
                    std::time::Duration::from_secs(5),
                    sem.acquire(),
                ).await {
                    Ok(Ok(p)) => Some(p),
                    _ => return Err(Status::resource_exhausted(
                        "Too many concurrent streaming connections"
                    )),
                }
            }
            None => None,
        };

        let mut stream = request.into_inner();
        let model_dir = self.model_dir.clone();
        let info = self.model_info.clone();
        
        let output = async_stream::try_stream! {
            let mut pipeline: Option<StreamingPipeline> = None;
            let mut sample_rate = 16000i32;
            
            while let Some(msg) = stream.message().await? {
                match msg.payload {
                    Some(stream_audio_request::Payload::Config(cfg)) => {
                        sample_rate = cfg.sample_rate;
                        let vad_path = if cfg.enable_vad { Some("models/silero_vad.onnx") } else { None };
                        pipeline = Some(StreamingPipeline::from_model_dir(&model_dir, &info, vad_path)
                            .map_err(|e| Status::internal(format!("Pipeline error: {e}")))?);
                    }
                    Some(stream_audio_request::Payload::AudioChunk(data)) => {
                        let p = pipeline.as_mut().ok_or_else(|| Status::failed_precondition("Stream not configured"))?;
                        let samples = crate::inference::audio::bytes_to_f32_samples(&data);
                        let samples = if sample_rate != 16000 {
                            crate::inference::resample(&samples, sample_rate as u32, 16000)
                                .map_err(|e| Status::internal(format!("Resample error: {e}")))?
                        } else { samples };
                        
                        let tokens = p.accept_audio(&samples)
                            .map_err(|e| Status::internal(format!("Inference error: {e}")))?;
                        
                        for token in tokens {
                            yield StreamTranscriptResponse {
                                result: Some(stream_transcript_response::Result::Word(pb::WordInfo {
                                    word: token.text,
                                    start: token.start,
                                    end: token.end,
                                    confidence: token.confidence,
                                })),
                                timestamp: token.start,
                                is_final: false,
                            };
                        }
                    }
                    Some(stream_audio_request::Payload::Finalize(_)) => {
                        let p = pipeline.as_mut().ok_or_else(|| Status::failed_precondition("Stream not configured"))?;
                        let (text, tokens) = p.flush_with_tokens()
                            .map_err(|e| Status::internal(format!("Flush error: {e}")))?;
                        
                        yield StreamTranscriptResponse {
                            result: Some(stream_transcript_response::Result::FinalText(text)),
                            timestamp: tokens.last().map(|t| t.end).unwrap_or(0.0),
                            is_final: true,
                        };
                    }
                    None => {}
                }
            }
        };
        
        Ok(Response::new(Box::pin(output)))
    }
}
