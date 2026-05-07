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
}

impl PhonexGrpcService {
    pub fn new(engine: Arc<Engine>, model_info: ModelInfo, model_dir: String) -> Self {
        Self { engine, model_info, model_dir }
    }

    pub fn into_server(self) -> TranscriptionServiceServer<Self> {
        TranscriptionServiceServer::new(self)
    }
}

#[tonic::async_trait]
impl TranscriptionService for PhonexGrpcService {
    async fn transcribe(
        &self,
        request: Request<TranscribeRequest>,
    ) -> Result<Response<TranscribeResponse>, Status> {
        let req = request.into_inner();
        
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

    async fn stream_transcribe(
        &self,
        request: Request<Streaming<StreamAudioRequest>>,
    ) -> Result<Response<Self::StreamTranscribeStream>, Status> {
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
