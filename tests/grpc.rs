//! Integration tests for the gRPC API.

#[cfg(feature = "grpc")]
mod grpc_tests {
    use std::net::SocketAddr;
    use std::sync::Arc;
    use tokio::net::TcpListener;
    use tonic_health::pb::health_client::HealthClient;
    use tonic_health::pb::HealthCheckRequest;

    use phonex::inference::pool::SessionPool;
    use phonex::inference::Engine;
    use phonex::tokenizer::Tokenizer;

    fn test_engine() -> (Arc<Engine>, phonex::model_config::ModelInfo) {
        let paths =
            phonex::model_config::discover_model_files("models/sherpa-onnx-zipformer-thai-2024-06-20")
                .unwrap();
        let tokenizer = Arc::new(
            Tokenizer::from_file(
                paths.tokenizer.to_str().unwrap_or(""),
                paths.tokens.to_str().unwrap_or(""),
                0,
            )
            .unwrap(),
        );
        let pool = SessionPool::new(vec![]);
        let info = phonex::model_config::ModelInfo {
            sample_rate: 16000,
            n_mels: 80,
            blank_id: 0,
            context_size: 2,
            d_model: 512,
            vocab_size: 2000,
            encoder_inputs: vec!["x".into()],
            encoder_outputs: vec!["encoder_out".into()],
            decoder_inputs: vec!["decoder_input".into()],
            decoder_outputs: vec!["decoder_out".into()],
            joiner_inputs: vec!["encoder_out".into(), "decoder_out".into()],
            joiner_outputs: vec!["logits".into()],
            model_id: "sherpa-onnx-zipformer-th".into(),
            model_name: "sherpa-onnx-zipformer-thai-2024-06-20".into(),
        };
        (
            Arc::new(Engine::new(pool, tokenizer, info.clone())),
            info,
        )
    }

    #[tokio::test]
    async fn test_grpc_health_check() {
        let (engine, info) = test_engine();
        let grpc_svc = phonex::server::grpc::PhonexGrpcService::new(
            engine,
            info,
            "models/sherpa-onnx-zipformer-thai-2024-06-20".to_string(),
            100,
            None,
        );

        let (mut health_reporter, health_service) = tonic_health::server::health_reporter();
        health_reporter
            .set_serving::<phonex::server::grpc::pb::transcription_service_server::TranscriptionServiceServer<phonex::server::grpc::PhonexGrpcService>>()
            .await;

        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr: SocketAddr = listener.local_addr().unwrap();

        tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(health_service)
                .add_service(grpc_svc.into_server())
                .serve_with_incoming(tokio_stream::wrappers::TcpListenerStream::new(listener))
                .await
                .unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;

        let channel = tonic::transport::Channel::from_shared(format!("http://{}", addr))
            .unwrap()
            .connect()
            .await
            .expect("Failed to connect to gRPC health service");
        let mut client = HealthClient::new(channel);

        let response = client
            .check(HealthCheckRequest {
                service: "phonex.TranscriptionService".to_string(),
            })
            .await
            .expect("Health check failed");

        assert_eq!(response.into_inner().status, tonic_health::pb::health_check_response::ServingStatus::Serving as i32);
    }
}
