fn main() {
    #[cfg(feature = "grpc")]
    tonic_build::compile_protos("proto/phonex.proto").expect("Failed to compile phonex.proto");
}
