//! Configuration file support for phonex.
//!
//! Load order (later overrides earlier):
//! 1. Built-in defaults
//! 2. Config file (`--config phonex.yaml`)
//! 3. Environment variables
//! 4. CLI flags

use serde::{Deserialize, Serialize};

/// Top-level configuration.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct PhonexConfig {
    #[serde(default)]
    pub model: ModelConfig,
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub tls: Option<TlsConfig>,
    #[serde(default)]
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfig {
    pub dir: Option<String>,
    pub language: Option<String>,
    pub pool_size: usize,
    pub diarization_model: Option<String>,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            dir: None,
            language: Some("english".into()),
            pool_size: 1,
            diarization_model: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ServerConfig {
    pub bind: String,
    pub port: u16,
    pub grpc_port: Option<u16>,
    #[serde(default)]
    pub limits: LimitsConfig,
    #[serde(default)]
    pub auth: AuthConfig,
    #[serde(default)]
    pub cors: CorsConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: "127.0.0.1".into(),
            port: 8080,
            grpc_port: None,
            limits: LimitsConfig::default(),
            auth: AuthConfig::default(),
            cors: CorsConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LimitsConfig {
    pub body_limit_mb: usize,
    pub rate_limit_per_minute: u32,
    pub max_ws_connections: usize,
    pub ws_idle_timeout_secs: u64,
    #[serde(default)]
    pub trust_proxy: bool,
    #[serde(default = "default_max_ws_message_size_bytes")]
    pub max_ws_message_size_bytes: usize,
    #[serde(default = "default_max_ws_audio_buffer_seconds")]
    pub max_ws_audio_buffer_seconds: u64,
}

fn default_max_ws_message_size_bytes() -> usize {
    10 * 1024 * 1024
}

fn default_max_ws_audio_buffer_seconds() -> u64 {
    30
}

impl Default for LimitsConfig {
    fn default() -> Self {
        Self {
            body_limit_mb: 500,
            rate_limit_per_minute: 0,
            max_ws_connections: 100,
            ws_idle_timeout_secs: 60,
            trust_proxy: false,
            max_ws_message_size_bytes: default_max_ws_message_size_bytes(),
            max_ws_audio_buffer_seconds: default_max_ws_audio_buffer_seconds(),
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct AuthConfig {
    pub api_key: Option<String>,
    pub admin_api_key: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CorsConfig {
    pub origins: Vec<String>,
}

impl Default for CorsConfig {
    fn default() -> Self {
        Self {
            origins: vec![
                "http://localhost:3000".into(),
                "http://localhost:5173".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TlsConfig {
    pub cert: String,
    pub key: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LoggingConfig {
    pub format: String,
    pub filter: String,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            format: "pretty".into(),
            filter: "info".into(),
        }
    }
}

/// Load configuration from a YAML file.
pub fn from_file(path: &str) -> anyhow::Result<PhonexConfig> {
    let contents = std::fs::read_to_string(path)?;
    let config: PhonexConfig = serde_yaml::from_str(&contents)?;
    Ok(config)
}
