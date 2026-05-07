use std::collections::HashMap;

pub struct Tokenizer {
    sp: Option<sentencepiece::SentencePieceProcessor>,
    token_map: HashMap<u32, String>,
    blank_id: u32,
}

impl Tokenizer {
    pub fn from_file(model_path: &str, tokens_path: &str, blank_id: u32) -> crate::Result<Self> {
        // Try SentencePiece model first
        let sp = if !model_path.is_empty() && std::path::Path::new(model_path).exists() {
            Some(
                sentencepiece::SentencePieceProcessor::open(model_path)
                    .map_err(|e| crate::SiamError::Tokenizer(e.to_string()))?,
            )
        } else {
            None
        };

        // Build plain token map from tokens.txt as fallback / supplement
        let mut token_map = HashMap::new();
        if !tokens_path.is_empty() && std::path::Path::new(tokens_path).exists() {
            let text = std::fs::read_to_string(tokens_path).map_err(crate::SiamError::Io)?;
            for line in text.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2
                    && let Ok(id) = parts.last().unwrap().parse::<u32>()
                {
                    let token = parts[..parts.len() - 1].join(" ");
                    token_map.insert(id, token);
                }
            }
        }

        Ok(Self {
            sp,
            token_map,
            blank_id,
        })
    }

    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let filtered: Vec<u32> = ids
            .iter()
            .filter(|&&id| id != self.blank_id)
            .copied()
            .collect();

        if let Some(ref sp) = self.sp {
            sp.decode_piece_ids(&filtered).unwrap_or_default()
        } else {
            let mut text = String::new();
            for id in filtered {
                if let Some(token) = self.token_map.get(&id) {
                    text.push_str(token);
                }
            }
            text
        }
    }

    pub fn vocab_size(&self) -> usize {
        if let Some(ref sp) = self.sp {
            sp.len()
        } else {
            self.token_map.len()
        }
    }

    pub fn blank_id(&self) -> u32 {
        self.blank_id
    }
}
