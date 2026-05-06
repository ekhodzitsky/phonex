pub struct Tokenizer {
    sp: sentencepiece::SentencePieceProcessor,
    blank_id: u32,
}

impl Tokenizer {
    pub fn from_file(model_path: &str) -> crate::Result<Self> {
        let sp = sentencepiece::SentencePieceProcessor::open(model_path)
            .map_err(|e| crate::SiamError::Tokenizer(e.to_string()))?;
        let blank_id = 0;
        Ok(Self { sp, blank_id })
    }

    pub fn decode_ids(&self, ids: &[u32]) -> String {
        let sp_ids: Vec<u32> = ids
            .iter()
            .filter(|&&id| id != self.blank_id)
            .copied()
            .collect();
        self.sp.decode_piece_ids(&sp_ids).unwrap_or_default()
    }

    pub fn vocab_size(&self) -> usize {
        self.sp.len()
    }

    pub fn blank_id(&self) -> u32 {
        self.blank_id
    }
}
