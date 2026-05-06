//! Session pool for ONNX inference triplets.

use std::ops::{Deref, DerefMut};

use crate::encoder::OfflineEncoder;
use crate::decoder::SherpaDecoder;
use crate::joiner::SherpaJoiner;
use crate::model_config::ModelInfo;

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    Closed,
}

impl std::fmt::Display for PoolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PoolError::Closed => write!(f, "session pool is closed"),
        }
    }
}

impl std::error::Error for PoolError {}

/// Pool of pre-loaded items of type `T` backed by an MPMC `async-channel`.
pub struct Pool<T> {
    sender: async_channel::Sender<T>,
    receiver: async_channel::Receiver<T>,
    total: usize,
}

/// Public alias for the production pool.
pub type SessionPool = Pool<SessionTriplet>;

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
pub struct SessionTriplet {
    pub encoder: OfflineEncoder,
    pub decoder: SherpaDecoder,
    pub joiner: SherpaJoiner,
}

impl SessionTriplet {
    pub fn new(
        encoder: OfflineEncoder,
        decoder: SherpaDecoder,
        joiner: SherpaJoiner,
    ) -> Self {
        Self {
            encoder,
            decoder,
            joiner,
        }
    }

    pub fn from_model_dir(model_dir: &str, info: &ModelInfo) -> crate::Result<Self> {
        let paths = crate::model_config::discover_model_files(model_dir)?;
        let encoder = OfflineEncoder::new(
            paths.encoder.to_str().unwrap_or(""),
            info,
        )?;
        let decoder = SherpaDecoder::new(
            paths.decoder.to_str().unwrap_or(""),
            info,
        )?;
        let joiner = SherpaJoiner::new(
            paths.joiner.to_str().unwrap_or(""),
            info,
        )?;
        Ok(Self::new(encoder, decoder, joiner))
    }
}

impl<T> Pool<T> {
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        let (sender, receiver) = async_channel::bounded(total.max(1));
        for item in items {
            sender.try_send(item).expect("channel capacity matches item count");
        }
        Self { sender, receiver, total }
    }

    pub async fn checkout(&self) -> Result<PoolGuard<'_, T>, PoolError> {
        match self.receiver.recv().await {
            Ok(item) => Ok(PoolGuard { pool: self, item: Some(item) }),
            Err(_) => Err(PoolError::Closed),
        }
    }

    pub fn try_checkout(&self) -> Option<PoolGuard<'_, T>> {
        match self.receiver.try_recv() {
            Ok(item) => Some(PoolGuard { pool: self, item: Some(item) }),
            Err(_) => None,
        }
    }

    pub fn close(&self) {
        self.sender.close();
        self.receiver.close();
    }

    pub fn total(&self) -> usize {
        self.total
    }

    pub fn available(&self) -> usize {
        self.receiver.len()
    }
}

/// RAII guard that auto-checks-in an item when dropped.
pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
}

impl<T> PoolGuard<'_, T> {
    pub fn into_owned(mut self) -> (T, OwnedReservation<T>) {
        let item = self.item.take().expect("PoolGuard::into_owned called after drop");
        let reservation = OwnedReservation { sender: self.pool.sender.clone() };
        (item, reservation)
    }
}

impl<T> Deref for PoolGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.item.as_ref().expect("PoolGuard accessed after item taken")
    }
}

impl<T> DerefMut for PoolGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item.as_mut().expect("PoolGuard accessed after item taken")
    }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take() {
            let _ = self.pool.sender.try_send(item);
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts.
pub struct OwnedReservation<T> {
    sender: async_channel::Sender<T>,
}

impl<T> OwnedReservation<T> {
    pub fn checkin(self, item: T) {
        let _ = self.sender.try_send(item);
    }
}
