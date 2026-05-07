//! Session pool for ONNX inference triplets.

use std::ops::{Deref, DerefMut};

use crate::decoder::SherpaDecoder;
use crate::encoder::OfflineEncoder;
use crate::joiner::SherpaJoiner;
use crate::model_config::ModelInfo;

/// Errors returned by [`Pool::checkout`].
#[derive(Debug)]
pub enum PoolError {
    /// The pool has been closed and can no longer hand out items.
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
///
/// Items are checked out asynchronously and automatically returned when the
/// guard is dropped. Use [`SessionPool`] for the concrete ONNX inference pool.
pub struct Pool<T> {
    sender: async_channel::Sender<T>,
    receiver: async_channel::Receiver<T>,
    total: usize,
}

/// Public alias for the production pool.
pub type SessionPool = Pool<SessionTriplet>;

/// A set of ONNX sessions for one inference pipeline (encoder + decoder + joiner).
pub struct SessionTriplet {
    /// Encoder session (converts audio features to encoder states).
    pub encoder: OfflineEncoder,
    /// Decoder session (predicts next token from previous tokens).
    pub decoder: SherpaDecoder,
    /// Joiner session (combines encoder and decoder states for scoring).
    pub joiner: SherpaJoiner,
}

impl SessionTriplet {
    pub fn new(encoder: OfflineEncoder, decoder: SherpaDecoder, joiner: SherpaJoiner) -> Self {
        Self {
            encoder,
            decoder,
            joiner,
        }
    }

    pub fn from_model_dir(model_dir: &str, info: &ModelInfo) -> crate::Result<Self> {
        let paths = crate::model_config::discover_model_files(model_dir)?;
        let encoder = OfflineEncoder::new(paths.encoder.to_str().unwrap_or(""), info)?;
        let decoder = SherpaDecoder::new(paths.decoder.to_str().unwrap_or(""), info)?;
        let joiner = SherpaJoiner::new(paths.joiner.to_str().unwrap_or(""), info)?;
        Ok(Self::new(encoder, decoder, joiner))
    }
}

impl<T> Pool<T> {
    pub fn new(items: Vec<T>) -> Self {
        let total = items.len();
        let (sender, receiver) = async_channel::bounded(total.max(1));
        for item in items {
            sender
                .try_send(item)
                .expect("channel capacity matches item count");
        }
        Self {
            sender,
            receiver,
            total,
        }
    }

    pub async fn checkout(&self) -> Result<PoolGuard<'_, T>, PoolError> {
        match self.receiver.recv().await {
            Ok(item) => Ok(PoolGuard {
                pool: self,
                item: Some(item),
            }),
            Err(_) => Err(PoolError::Closed),
        }
    }

    pub fn try_checkout(&self) -> Option<PoolGuard<'_, T>> {
        match self.receiver.try_recv() {
            Ok(item) => Some(PoolGuard {
                pool: self,
                item: Some(item),
            }),
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
///
/// Derefs to the underlying item for convenient access. Use [`PoolGuard::into_owned`]
/// to convert into a [`CheckoutGuard`] for `'static` contexts.
pub struct PoolGuard<'a, T> {
    pool: &'a Pool<T>,
    item: Option<T>,
}

impl<T> PoolGuard<'_, T> {
    pub fn into_owned(mut self) -> CheckoutGuard<T> {
        let item = self
            .item
            .take()
            .expect("PoolGuard::into_owned called after drop");
        let reservation = OwnedReservation {
            sender: self.pool.sender.clone(),
        };
        CheckoutGuard {
            item: Some(item),
            reservation: Some(reservation),
        }
    }
}

impl<T> Deref for PoolGuard<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .expect("PoolGuard accessed after item taken")
    }
}

impl<T> DerefMut for PoolGuard<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .expect("PoolGuard accessed after item taken")
    }
}

impl<T> Drop for PoolGuard<'_, T> {
    fn drop(&mut self) {
        if let Some(item) = self.item.take()
            && let Err(e) = self.pool.sender.try_send(item)
        {
            tracing::warn!("PoolGuard drop failed to return item: {e}");
        }
    }
}

/// Owned counterpart to [`PoolGuard`] for `'static` contexts.
///
/// Holds a clone of the pool's sender so the item can be returned later via
/// [`OwnedReservation::checkin`].
pub struct OwnedReservation<T> {
    sender: async_channel::Sender<T>,
}

impl<T> OwnedReservation<T> {
    pub fn checkin(self, item: T) {
        if let Err(e) = self.sender.send_blocking(item) {
            tracing::warn!("OwnedReservation checkin failed (channel closed during shutdown): {e}");
        }
    }
}

/// RAII guard that auto-checks-in an item when dropped — used after [`PoolGuard::into_owned`].
///
/// Owns both the pooled item and the reservation needed to return it. Dropping
/// this guard automatically sends the item back to the pool.
pub struct CheckoutGuard<T> {
    item: Option<T>,
    reservation: Option<OwnedReservation<T>>,
}

impl<T> CheckoutGuard<T> {
    /// Take the inner item out of the guard. The caller becomes responsible for calling
    /// [`OwnedReservation::checkin`] manually. Prefer letting the guard drop instead.
    pub fn into_inner(mut self) -> (T, OwnedReservation<T>) {
        let item = self
            .item
            .take()
            .expect("CheckoutGuard::into_inner called after drop");
        let reservation = self
            .reservation
            .take()
            .expect("CheckoutGuard::into_inner called twice");
        (item, reservation)
    }
}

impl<T> Deref for CheckoutGuard<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.item
            .as_ref()
            .expect("CheckoutGuard accessed after into_inner")
    }
}

impl<T> DerefMut for CheckoutGuard<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.item
            .as_mut()
            .expect("CheckoutGuard accessed after into_inner")
    }
}

impl<T> Drop for CheckoutGuard<T> {
    fn drop(&mut self) {
        if let (Some(item), Some(reservation)) = (self.item.take(), self.reservation.take()) {
            reservation.checkin(item);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkout_guard_drop_returns_item_on_panic() {
        let pool = Pool::new(vec![42i32]);
        assert_eq!(pool.available(), 1);

        // Simulate what happens when spawn_blocking panics:
        // CheckoutGuard is dropped during unwinding and must return the item.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let guard = pool.try_checkout().unwrap();
            let owned = guard.into_owned();
            let _ = owned;
            panic!("simulated inference panic");
        }));

        assert!(result.is_err());
        // After panic unwinding, the item must have been returned to the pool
        // even though no manual checkin happened.
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_checkout_guard_drop_returns_item_without_panic() {
        let pool = Pool::new(vec![42i32]);
        assert_eq!(pool.available(), 1);

        {
            let guard = pool.try_checkout().unwrap();
            assert_eq!(pool.available(), 0);
            let owned = guard.into_owned();
            drop(owned);
        }

        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_exhaustion() {
        let pool = Pool::new(vec![1i32, 2i32]);
        assert_eq!(pool.available(), 2);

        let _g1 = pool.try_checkout().unwrap();
        let _g2 = pool.try_checkout().unwrap();
        assert_eq!(pool.available(), 0);

        assert!(pool.try_checkout().is_none());
    }

    #[tokio::test]
    async fn test_pool_close() {
        let pool = Pool::new(vec![42i32]);
        let _guard = pool.try_checkout().unwrap(); // drain the pool
        pool.close();
        assert!(matches!(pool.checkout().await, Err(PoolError::Closed)));
    }

    #[test]
    fn test_checkout_guard_into_inner() {
        let pool = Pool::new(vec![42i32]);
        let guard = pool.try_checkout().unwrap();
        let owned = guard.into_owned();
        let (item, reservation) = owned.into_inner();
        assert_eq!(item, 42);
        // Manual checkin
        reservation.checkin(item);
        assert_eq!(pool.available(), 1);
    }

    #[test]
    fn test_pool_guard_drop_checkin() {
        let pool = Pool::new(vec![42i32]);
        assert_eq!(pool.available(), 1);
        {
            let guard = pool.try_checkout().unwrap();
            assert_eq!(pool.available(), 0);
            drop(guard);
        }
        assert_eq!(pool.available(), 1);
    }
}
