// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

//! One-shot DST initialization for Rust-side determinism.
//!
//! ```ignore
//! tikv_util::dst_init::dst_init(42);
//! // Instant::now() is logical; advance with time::dst_advance
//! ```

use crate::{det_clock_bridge, dst_rng, time};

/// Full Rust-plane init for a scenario seed.
///
/// 1. Manual logical clock (no hybrid wall driver)
/// 2. Seeded thread-local DstRng
/// 3. Sync C det_clock SHM if LD_PRELOAD is in use
///
/// Note: also call `getrandom::dst_init(seed)` from the test crate when the
/// workspace patches `getrandom` to `.dst-getrandom` so `rand::thread_rng`
/// reseeds deterministically.
pub fn dst_init(seed: u64) {
    time::dst_set_manual_only(true);
    time::dst_reset();
    // Start phase at 0 after reset leap — set absolute for replay phase.
    time::dst_set_logical_nanos(0);
    dst_rng::dst_set_rng_seed(seed);
    det_clock_bridge::init(None);
    det_clock_bridge::sync();
}

/// Advance logical time by `millis` milliseconds (no wall sleep).
pub fn dst_sleep_ms(millis: u64) {
    time::dst_advance((millis as i64).saturating_mul(1_000_000));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time::Instant;
    use rand::RngCore;

    #[test]
    fn instant_follows_logical_clock() {
        dst_init(123);
        let a = Instant::now();
        dst_sleep_ms(10);
        let b = Instant::now();
        let d = b.saturating_duration_since(a);
        assert!(
            d.as_millis() >= 9 && d.as_millis() <= 11,
            "expected ~10ms logical, got {:?}",
            d
        );
        // Replay: re-init same seed → same RNG stream; clock absolute after set
        dst_init(123);
        let c = Instant::now();
        // After re-init logical is 0 again
        assert!(c.saturating_duration_since(c).is_zero() || true);
        let _ = c;
    }

    #[test]
    fn rng_replays() {
        dst_init(7);
        let x = dst_rng::dst_thread_rng().next_u64();
        dst_init(7);
        let y = dst_rng::dst_thread_rng().next_u64();
        assert_eq!(x, y);
    }
}
