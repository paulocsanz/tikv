// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

//! One-shot DST initialization for the pure-Rust determinism plane.
//!
//! Closes (when tests enable `tikv_util/dst` + workspace getrandom patch):
//! - time: `Instant` → logical nanos
//! - entropy: `getrandom` 0.3 + `dst_rng` + ThreadRng reseed
//! - C clocks: `/tmp/dst_clock` via det_clock_bridge
//!
//! Still open for full process freeze: OS thread schedule, HashMap order
//! (use `dst_collections`), non-patched getrandom 0.2 call paths.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::{det_clock_bridge, dst_rng, time};

/// Bumped on each `dst_init` so env-based getrandom 0.1/0.2 shims (if any)
/// can reset their counters via `DST_RNG_RESET`.
static DST_RESET_NONCE: AtomicU64 = AtomicU64::new(0);

/// Full scenario init. Prefer this at the start of every DST test case.
pub fn dst_init(seed: u64) {
    // Manual logical clock (tests drive time with dst_advance / dst_sleep_ms).
    time::dst_set_manual_only(true);
    time::dst_set_step(1_000_000); // 1ms step unit
    time::dst_reset();
    time::dst_set_logical_nanos(0);

    dst_rng::dst_set_rng_seed(seed);

    det_clock_bridge::init(None);
    det_clock_bridge::sync();
    if std::env::var_os("DET_CLOCK_SHM").is_none() {
        // SAFETY: test-only env setup; process-wide and expected under dst.
        unsafe {
            std::env::set_var("DET_CLOCK_SHM", det_clock_bridge::shm_path());
        }
    }

    let nonce = DST_RESET_NONCE.fetch_add(1, Ordering::SeqCst) + 1;
    unsafe {
        std::env::set_var("DST_RNG_SEED", seed.to_string());
        std::env::set_var("DST_RNG_RESET", nonce.to_string());
    }

    // Workspace-patched getrandom 0.3 (rand 0.9). Safe even if unused.
    getrandom::dst_init(seed);
    force_thread_rng_reseed();
    getrandom::dst_reseed();
}

/// After bootstrap, call before the deterministic phase so CALL_SEQ = 0
/// and ThreadRng has not cached post-bootstrap entropy.
pub fn dst_reseed_before_phase() {
    let nonce = DST_RESET_NONCE.fetch_add(1, Ordering::SeqCst) + 1;
    unsafe {
        std::env::set_var("DST_RNG_RESET", nonce.to_string());
    }
    getrandom::dst_reseed();
    force_thread_rng_reseed();
    getrandom::dst_reseed();
    let nonce2 = DST_RESET_NONCE.fetch_add(1, Ordering::SeqCst) + 1;
    unsafe {
        std::env::set_var("DST_RNG_RESET", nonce2.to_string());
    }
}

/// Advance logical time by `millis` ms (no wall sleep).
pub fn dst_sleep_ms(millis: u64) {
    time::dst_advance((millis as i64).saturating_mul(1_000_000));
    // Wake SteadyTimer under dst so delays fire before next test step.
    crate::timer::dst_process_timers();
}

/// Advance by one logical step (default 1ms) and process timers.
pub fn dst_tick() {
    time::dst_step(1);
    crate::timer::dst_process_timers();
}

fn force_thread_rng_reseed() {
    // rand 0.8 ReseedingRng reseeds every 64 KiB from OsRng → getrandom.
    use rand::Rng;
    let mut sink = vec![0u8; 65_536];
    rand::thread_rng().fill(&mut sink[..]);
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
    }

    #[test]
    fn rng_and_getrandom_replay() {
        dst_init(7);
        let x = dst_rng::dst_thread_rng().next_u64();
        let mut buf_a = [0u8; 16];
        getrandom::fill(&mut buf_a).unwrap();

        dst_init(7);
        let y = dst_rng::dst_thread_rng().next_u64();
        let mut buf_b = [0u8; 16];
        getrandom::fill(&mut buf_b).unwrap();

        assert_eq!(x, y);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn reseed_before_phase_resets_call_seq() {
        dst_init(99);
        let mut a = [0u8; 8];
        getrandom::fill(&mut a).unwrap();
        dst_reseed_before_phase();
        let mut b = [0u8; 8];
        getrandom::fill(&mut b).unwrap();
        // After reseed, seq restarts at 0 with same seed → same first fill
        // as after dst_init — but ThreadRng reseed consumed fills.
        // Direct fill after reseed should match first fill after fresh init.
        dst_init(99);
        dst_reseed_before_phase();
        let mut c = [0u8; 8];
        getrandom::fill(&mut c).unwrap();
        assert_eq!(b, c);
    }
}
