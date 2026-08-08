// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

//! Deterministic RNG for DST mode.
//!
//! Under feature `dst`, tests should seed this at case start and use
//! `dst_thread_rng()` instead of `rand::thread_rng()`.
//!
//! Note: libc `getrandom` is also intercepted by `det_clock` (LD_PRELOAD),
//! which covers C++ code paths (RocksDB hash seed, etc.).

use std::cell::RefCell;

use rand::{RngCore, SeedableRng, rngs::StdRng};

/// Deterministic RNG backed by `StdRng` (ChaCha-based, seedable).
pub struct DstRng {
    inner: StdRng,
}

impl DstRng {
    pub fn seed_from_u64(seed: u64) -> Self {
        Self {
            inner: StdRng::seed_from_u64(seed),
        }
    }
}

impl RngCore for DstRng {
    fn next_u32(&mut self) -> u32 {
        self.inner.next_u32()
    }
    fn next_u64(&mut self) -> u64 {
        self.inner.next_u64()
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest)
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        self.inner.try_fill_bytes(dest)
    }
}

thread_local! {
    static THREAD_RNG: RefCell<DstRng> = RefCell::new(DstRng::seed_from_u64(0xDEAD_BEEF_CAFE_BABE));
}

/// Set the thread-local RNG seed. Call at the start of each DST test case.
pub fn dst_set_rng_seed(seed: u64) {
    THREAD_RNG.with(|r| {
        *r.borrow_mut() = DstRng::seed_from_u64(seed);
    });
}

/// Drop-in replacement for `rand::thread_rng()` under DST.
pub fn dst_thread_rng() -> DstRngThreadLocal {
    DstRngThreadLocal
}

pub struct DstRngThreadLocal;

impl RngCore for DstRngThreadLocal {
    fn next_u32(&mut self) -> u32 {
        THREAD_RNG.with(|r| r.borrow_mut().next_u32())
    }
    fn next_u64(&mut self) -> u64 {
        THREAD_RNG.with(|r| r.borrow_mut().next_u64())
    }
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        THREAD_RNG.with(|r| r.borrow_mut().fill_bytes(dest))
    }
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand::Error> {
        THREAD_RNG.with(|r| r.borrow_mut().try_fill_bytes(dest))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic() {
        let mut a = DstRng::seed_from_u64(42);
        let mut b = DstRng::seed_from_u64(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn test_thread_local() {
        dst_set_rng_seed(99);
        let v1 = dst_thread_rng().next_u64();
        dst_set_rng_seed(99);
        let v2 = dst_thread_rng().next_u64();
        assert_eq!(v1, v2);
    }
}
