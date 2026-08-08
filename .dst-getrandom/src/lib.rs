//! Deterministic getrandom replacement for DST.

#![cfg_attr(not(feature = "std"), no_std)]

use core::mem::MaybeUninit;

/// Error type matching getrandom 0.3.x API.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Error {
    code: u32,
}

impl Error {
    pub const UNSUPPORTED: Error = Error { code: 1 };
    pub const UNEXPECTED: Error = Error { code: 2 };
    pub const INTERNAL: Error = Error { code: 3 };

    pub const fn code(&self) -> u32 {
        self.code
    }

    pub fn raw_os_error(&self) -> Option<i32> {
        None
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "getrandom error (code {})", self.code)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

// ── std implementation ──────────────────────────────────────────────

#[cfg(feature = "std")]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    static SEEDED: AtomicBool = AtomicBool::new(false);

    /// Global PRNG seed — set by `dst_init` / `dst_reseed`.
    /// Each `fill()` call derives its output purely from `(SEED, seq)`,
    /// making the Nth call always produce the same bytes for a given seed.
    static SEED: AtomicU64 = AtomicU64::new(0);
    /// Monotonic call counter — atomically incremented by every `fill()`.
    /// Reset to 0 by `dst_init` / `dst_reseed` so the call sequence restarts.
    static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

    /// Initialize (or reset) the deterministic RNG.
    ///
    /// Resets both the seed AND the call counter, so the very next `fill()`
    /// is always call #0 for this seed.  Call between scenario invocations
    /// in the same process to ensure deterministic replay.
    pub fn dst_init(seed: u64) {
        let s = if seed == 0 { 1 } else { seed };
        SEED.store(s, Ordering::SeqCst);
        CALL_SEQ.store(0, Ordering::SeqCst);
        SEEDED.store(true, Ordering::SeqCst);
    }

    /// Reset ONLY the call counter, keeping the same seed.
    /// Call this right before the deterministic phase (after bootstrap)
    /// then force a ThreadRng reseed by consuming 64 KiB.
    /// This ensures the first ThreadRng reseed gets call #0, making
    /// its internal ChaCha state identical across runs.
    pub fn dst_reseed() {
        CALL_SEQ.store(0, Ordering::SeqCst);
    }

    /// Alias for `dst_init` — reset seed + counter for within-process replay.
    pub fn dst_reset_rng(seed: u64) {
        dst_init(seed);
    }

    /// splitmix64 — high-quality, collision-free hash from a u64 state.
    /// Used to derive per-call PRNG output from `(seed, seq)`.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Deterministic fill: the Nth call (atomically numbered via `CALL_SEQ`)
    /// always produces the same bytes for a given seed, regardless of which
    /// thread calls it or how much `ThreadRng` has cached.
    ///
    /// This eliminates the Mutex contention and state-advance nondeterminism
    /// of the previous design.  The only requirement for full determinism is
    /// that the **call count** between resets is reproducible — which is
    /// guaranteed when `dst_reseed` is called right before the deterministic
    /// step-driven phase (see `dst_init` in the test harness).
    pub fn fill_impl(buf: &mut [u8]) -> Result<(), super::Error> {
        if SEEDED.load(Ordering::Relaxed) {
            let seq = CALL_SEQ.fetch_add(1, Ordering::SeqCst);
            let seed = SEED.load(Ordering::SeqCst);
            // Mix seed and seq into a per-call base state.
            let mut state = seed.wrapping_add(seq.wrapping_mul(0x9E37_79B9_7F4A_7C15));
            for chunk in buf.chunks_mut(8) {
                let val = splitmix64(&mut state);
                for (i, byte) in chunk.iter_mut().enumerate() {
                    *byte = ((val >> (i * 8)) & 0xFF) as u8;
                }
            }
            Ok(())
        } else {
            use std::io::Read;
            let mut file = std::fs::File::open("/dev/urandom")
                .map_err(|_| super::Error::UNEXPECTED)?;
            file.read_exact(buf).map_err(|_| super::Error::UNEXPECTED)?;
            Ok(())
        }
    }
}

#[cfg(not(feature = "std"))]
mod imp {
    pub fn dst_init(_seed: u64) {}

    pub fn fill_impl(_buf: &mut [u8]) -> Result<(), super::Error> {
        Err(super::Error::UNSUPPORTED)
    }
}

/// Initialize the deterministic RNG with a seed.
pub fn dst_init(seed: u64) {
    imp::dst_init(seed);
}

/// Reset the call counter (keep seed). Call after bootstrap, before deterministic phase.
#[cfg(feature = "std")]
pub fn dst_reseed() {
    imp::dst_reseed();
}

/// Reset the global PRNG to a fresh seed (within-process replay).
#[cfg(feature = "std")]
pub fn dst_reset_rng(seed: u64) {
    imp::dst_reset_rng(seed);
}

/// Generate a deterministic u64 from the global PRNG (for direct use
/// by test harness code that wants to avoid ThreadRng caching).
#[cfg(feature = "std")]
pub fn dst_random_u64() -> u64 {
    let mut buf = [0u8; 8];
    fill(&mut buf).expect("getrandom fill failed");
    u64::from_ne_bytes(buf)
}

/// Fill `dest` with random bytes (matches getrandom 0.3.x API).
pub fn fill(dest: &mut [u8]) -> Result<(), Error> {
    imp::fill_impl(dest)
}

/// Fill uninit buffer (matches getrandom 0.3.x API).
pub fn fill_uninit(dest: &mut [MaybeUninit<u8>]) -> Result<&mut [u8], Error> {
    let ptr: *mut u8 = dest.as_mut_ptr().cast();
    let len = dest.len();
    let buf = unsafe { core::slice::from_raw_parts_mut(ptr, len) };
    fill(buf)?;
    Ok(unsafe { core::slice::from_raw_parts_mut(ptr, len) })
}

/// Generate a u32 (matches getrandom 0.3.x API).
pub fn u32() -> Result<u32, Error> {
    let mut buf = [0u8; 4];
    fill(&mut buf)?;
    Ok(u32::from_ne_bytes(buf))
}

/// Generate a u64 (matches getrandom 0.3.x API).
pub fn u64() -> Result<u64, Error> {
    let mut buf = [0u8; 8];
    fill(&mut buf)?;
    Ok(u64::from_ne_bytes(buf))
}
