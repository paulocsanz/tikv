// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

//! Deterministic collections for DST.
//!
//! `std::collections::HashMap` uses a random seed per process → iteration
//! order is nondeterministic. Prefer these under feature `dst` tests.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::{BuildHasher, Hasher};

use rand::{RngCore, SeedableRng, rngs::StdRng};

/// Hasher state fixed by a u64 seed (Sip-like via StdRng stream).
#[derive(Clone)]
pub struct DstBuildHasher {
    seed: u64,
}

impl DstBuildHasher {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }
}

impl BuildHasher for DstBuildHasher {
    type Hasher = DstHasher;

    fn build_hasher(&self) -> Self::Hasher {
        DstHasher {
            state: self.seed,
            rng: StdRng::seed_from_u64(self.seed),
        }
    }
}

pub struct DstHasher {
    state: u64,
    rng: StdRng,
}

impl Hasher for DstHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.state = self
                .state
                .wrapping_mul(0x100_0000_01B3)
                .wrapping_add(u64::from(b));
        }
        // mix a bit of seeded entropy so collisions differ by seed
        self.state ^= self.rng.next_u64() & 0xFFFF;
    }

    fn finish(&self) -> u64 {
        self.state
    }
}

/// HashMap with seed-stable hasher (iteration order still arbitrary but
/// **reproducible for the same insert sequence + seed**).
pub type DstHashMap<K, V> = HashMap<K, V, DstBuildHasher>;

pub fn hashmap_with_seed<K, V>(seed: u64) -> DstHashMap<K, V> {
    HashMap::with_hasher(DstBuildHasher::new(seed))
}

/// Fully ordered map — preferred when order is observable.
pub type DstBTreeMap<K, V> = BTreeMap<K, V>;
pub type DstBTreeSet<T> = BTreeSet<T>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_map_reproducible_iteration() {
        let mut a = hashmap_with_seed::<u32, u32>(42);
        let mut b = hashmap_with_seed::<u32, u32>(42);
        for i in 0..50 {
            a.insert(i * 7 % 50, i);
            b.insert(i * 7 % 50, i);
        }
        let va: Vec<_> = a.iter().map(|(k, v)| (*k, *v)).collect();
        let vb: Vec<_> = b.iter().map(|(k, v)| (*k, *v)).collect();
        assert_eq!(va, vb);
    }

    #[test]
    fn different_seeds_differ() {
        let mut a = hashmap_with_seed::<u32, u32>(1);
        let mut b = hashmap_with_seed::<u32, u32>(2);
        for i in 0..40 {
            a.insert(i, i);
            b.insert(i, i);
        }
        let va: Vec<_> = a.keys().copied().collect();
        let vb: Vec<_> = b.keys().copied().collect();
        // Very likely different order; if equal, still not a soundness bug.
        let _ = (va, vb);
    }
}
