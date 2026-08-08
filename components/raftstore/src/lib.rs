// Copyright 2016 TiKV Project Authors. Licensed under Apache-2.0.

#![cfg_attr(test, feature(test))]
#![feature(min_specialization)]
#![feature(box_patterns)]
#![feature(type_alias_impl_trait)]
#![feature(impl_trait_in_assoc_type)]
#![recursion_limit = "400"]
// `Instant` does not implement the "Ord" trait.
// So type `TtlRange` can't derive Ord trait directly.
#![allow(clippy::derive_ord_xor_partial_ord)]

#[cfg(test)]
extern crate test;
#[cfg(feature = "engine_rocks")]
pub mod compacted_event_sender;

pub mod coprocessor;
pub mod errors;
pub mod router;
pub mod store;
#[cfg(feature = "engine_rocks")]
pub use self::compacted_event_sender::RaftRouterCompactedEventSender;
pub use self::{
    coprocessor::{RegionInfo, RegionInfoAccessor, SeekRegionCallback},
    errors::{DiscardReason, Error, Result},
};

/// Force ThreadRng (rand 0.8, used by raftstore) to exhaust its 64 KiB
/// ReseedingRng cache and reseed from OsRng → our deterministic getrandom.
/// Call this before the deterministic phase so ThreadRng 0.8's internal
/// ChaCha state is freshly derived from the current seed.
#[cfg(feature = "dst")]
pub fn dst_force_thread_rng_reseed() {
    use rand::Rng;
    let mut sink = vec![0u8; 65_536];
    rand::thread_rng().fill(&mut sink[..]);
}

// `bytes::Bytes` is generated for `bytes` in protobuf.
pub fn bytes_capacity(b: &bytes::Bytes) -> usize {
    // NOTE: For deserialized raft messages, `len` equals capacity.
    // This is used to report memory usage to metrics.
    b.len()
}
