// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.
//
// Virtual Raft network for pure-Rust DST: buffer → deterministic sort →
// controlled release. Used with batch_system::drive_once / step_all_once so
// poller production and message delivery share one schedule on the test thread.

#![cfg(feature = "dst")]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use kvproto::raft_serverpb::RaftMessage;
use raftstore::Result;

use crate::transport_simulate::{Filter, check_messages};

const NET_SEED_TAG: u64 = 0x4e45_54; // 'NET'

/// eraftpb::MessageType discriminants used for fingerprint summaries.
pub const MSG_HUP: i32 = 0;
pub const MSG_APP: i32 = 3;
pub const MSG_APP_RESP: i32 = 4;
pub const MSG_HEARTBEAT: i32 = 8;
pub const MSG_HEARTBEAT_RESP: i32 = 9;

/// Simple xorshift64 — local, no extra deps.
#[derive(Clone, Debug)]
pub struct SeededRng {
    state: u64,
}

impl SeededRng {
    pub fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        (self.state >> 32) as u32
    }

    /// Returns a value in [0, max).
    pub fn gen_range(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        self.next_u32() % max
    }
}

/// Deterministic sort key for a RaftMessage.
pub fn msg_sort_key(m: &RaftMessage) -> (u64, u64, u64, i32, u64, u64) {
    (
        m.get_from_peer().get_store_id(),
        m.get_to_peer().get_store_id(),
        m.get_region_id(),
        m.get_message().get_msg_type() as i32,
        m.get_message().get_term(),
        m.get_message().get_index(),
    )
}

fn log_entry_msg_type(entry: &str) -> Option<i32> {
    let body = entry.strip_prefix("DROP:").unwrap_or(entry);
    // format: from>to:region:msgtype:term:index
    let parts: Vec<&str> = body.split(':').collect();
    if parts.len() < 3 {
        return None;
    }
    parts[2].parse().ok()
}

/// Heartbeats + Hup — wall/timer driven noise.
pub fn is_noise_log_entry(entry: &str) -> bool {
    matches!(
        log_entry_msg_type(entry),
        Some(MSG_HUP) | Some(MSG_HEARTBEAT) | Some(MSG_HEARTBEAT_RESP)
    )
}

/// Client replication path only.
pub fn is_app_path_log_entry(entry: &str) -> bool {
    matches!(
        log_entry_msg_type(entry),
        Some(MSG_APP) | Some(MSG_APP_RESP)
    )
}

/// Adversarial reorder mode — perturbs the canonical sort key with a
/// seed-derived salt so that different seeds explore different delivery orders.
#[derive(Clone, Copy, Debug)]
pub enum ReorderMode {
    /// Always sort by canonical key (default, backward-compatible).
    Canonical,
    /// Sort by a seed-salted hash of the canonical key.
    Adversarial(u64),
    /// Deliver in reverse canonical order.
    Reverse,
}

impl Default for ReorderMode {
    fn default() -> Self {
        ReorderMode::Canonical
    }
}

/// Deterministic hash for adversarial reorder salt.
fn salt_hash(salt: u64, key: &(u64, u64, u64, i32, u64, u64)) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64 ^ salt;
    for &word in &[key.0, key.1, key.2, key.3 as u64, key.4, key.5] {
        h ^= word;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// One buffered message with optional remaining delay steps.
struct Buffered {
    msg: RaftMessage,
    /// Steps until eligible for `take_sorted` (0 = ready).
    delay: u32,
}

/// Virtual network: buffer outbound Raft messages, sort by deterministic key,
/// release in controlled batches. Optional delivery log for fingerprints.
///
/// Delay plane: each inbound message is assigned a seed-stable delay in
/// `0..=max_delay` **drive steps**. Call [`tick_delays`] once per outer
/// schedule step; [`take_sorted`] never decrements — only releases ready
/// (`delay == 0`) messages. Coupling delay to every `take_sorted` made the
/// delay clock depend on how many flush rounds had ready msgs (nondeterministic
/// under dual-run residual).
#[derive(Clone)]
pub struct DstNetworkQueue {
    buffer: Arc<Mutex<Vec<Buffered>>>,
    /// Auto-release every `batch_size` `before()` calls (0 = hold until step).
    batch_size: usize,
    call_count: Arc<AtomicUsize>,
    /// Log of released messages: "from>to:region:msgtype:term:index"
    log: Arc<Mutex<Vec<String>>>,
    drop_rate_pct: u32,
    /// Max seed-stable delay steps assigned on ingress (0 = no delay).
    max_delay: u32,
    /// Probability (0..=100) of duplicating each released message.
    dup_rate_pct: u32,
    /// Reorder mode for delivery schedule.
    reorder: ReorderMode,
    /// Partitioned peer pairs: messages between these (from, to) store IDs
    /// are silently dropped. Simulates network partitions.
    partitioned: Arc<Mutex<Vec<(u64, u64)>>>,
    rng: Arc<Mutex<SeededRng>>,
    pending_release: Arc<Mutex<Vec<RaftMessage>>>,
    passthrough: Arc<AtomicBool>,
    recording: Arc<AtomicBool>,
}

impl DstNetworkQueue {
    pub fn new(seed: u64, batch_size: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(Vec::new())),
            batch_size,
            call_count: Arc::new(AtomicUsize::new(0)),
            log: Arc::new(Mutex::new(Vec::new())),
            drop_rate_pct: 0,
            max_delay: 0,
            dup_rate_pct: 0,
            reorder: ReorderMode::default(),
            partitioned: Arc::new(Mutex::new(Vec::new())),
            rng: Arc::new(Mutex::new(SeededRng::new(seed.wrapping_add(NET_SEED_TAG)))),
            pending_release: Arc::new(Mutex::new(Vec::new())),
            passthrough: Arc::new(AtomicBool::new(false)),
            recording: Arc::new(AtomicBool::new(true)),
        }
    }

    pub fn with_drop_rate(mut self, pct: u32) -> Self {
        self.drop_rate_pct = pct.min(100);
        self
    }

    /// Seed-stable per-message delay in `0..=max` steps before release.
    pub fn with_max_delay(mut self, max: u32) -> Self {
        self.max_delay = max;
        self
    }

    /// Seed-stable probability of duplicating each released message.
    pub fn with_dup_rate(mut self, pct: u32) -> Self {
        self.dup_rate_pct = pct.min(100);
        self
    }

    /// Set the adversarial reorder mode for delivery schedule.
    pub fn with_reorder(mut self, mode: ReorderMode) -> Self {
        self.reorder = mode;
        self
    }

    /// Add a partition between two peers (bidirectional). Messages between
    /// these store IDs are silently dropped. Can be cleared with `clear_partitions`.
    pub fn add_partition(&self, from: u64, to: u64) {
        let mut p = self.partitioned.lock().unwrap();
        p.push((from, to));
        p.push((to, from));
    }

    /// Remove all partitions.
    pub fn clear_partitions(&self) {
        self.partitioned.lock().unwrap().clear();
    }

    /// Check if a message's (from, to) is partitioned.
    fn is_partitioned(&self, from: u64, to: u64) -> bool {
        let p = self.partitioned.lock().unwrap();
        p.contains(&(from, to))
    }

    pub fn set_passthrough(&self, on: bool) {
        self.passthrough.store(on, Ordering::SeqCst);
    }

    pub fn set_recording(&self, on: bool) {
        self.recording.store(on, Ordering::SeqCst);
    }

    pub fn clear_log(&self) {
        self.log.lock().unwrap().clear();
    }

    /// Advance the delay clock by one step. Call **once** per outer schedule
    /// iteration (e.g. each `put_stepped` poll round), not on every flush.
    pub fn tick_delays(&self) {
        if self.max_delay == 0 {
            return;
        }
        let mut buffer = self.buffer.lock().unwrap();
        for b in buffer.iter_mut() {
            if b.delay > 0 {
                b.delay -= 1;
            }
        }
    }

    /// Take up to `n` **ready** buffered messages in sort order.
    ///
    /// Only messages with `delay == 0` are eligible. Does **not** tick delays
    /// — use [`tick_delays`] on the schedule boundary.
    pub fn take_sorted(&self, n: usize) -> Vec<RaftMessage> {
        let mut buffer = self.buffer.lock().unwrap();
        // Partition ready vs delayed without reordering delayed relative order.
        let mut ready: Vec<RaftMessage> = Vec::new();
        let mut delayed: Vec<Buffered> = Vec::new();
        for b in buffer.drain(..) {
            if b.delay == 0 {
                ready.push(b.msg);
            } else {
                delayed.push(b);
            }
        }
        match self.reorder {
            ReorderMode::Canonical => {
                ready.sort_by(|a, b| msg_sort_key(a).cmp(&msg_sort_key(b)));
            }
            ReorderMode::Reverse => {
                ready.sort_by(|a, b| msg_sort_key(a).cmp(&msg_sort_key(b)));
                ready.reverse();
            }
            ReorderMode::Adversarial(salt) => {
                ready.sort_by(|a, b| {
                    let ka = salt_hash(salt, &msg_sort_key(a));
                    let kb = salt_hash(salt, &msg_sort_key(b));
                    ka.cmp(&kb)
                });
            }
        }
        let take = n.min(ready.len());
        let out: Vec<_> = ready.drain(..take).collect();
        // Put unreleased ready msgs back with delay 0 (still eligible next time).
        for m in ready {
            delayed.push(Buffered { msg: m, delay: 0 });
        }
        // Apply duplication: seed-stable dup of released messages.
        let out = if self.dup_rate_pct > 0 {
            let mut rng = self.rng.lock().unwrap();
            let mut expanded = Vec::with_capacity(out.len() * 2);
            for m in out {
                expanded.push(m.clone());
                if rng.gen_range(100) < self.dup_rate_pct {
                    expanded.push(m);
                }
            }
            expanded
        } else {
            out
        };
        *buffer = delayed;
        out
    }

    pub fn pending(&self) -> usize {
        self.buffer.lock().unwrap().len()
    }

    /// Messages still waiting on delay > 0.
    pub fn delayed_pending(&self) -> usize {
        self.buffer
            .lock()
            .unwrap()
            .iter()
            .filter(|b| b.delay > 0)
            .count()
    }

    pub fn delivery_log(&self) -> Vec<String> {
        self.log.lock().unwrap().clone()
    }

    pub fn log_summary_no_hb(&self) -> String {
        self.summarize_log(|e| !is_noise_log_entry(e))
    }

    pub fn log_summary_app_only(&self) -> String {
        self.summarize_log(is_app_path_log_entry)
    }

    pub fn log_ops_sequence(&self) -> String {
        let log = self.log.lock().unwrap();
        log.iter()
            .filter(|e| is_app_path_log_entry(e))
            .map(|e| {
                e.rsplit_once(':')
                    .and_then(|(r, _)| r.rsplit_once(':'))
                    .map(|(r, _)| r.to_string())
                    .unwrap_or_else(|| e.clone())
            })
            .collect::<Vec<_>>()
            .join("|")
    }

    fn summarize_log(&self, keep: impl Fn(&str) -> bool) -> String {
        let log = self.log.lock().unwrap();
        let mut counts: BTreeMap<String, usize> = BTreeMap::new();
        for entry in log.iter() {
            if !keep(entry) {
                continue;
            }
            let key = if let Some(rest) = entry.strip_prefix("DROP:") {
                let base = rest
                    .rsplit_once(':')
                    .and_then(|(r, _)| r.rsplit_once(':'))
                    .map(|(r, _)| r)
                    .unwrap_or(rest);
                format!("DROP:{base}")
            } else {
                entry
                    .rsplit_once(':')
                    .and_then(|(r, _)| r.rsplit_once(':'))
                    .map(|(r, _)| r.to_string())
                    .unwrap_or_else(|| entry.clone())
            };
            *counts.entry(key).or_default() += 1;
        }
        counts
            .iter()
            .map(|(k, c)| format!("{k}x{c}"))
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Apply drop filter + partition filter + append delivery log; returns survivors.
    pub fn record_and_filter(&self, msgs: Vec<RaftMessage>) -> Vec<RaftMessage> {
        let mut out = Vec::with_capacity(msgs.len());
        let recording = self.recording.load(Ordering::SeqCst);
        let mut log = self.log.lock().unwrap();
        let mut rng = self.rng.lock().unwrap();
        for m in msgs {
            let from = m.get_from_peer().get_store_id();
            let to = m.get_to_peer().get_store_id();
            let entry = format!(
                "{}>{}:{}:{}:{}:{}",
                from,
                to,
                m.get_region_id(),
                m.get_message().get_msg_type() as i32,
                m.get_message().get_term(),
                m.get_message().get_index(),
            );
            // Partition check: silently drop messages between partitioned peers.
            if self.is_partitioned(from, to) {
                if recording {
                    log.push(format!("PART:{entry}"));
                }
                continue;
            }
            if self.drop_rate_pct > 0 && rng.gen_range(100) < self.drop_rate_pct {
                if recording {
                    log.push(format!("DROP:{entry}"));
                }
                continue;
            }
            if recording {
                log.push(entry);
            }
            out.push(m);
        }
        out
    }
}

impl Filter for DstNetworkQueue {
    fn before(&self, msgs: &mut Vec<RaftMessage>) -> Result<()> {
        if self.passthrough.load(Ordering::SeqCst) {
            return if msgs.is_empty() {
                Ok(())
            } else {
                check_messages(msgs)
            };
        }

        {
            let mut buffer = self.buffer.lock().unwrap();
            let mut rng = self.rng.lock().unwrap();
            for m in msgs.drain(..) {
                let delay = if self.max_delay > 0 {
                    rng.gen_range(self.max_delay + 1)
                } else {
                    0
                };
                buffer.push(Buffered { msg: m, delay });
            }
        }

        {
            let mut staged = self.pending_release.lock().unwrap();
            if !staged.is_empty() {
                msgs.extend(staged.drain(..));
            }
        }

        let count = self.call_count.fetch_add(1, Ordering::Relaxed);

        if self.batch_size > 0 {
            // Auto-release path: one delay tick per before() release, then flush.
            if (count + 1) % self.batch_size == 0 || self.pending() > 64 {
                self.tick_delays();
                let released = self.take_sorted(usize::MAX);
                msgs.extend(self.record_and_filter(released));
            }
        }

        if msgs.is_empty() {
            Ok(())
        } else {
            check_messages(msgs)
        }
    }
}

#[cfg(test)]
mod tests {
    use raft::eraftpb::MessageType;

    use super::*;

    fn make_msg(from: u64, to: u64, region: u64, term: u64, index: u64) -> RaftMessage {
        let mut m = RaftMessage::default();
        m.set_region_id(region);
        m.mut_from_peer().set_store_id(from);
        m.mut_to_peer().set_store_id(to);
        m.mut_message().set_msg_type(MessageType::MsgAppend);
        m.mut_message().set_term(term);
        m.mut_message().set_index(index);
        m
    }

    #[test]
    fn take_sorted_is_deterministic() {
        let net = DstNetworkQueue::new(0xABC, 0);
        // Inject via filter (hold mode).
        let mut batch = vec![
            make_msg(3, 1, 1, 5, 10),
            make_msg(1, 2, 1, 5, 1),
            make_msg(2, 3, 1, 5, 5),
            make_msg(1, 2, 1, 5, 2),
        ];
        let _ = net.before(&mut batch);
        assert!(batch.is_empty());
        assert_eq!(net.pending(), 4);

        let a = net.take_sorted(usize::MAX);
        assert_eq!(a.len(), 4);
        // Re-buffer same multiset via another queue with same seed ops.
        let net2 = DstNetworkQueue::new(0xABC, 0);
        let mut batch2 = vec![
            make_msg(3, 1, 1, 5, 10),
            make_msg(1, 2, 1, 5, 1),
            make_msg(2, 3, 1, 5, 5),
            make_msg(1, 2, 1, 5, 2),
        ];
        let _ = net2.before(&mut batch2);
        let b = net2.take_sorted(usize::MAX);
        let keys_a: Vec<_> = a.iter().map(msg_sort_key).collect();
        let keys_b: Vec<_> = b.iter().map(msg_sort_key).collect();
        assert_eq!(keys_a, keys_b);
        // Sorted by from,to,region,type,term,index
        assert_eq!(keys_a[0].0, 1);
        assert_eq!(keys_a[0].5, 1);
        assert_eq!(keys_a[1].0, 1);
        assert_eq!(keys_a[1].5, 2);
    }

    #[test]
    fn max_delay_is_seed_stable() {
        fn release_trace(seed: u64) -> Vec<u64> {
            let net = DstNetworkQueue::new(seed, 0).with_max_delay(3);
            let mut batch: Vec<_> = (0..6).map(|i| make_msg(1, 2, 1, 1, i)).collect();
            let _ = net.before(&mut batch);
            let mut indices = Vec::new();
            // tick once per step, then take ready — same as put_stepped schedule.
            for _ in 0..8 {
                net.tick_delays();
                for m in net.take_sorted(usize::MAX) {
                    indices.push(m.get_message().get_index());
                }
            }
            indices
        }
        assert_eq!(release_trace(77), release_trace(77));
        assert_eq!(release_trace(77).len(), 6);
        // Different seeds can differ (not required, but likely).
        let _ = release_trace(78);
    }

    #[test]
    fn take_sorted_does_not_tick_delays() {
        let net = DstNetworkQueue::new(1, 0).with_max_delay(2);
        // Force delay=2 by injecting until we see delayed_pending (rng may assign 0).
        // Direct: push via before many msgs; at least some delayed with max=2.
        let mut batch: Vec<_> = (0..20).map(|i| make_msg(1, 2, 1, 1, i)).collect();
        let _ = net.before(&mut batch);
        let delayed_before = net.delayed_pending();
        // Many take_sorted without tick must not free delayed msgs.
        for _ in 0..10 {
            let _ = net.take_sorted(usize::MAX);
        }
        assert_eq!(
            net.delayed_pending(),
            delayed_before,
            "take_sorted must not advance delay clock"
        );
        net.tick_delays();
        assert!(
            net.delayed_pending() <= delayed_before,
            "tick_delays should reduce or keep delayed count"
        );
    }

    #[test]
    fn drop_rate_is_seed_stable() {
        fn once(seed: u64) -> (Vec<String>, usize) {
            let net = DstNetworkQueue::new(seed, 0).with_drop_rate(40);
            let mut batch: Vec<_> = (0..20)
                .map(|i| make_msg(1, 2, 1, 1, i as u64))
                .collect();
            let _ = net.before(&mut batch);
            let out = net.record_and_filter(net.take_sorted(usize::MAX));
            (net.delivery_log(), out.len())
        }
        let (log1, n1) = once(99);
        let (log2, n2) = once(99);
        assert_eq!(log1, log2);
        assert_eq!(n1, n2);
        assert!(n1 < 20, "expected some drops at 40%, got {}", n1);
    }

    #[test]
    fn adversarial_reorder_is_seed_stable() {
        fn trace(salt: u64) -> Vec<u64> {
            let net = DstNetworkQueue::new(42, 0)
                .with_reorder(ReorderMode::Adversarial(salt));
            let mut batch: Vec<_> = (0..10)
                .map(|i| make_msg(1, 2, 1, 1, i))
                .collect();
            let _ = net.before(&mut batch);
            net.take_sorted(usize::MAX)
                .iter()
                .map(|m| m.get_message().get_index())
                .collect()
        }
        assert_eq!(trace(0xDEAD), trace(0xDEAD));
        assert_ne!(trace(0xDEAD), trace(0xBEEF));
    }

    #[test]
    fn reverse_reorder_flips_canonical() {
        let net = DstNetworkQueue::new(42, 0).with_reorder(ReorderMode::Reverse);
        let mut batch: Vec<_> = (0..5).map(|i| make_msg(1, 2, 1, 1, i)).collect();
        let _ = net.before(&mut batch);
        let out = net.take_sorted(usize::MAX);
        let indices: Vec<_> = out.iter().map(|m| m.get_message().get_index()).collect();
        assert_eq!(indices, vec![4, 3, 2, 1, 0]);
    }

    #[test]
    fn dup_rate_inflates_output() {
        let net = DstNetworkQueue::new(7, 0).with_dup_rate(100);
        let mut batch: Vec<_> = (0..5).map(|i| make_msg(1, 2, 1, 1, i)).collect();
        let _ = net.before(&mut batch);
        let out = net.take_sorted(usize::MAX);
        assert_eq!(out.len(), 10);
        let indices: std::collections::HashSet<_> = out
            .iter()
            .map(|m| m.get_message().get_index())
            .collect();
        assert_eq!(indices.len(), 5);
    }

    #[test]
    fn partition_drops_only_specified_peers() {
        let net = DstNetworkQueue::new(42, 0);
        net.add_partition(1, 2);
        // Messages from 1→2 and 2→1 should be dropped; 1→3 should pass.
        let mut batch = vec![
            make_msg(1, 2, 1, 1, 0), // partitioned
            make_msg(2, 1, 1, 1, 1), // partitioned
            make_msg(1, 3, 1, 1, 2), // survives
        ];
        let _ = net.before(&mut batch);
        let out = net.record_and_filter(net.take_sorted(usize::MAX));
        assert_eq!(out.len(), 1, "only 1→3 should survive partition");
        assert_eq!(out[0].get_to_peer().get_store_id(), 3);
    }

    #[test]
    fn clear_partitions_restores_delivery() {
        let net = DstNetworkQueue::new(42, 0);
        net.add_partition(1, 2);
        net.clear_partitions();
        let mut batch = vec![make_msg(1, 2, 1, 1, 0)];
        let _ = net.before(&mut batch);
        let out = net.record_and_filter(net.take_sorted(usize::MAX));
        assert_eq!(out.len(), 1, "message should pass after clear");
    }
}
