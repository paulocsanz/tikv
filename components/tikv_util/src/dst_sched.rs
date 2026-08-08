// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.

//! Cooperative multi-worker scheduler for DST (Peça 2 building block).
//!
//! OS `thread::spawn` interleaving is nondeterministic. This module runs
//! N worker queues under an explicit interleaving π derived from a seed —
//! same seed ⇒ same step order (bit-stable).
//!
//! Pair with `dst_init` + `dst_tick` so timers fire between steps.

use std::collections::VecDeque;

/// Which worker runs at this step.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Step {
    pub worker: usize,
}

/// SplitMix64 — local, no external rng dep beyond seed.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Build interleaving of `ops_per_worker` work items.
pub fn schedule_from_seed(seed: u64, ops_per_worker: &[usize]) -> Vec<Step> {
    let n = ops_per_worker.len();
    let mut rem = ops_per_worker.to_vec();
    let total: usize = rem.iter().sum();
    let mut state = seed ^ 0x5CED_5CED_5CED_5CED;
    let mut out = Vec::with_capacity(total);
    for _ in 0..total {
        let cands: Vec<usize> = (0..n).filter(|&w| rem[w] > 0).collect();
        let pick = (splitmix(&mut state) as usize) % cands.len();
        let w = cands[pick];
        rem[w] -= 1;
        out.push(Step { worker: w });
    }
    out
}

/// Run schedule: each worker has a queue; at each step pop one op and exec.
pub fn run_interleaved<T, F>(schedule: &[Step], queues: &mut [VecDeque<T>], mut exec: F)
where
    F: FnMut(usize, T),
{
    for s in schedule {
        if let Some(op) = queues[s.worker].pop_front() {
            exec(s.worker, op);
        }
    }
}

/// Convenience: distribute `ops` round-robin then re-interleave by seed.
pub fn run_seeded<T: Clone, F>(seed: u64, n_workers: usize, ops: &[T], mut exec: F)
where
    F: FnMut(usize, T),
{
    let mut counts = vec![0usize; n_workers];
    for i in 0..ops.len() {
        counts[i % n_workers] += 1;
    }
    let sched = schedule_from_seed(seed, &counts);
    let mut queues: Vec<VecDeque<T>> = (0..n_workers).map(|_| VecDeque::new()).collect();
    for (i, op) in ops.iter().enumerate() {
        queues[i % n_workers].push_back(op.clone());
    }
    run_interleaved(&sched, &mut queues, |w, op| exec(w, op));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schedule_bitstable() {
        let c = [3usize, 5, 2];
        assert_eq!(schedule_from_seed(7, &c), schedule_from_seed(7, &c));
        assert_eq!(schedule_from_seed(7, &c).len(), 10);
    }

    #[test]
    fn interleaved_order_deterministic() {
        let ops: Vec<u32> = (0..12).collect();
        let mut log1 = Vec::new();
        let mut log2 = Vec::new();
        run_seeded(99, 3, &ops, |w, op| log1.push((w, op)));
        run_seeded(99, 3, &ops, |w, op| log2.push((w, op)));
        assert_eq!(log1, log2);
        assert_eq!(log1.len(), 12);
    }
}
