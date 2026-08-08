// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.
//
// Single-threaded cooperative executor for Deterministic Simulation Testing.
// Under feature `dst`, pollers are registered here instead of each getting an
// OS thread. One driver thread steps every poller in registration order
// (round-robin by id), so store/apply across all nodes share one schedule
// without a process-wide mutex held during `step()` (which deadlocks when
// store waits for apply).
//
// Manual drive mode: tests can pause the background driver and call
// `step_all_once()` on the test thread, then flush the virtual network on the
// same thread — fully serializing poller production + message delivery.

use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    thread,
    time::Duration,
};

use tikv_util::thread_group::{self, GroupProperties};

/// One cooperative step. Returns `false` when the poller has permanently
/// stopped (shutdown / channel disconnect) and should be removed.
pub trait Pollable: Send {
    fn step(&mut self) -> bool;
}

struct Entry {
    id: u64,
    poller: Box<dyn Pollable>,
    done: Arc<AtomicBool>,
    /// Thread-group properties captured at registration (required by TiKV
    /// code paths that call `is_shutdown(true)`).
    props: GroupProperties,
}

struct State {
    pollers: Vec<Entry>,
}

static STATE: Mutex<State> = Mutex::new(State {
    pollers: Vec::new(),
});
static NEXT_ID: AtomicU64 = AtomicU64::new(1);
static DRIVER_STARTED: AtomicBool = AtomicBool::new(false);
/// When true, the background driver idles and only `step_all_once` advances pollers.
static MANUAL_DRIVE: AtomicBool = AtomicBool::new(false);

/// Register a poller with the global single-threaded executor.
///
/// `done` is set to `true` when the poller exits permanently so callers can
/// join a lightweight waiter thread (keeps `BatchSystem::shutdown` unchanged).
pub fn register(poller: Box<dyn Pollable>, done: Arc<AtomicBool>) -> u64 {
    let id = NEXT_ID.fetch_add(1, Ordering::SeqCst);
    // Capture props from the registering thread (test/main). Fall back to a
    // fresh GroupProperties so the driver never panics on is_shutdown(true).
    let props = thread_group::current_properties().unwrap_or_default();
    {
        let mut state = STATE.lock().unwrap();
        // Keep sorted by id so step order is deterministic (registration order).
        let pos = state.pollers.partition_point(|e| e.id < id);
        state.pollers.insert(
            pos,
            Entry {
                id,
                poller,
                done,
                props,
            },
        );
    }
    ensure_driver();
    id
}

/// Force-remove a poller identified by its `done` flag (pointer equality).
pub fn force_stop(done: &Arc<AtomicBool>) {
    done.store(true, Ordering::SeqCst);
    let mut state = STATE.lock().unwrap();
    if let Some(i) = state
        .pollers
        .iter()
        .position(|e| Arc::ptr_eq(&e.done, done))
    {
        let entry = state.pollers.remove(i);
        drop(entry);
    }
}

/// Pause/resume the background driver. While manual, only `step_all_once`
/// advances pollers (call from the test thread, then flush the virtual net).
pub fn set_manual_drive(on: bool) {
    MANUAL_DRIVE.store(on, Ordering::SeqCst);
}

pub fn is_manual_drive() -> bool {
    MANUAL_DRIVE.load(Ordering::SeqCst)
}

/// Run one full round: every live poller once, in ascending registration id.
/// Returns how many pollers were stepped. Safe to call from the test thread
/// under `set_manual_drive(true)`.
pub fn step_all_once() -> usize {
    let ids: Vec<u64> = {
        let state = STATE.lock().unwrap();
        state.pollers.iter().map(|e| e.id).collect()
    };
    let mut stepped = 0usize;
    for id in ids {
        if step_one(id) {
            stepped += 1;
        }
    }
    stepped
}

/// Step a single poller by id. Returns true if it was found and stepped.
fn step_one(id: u64) -> bool {
    let entry = {
        let mut state = STATE.lock().unwrap();
        match state.pollers.iter().position(|e| e.id == id) {
            Some(i) => Some(state.pollers.remove(i)),
            None => None,
        }
    };
    let Some(mut entry) = entry else {
        return false;
    };
    if entry.done.load(Ordering::SeqCst) {
        return false;
    }

    thread_group::set_properties(Some(entry.props.clone()));
    let cont = entry.poller.step();

    let mut state = STATE.lock().unwrap();
    if cont && !entry.done.load(Ordering::SeqCst) {
        let pos = state.pollers.partition_point(|e| e.id < entry.id);
        state.pollers.insert(pos, entry);
    } else {
        entry.done.store(true, Ordering::SeqCst);
    }
    true
}

fn ensure_driver() {
    if DRIVER_STARTED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return;
    }
    thread::Builder::new()
        .name("dst-batch-driver".to_owned())
        .spawn(driver_loop)
        .expect("failed to spawn dst batch driver");
}

fn driver_loop() {
    loop {
        if MANUAL_DRIVE.load(Ordering::SeqCst) {
            // Test thread owns stepping via step_all_once().
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        let ids: Vec<u64> = {
            let state = STATE.lock().unwrap();
            state.pollers.iter().map(|e| e.id).collect()
        };

        if ids.is_empty() {
            thread::sleep(Duration::from_millis(1));
            continue;
        }

        for id in ids {
            step_one(id);
        }

        thread::sleep(Duration::from_millis(1));
    }
}

/// Number of live pollers (test/debug helper).
pub fn live_count() -> usize {
    STATE.lock().unwrap().pollers.len()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Global executor state is process-wide; isolate unit tests from each other
    /// and from any concurrent integration cases that may register pollers.
    fn take_over_for_test() {
        set_manual_drive(true);
        let mut state = STATE.lock().unwrap();
        for e in state.pollers.drain(..) {
            e.done.store(true, Ordering::SeqCst);
        }
    }

    struct CountingPoller {
        left: usize,
        hits: Arc<AtomicU64>,
    }

    impl Pollable for CountingPoller {
        fn step(&mut self) -> bool {
            self.hits.fetch_add(1, Ordering::SeqCst);
            if self.left == 0 {
                return false;
            }
            self.left -= 1;
            true
        }
    }

    #[test]
    fn manual_drive_steps_registered_pollers_to_completion() {
        take_over_for_test();
        let hits = Arc::new(AtomicU64::new(0));
        let done_a = Arc::new(AtomicBool::new(false));
        let done_b = Arc::new(AtomicBool::new(false));

        let id_a = register(
            Box::new(CountingPoller {
                left: 2,
                hits: Arc::clone(&hits),
            }),
            Arc::clone(&done_a),
        );
        let id_b = register(
            Box::new(CountingPoller {
                left: 2,
                hits: Arc::clone(&hits),
            }),
            Arc::clone(&done_b),
        );
        assert!(id_a < id_b);
        assert_eq!(live_count(), 2);

        // Three full rounds: each poller hits while left>0, then exits on left==0.
        // left=2 → hit+dec→1, hit+dec→0, hit+false → 3 hits each = 6 total.
        for _ in 0..4 {
            let _ = step_all_once();
        }

        assert_eq!(hits.load(Ordering::SeqCst), 6);
        assert!(done_a.load(Ordering::SeqCst));
        assert!(done_b.load(Ordering::SeqCst));
        assert_eq!(live_count(), 0);

        set_manual_drive(false);
    }

    #[test]
    fn force_stop_removes_live_poller() {
        take_over_for_test();
        let hits = Arc::new(AtomicU64::new(0));
        let done = Arc::new(AtomicBool::new(false));
        register(
            Box::new(CountingPoller {
                left: 100,
                hits: Arc::clone(&hits),
            }),
            Arc::clone(&done),
        );
        assert_eq!(live_count(), 1);
        force_stop(&done);
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(live_count(), 0);
        assert_eq!(step_all_once(), 0);
        set_manual_drive(false);
    }
}
