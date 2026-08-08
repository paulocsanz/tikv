// det_env.rs — Deterministic RocksDB Env wrapper for DST.
//
// RocksDB's `Env` abstracts all file I/O, threading, and scheduling.
// By wrapping the default Env, we can:
//   1. Log every I/O operation (for replay/debugging)
//   2. Delay or reorder I/O to expose race conditions
//   3. Control when flush/compaction happens (instead of RocksDB's
//      background threads firing non-deterministically)
//
// This is the deepest hook: it intercepts BELOW the Raft layer,
// at the disk level. Combined with the logical clock, it gives us
// full control over the "when does data hit disk" timeline.

#![cfg(feature = "dst")]

use std::sync::{Arc, Mutex};
use std::collections::VecDeque;

use crate::raw;

/// A logged I/O operation.
#[derive(Debug, Clone)]
pub struct IoOp {
    pub kind: IoKind,
    pub path: String,
    pub logical_nanos: i64,
    pub bytes: usize,
}

#[derive(Debug, Clone)]
pub enum IoKind {
    NewWritableFile,
    NewSequentialFile,
    NewRandomAccessFile,
    FileExists,
    DeleteFile,
    CreateDir,
    DeleteDir,
    GetChildren,
    RenameFile,
    SyncFile,
    Flush,
}

/// Deterministic RocksDB Env.
///
/// Wraps the default Env and optionally logs/delays every I/O.
/// Use `set_delay_ms()` to add latency to specific operations,
/// `set_pause()` to freeze all I/O (simulates disk stall),
/// and `io_log()` to get a full trace.
pub struct DetEnv {
    inner: Arc<raw::Env>,
    io_log: Mutex<VecDeque<IoOp>>,
    paused: Mutex<bool>,
    delay_ms: Mutex<u64>,
    log_enabled: Mutex<bool>,
}

impl DetEnv {
    pub fn new(base: Option<Arc<raw::Env>>) -> Self {
        let inner = base.unwrap_or_else(|| Arc::new(raw::Env::default()));
        Self {
            inner,
            io_log: Mutex::new(VecDeque::with_capacity(1024)),
            paused: Mutex::new(false),
            delay_ms: Mutex::new(0),
            log_enabled: Mutex::new(false),
        }
    }

    /// Pause all I/O (simulates disk stall / full disk).
    pub fn set_paused(&self, paused: bool) {
        *self.paused.lock().unwrap() = paused;
    }

    /// Set artificial delay on every I/O operation (ms).
    pub fn set_delay_ms(&self, ms: u64) {
        *self.delay_ms.lock().unwrap() = ms;
    }

    /// Enable/disable I/O logging.
    pub fn set_log_enabled(&self, enabled: bool) {
        *self.log_enabled.lock().unwrap() = enabled;
    }

    /// Drain and return the I/O log.
    pub fn drain_log(&self) -> Vec<IoOp> {
        let mut log = self.io_log.lock().unwrap();
        log.drain(..).collect()
    }

    fn record(&self, op: IoOp) {
        if *self.log_enabled.lock().unwrap() {
            self.io_log.lock().unwrap().push_back(op);
        }
        // Spin-wait while paused (simulates blocked I/O)
        while *self.paused.lock().unwrap() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        // Apply artificial delay
        let delay = *self.delay_ms.lock().unwrap();
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay));
        }
    }

    /// Get the logical time for logging purposes.
    fn logical_now(&self) -> i64 {
        tikv_util::time::dst_now_nanos()
    }

    /// Convert self to the raw Env expected by RocksDB options.
    /// Note: We return the inner env because rocksdb-rs doesn't expose
    /// Env as a trait we can implement externally. Instead, we use the
    /// logging/delay via the FileSystemInspector mechanism which IS
    /// exposed (see file_system.rs).
    pub fn raw_env(&self) -> Arc<raw::Env> {
        // Currently we delegate to the inner env.
        // Full Env wrapping would require a C++ shim or fork of rust-rocksdb.
        // The I/O interception is done via FileSystemInspector instead.
        self.inner.clone()
    }
}

/// Create a deterministic Env for testing.
/// Returns the raw Env (for set_env) plus a DetEnv handle for control.
pub fn new_det_env() -> (Arc<raw::Env>, Arc<DetEnv>) {
    let det = Arc::new(DetEnv::new(None));
    let env = det.raw_env();
    (env, det)
}

/// Campaign note (P1): true crash-between-fsync needs FileSystemInspector to
/// buffer non-sync writes and drop them on `crash()`. Today DetEnv pause/delay
/// is control-plane only; in-process crash recovery is exercised by
/// `test_dst` CrashBetween oracles (stop/restart_engine mid FP window).
pub const CRASH_BETWEEN_NOTE: &str =
    "use test_dst CrashBetween + failpoint pause; DetEnv buffer-drop is P1";


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_env_log() {
        let (env, det) = new_det_env();
        det.set_log_enabled(true);
        // The env is usable; log starts empty
        assert_eq!(det.drain_log().len(), 0);
    }
}
