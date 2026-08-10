// det_env.rs — Deterministic RocksDB Env wrapper for DST.
//
// RocksDB's `Env` abstracts file I/O. TiKV's vendored `rocksdb` bindings
// expose exactly one hook into it usable from pure Rust without a new C++
// shim: `Env::new_file_system_inspected_env` + `engine_traits::
// FileSystemInspector` (`read`/`write`, byte-length granularity only -- no
// path, no operation kind). This is the *same* mechanism `file_system::
// get_env` already uses in production for I/O rate limiting.
//
// An earlier version of this file believed no such hook existed ("rocksdb-rs
// doesn't expose Env as a trait ... would require a C++ shim") and left
// `raw_env()` as a pure pass-through to the un-wrapped inner Env -- meaning
// every `set_delay_ms`/`set_paused`/log call below was silently a no-op:
// nothing this type did ever actually reached RocksDB's I/O path, and
// nothing in the codebase ever constructed it outside its own module. Fixed
// 2026-08-10: `DetEnv` now implements `FileSystemInspector` directly and is
// wired through the same shim `file_system::get_env` already uses.
//
// What this DOES give:
//   1. Log every read/write (byte length + logical time -- not path/kind,
//      that granularity genuinely isn't exposed by this hook)
//   2. Delay reads/writes to expose race conditions
//   3. Pause all I/O for one node (simulates a disk stall)
// What this does NOT give (would need the C++ shim work the old comment
// described): per-operation-kind interception (NewWritableFile vs
// DeleteFile vs SyncFile), path visibility, or making an operation fail --
// only length-preserving delay/pause on reads and writes.

#![cfg(feature = "dst")]

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use engine_traits::FileSystemInspector;

use crate::{file_system::WrappedFileSystemInspector, r2e, raw};

/// A logged I/O operation. Path/operation-kind are not observable through
/// the `FileSystemInspector` hook (see module doc) -- this is exactly what
/// RocksDB reports: a read or write of `bytes` length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoKind {
    Read,
    Write,
}

#[derive(Debug, Clone)]
pub struct IoOp {
    pub kind: IoKind,
    pub bytes: usize,
    pub logical_nanos: i64,
}

struct DetEnvState {
    io_log: Mutex<VecDeque<IoOp>>,
    paused: Mutex<bool>,
    delay_ms: Mutex<u64>,
    log_enabled: Mutex<bool>,
}

/// Deterministic RocksDB I/O control, wired via `FileSystemInspector`.
///
/// A cheap `Clone`-able handle (`Arc` internally, same idiom as
/// `DstNetworkQueue`) rather than something you wrap in `Arc` yourself --
/// `WrappedFileSystemInspector<T>` takes ownership of a `T: FileSystemInspector`
/// by value, and Rust's orphan rules forbid `impl ForeignTrait for
/// Arc<LocalType>` (`Arc` isn't `#[fundamental]`), so the sharing has to live
/// inside `DetEnv` itself, not around it.
///
/// Use `set_delay_ms()` to add latency to every read/write, `set_paused()`
/// to freeze all I/O (simulates a disk stall / full disk -- spin-waits
/// inside the hook, blocking whichever RocksDB thread issued the I/O,
/// exactly like a real stalled disk would), and `drain_log()` for a trace.
///
/// One `DetEnv` per node, not one per process: kv_db and raft_db already
/// share a single `Env` per node (`Config::build_shared_rocks_env`), so
/// wiring one `DetEnv` per node via [`get_det_env`] pauses/delays that
/// node's disk without touching the others'.
#[derive(Clone)]
pub struct DetEnv(Arc<DetEnvState>);

impl Default for DetEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl DetEnv {
    pub fn new() -> Self {
        Self(Arc::new(DetEnvState {
            io_log: Mutex::new(VecDeque::with_capacity(1024)),
            paused: Mutex::new(false),
            delay_ms: Mutex::new(0),
            log_enabled: Mutex::new(false),
        }))
    }

    /// Pause all I/O (simulates disk stall / full disk).
    pub fn set_paused(&self, paused: bool) {
        *self.0.paused.lock().unwrap() = paused;
    }

    /// Set artificial delay on every I/O operation (ms).
    pub fn set_delay_ms(&self, ms: u64) {
        *self.0.delay_ms.lock().unwrap() = ms;
    }

    /// Enable/disable I/O logging.
    pub fn set_log_enabled(&self, enabled: bool) {
        *self.0.log_enabled.lock().unwrap() = enabled;
    }

    /// Drain and return the I/O log.
    pub fn drain_log(&self) -> Vec<IoOp> {
        self.0.io_log.lock().unwrap().drain(..).collect()
    }

    fn record(&self, kind: IoKind, bytes: usize) {
        if *self.0.log_enabled.lock().unwrap() {
            self.0.io_log.lock().unwrap().push_back(IoOp {
                kind,
                bytes,
                logical_nanos: tikv_util::time::dst_now_nanos(),
            });
        }
        // Spin-wait while paused (simulates blocked I/O).
        while *self.0.paused.lock().unwrap() {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let delay = *self.0.delay_ms.lock().unwrap();
        if delay > 0 {
            std::thread::sleep(std::time::Duration::from_millis(delay));
        }
    }
}

impl FileSystemInspector for DetEnv {
    fn read(&self, len: usize) -> engine_traits::Result<usize> {
        self.record(IoKind::Read, len);
        Ok(len)
    }

    fn write(&self, len: usize) -> engine_traits::Result<usize> {
        self.record(IoKind::Write, len);
        Ok(len)
    }
}

/// Build a RocksDB `Env` whose I/O is controlled by `det` -- the same
/// `Env::new_file_system_inspected_env` C++ shim `file_system::get_env`
/// already uses in production for I/O rate limiting, just with `DetEnv`
/// (pause/delay/log) as the inspector instead of `IoRateLimiter`. `det` is
/// a cheap `Clone` (`Arc` internally) -- keep your own clone to control it
/// after wiring; the `Env` embeds its own clone of the same shared state.
pub fn get_det_env(
    key_manager: Option<Arc<::encryption::DataKeyManager>>,
    det: DetEnv,
) -> engine_traits::Result<Arc<raw::Env>> {
    let base_env = crate::encryption::get_env(None, key_manager)?
        .unwrap_or_else(|| Arc::new(raw::Env::default()));
    Ok(Arc::new(
        raw::Env::new_file_system_inspected_env(base_env, WrappedFileSystemInspector::new(det))
            .map_err(r2e)?,
    ))
}

/// Campaign note (P1): true crash-between-fsync needs `FileSystemInspector`
/// to buffer non-sync writes and drop them on `crash()`. Today `DetEnv`
/// pause/delay is control-plane only (delay/freeze real I/O, not simulate a
/// torn write); in-process crash recovery is exercised by `test_dst`
/// CrashBetween oracles (stop/restart_engine mid failpoint window) instead.
pub const CRASH_BETWEEN_NOTE: &str =
    "use test_dst CrashBetween + failpoint pause; DetEnv buffer-drop is P1";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det_env_log() {
        let det = DetEnv::new();
        det.set_log_enabled(true);
        assert_eq!(det.drain_log().len(), 0);
        FileSystemInspector::read(&det, 128).unwrap();
        FileSystemInspector::write(&det, 256).unwrap();
        let log = det.drain_log();
        assert_eq!(log.len(), 2);
        assert_eq!((log[0].kind, log[0].bytes), (IoKind::Read, 128));
        assert_eq!((log[1].kind, log[1].bytes), (IoKind::Write, 256));
        // Draining empties it.
        assert_eq!(det.drain_log().len(), 0);
    }

    #[test]
    fn test_det_env_log_disabled_by_default() {
        let det = DetEnv::new();
        FileSystemInspector::read(&det, 64).unwrap();
        assert_eq!(det.drain_log().len(), 0, "logging must be opt-in");
    }

    #[test]
    fn test_det_env_delay_applies_to_every_op() {
        let det = DetEnv::new();
        det.set_delay_ms(15);
        let start = std::time::Instant::now();
        FileSystemInspector::write(&det, 64).unwrap();
        assert!(start.elapsed() >= std::time::Duration::from_millis(15));
    }

    #[test]
    fn test_det_env_pause_blocks_until_unpaused() {
        let det = DetEnv::new();
        det.set_paused(true);
        let d2 = det.clone();
        let handle = std::thread::spawn(move || {
            FileSystemInspector::read(&d2, 1).unwrap();
        });
        std::thread::sleep(std::time::Duration::from_millis(30));
        assert!(!handle.is_finished(), "read should still be blocked while paused");
        det.set_paused(false);
        handle.join().unwrap();
    }

    #[test]
    fn test_det_env_clones_share_state() {
        // The whole point of the Arc-inside-Clone design: a clone must
        // control the *same* underlying DetEnv, not an independent copy.
        let det = DetEnv::new();
        let det2 = det.clone();
        det2.set_log_enabled(true);
        FileSystemInspector::write(&det, 42).unwrap();
        assert_eq!(det2.drain_log().len(), 1, "clone did not observe the original's I/O");
    }

    #[test]
    fn test_get_det_env_wires_into_a_real_rocksdb_open() {
        // The bug this file had: raw_env() silently bypassed everything, so
        // no I/O ever reached DetEnv. Prove the fix end-to-end: open a real
        // RocksDB instance through get_det_env and confirm writes are
        // actually observed by the hook, not just constructible-but-inert.
        let det = DetEnv::new();
        det.set_log_enabled(true);
        let env = get_det_env(None, det.clone()).unwrap();

        let dir = tempfile::Builder::new()
            .prefix("det_env_wiring_test")
            .tempdir()
            .unwrap();
        use engine_traits::{CfOptions, SyncMutable};
        let mut db_opts = crate::RocksDbOptions::default();
        db_opts.set_env(env);
        let cf_opts = vec![(engine_traits::CF_DEFAULT, crate::RocksCfOptions::new())];
        let db = crate::util::new_engine_opt(dir.path().to_str().unwrap(), db_opts, cf_opts)
            .unwrap();
        db.put(b"det_env_key", b"det_env_value").unwrap();

        let log = det.drain_log();
        assert!(
            log.iter().any(|op| op.kind == IoKind::Write && op.bytes > 0),
            "expected at least one logged write reaching DetEnv's hook, got: {log:?}"
        );
    }
}
