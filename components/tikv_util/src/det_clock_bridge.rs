// det_clock_bridge.rs — Bridge between the Rust DST logical clock and the
// C det_clock.so / libdet_clock.dylib LD_PRELOAD library.
//
// Writes `dst_now_nanos()` into a shared file so C/C++ code (RocksDB, gRPC)
// that calls clock_gettime sees the same logical time as Rust.
//
// Only compiled under feature "dst".

use std::fs::OpenOptions;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

static INITIALIZED: AtomicBool = AtomicBool::new(false);
static SHM_PATH: Mutex<Option<PathBuf>> = Mutex::new(None);

const DEFAULT_PATH: &str = "/tmp/dst_clock";

/// Initialize the bridge. Call once at test startup.
pub fn init(shm_path: Option<&str>) {
    let path = PathBuf::from(shm_path.unwrap_or(DEFAULT_PATH));
    if let Err(e) = write_nanos_to(&path, crate::time::dst_now_nanos()) {
        eprintln!("[det_clock_bridge] init failed ({path:?}): {e}");
        return;
    }
    *SHM_PATH.lock().unwrap() = Some(path);
    INITIALIZED.store(true, Ordering::SeqCst);
}

/// Sync the current logical clock to the shared file.
/// Call after every `dst_advance` / `dst_step` so C code stays in lockstep.
pub fn sync() {
    if !INITIALIZED.load(Ordering::SeqCst) {
        return;
    }
    let guard = SHM_PATH.lock().unwrap();
    if let Some(ref path) = *guard {
        let nanos = crate::time::dst_now_nanos();
        if let Err(e) = write_nanos_to(path, nanos) {
            eprintln!("[det_clock_bridge] sync failed: {e}");
        }
    }
}

fn write_nanos_to(path: &Path, nanos: i64) -> std::io::Result<()> {
    let mut f = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(false)
        .open(path)?;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&nanos.to_le_bytes())?;
    f.flush()?;
    Ok(())
}

/// Path currently used for the shared clock (for LD_PRELOAD env setup).
pub fn shm_path() -> String {
    SHM_PATH
        .lock()
        .unwrap()
        .as_ref()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|| DEFAULT_PATH.to_string())
}
