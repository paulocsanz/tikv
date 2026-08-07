// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    cell::{RefCell, UnsafeCell},
    mem::{self, ManuallyDrop},
    sync::Arc,
};

use parking_lot::Mutex;
use tokio::sync::{Mutex as AsyncMutex, MutexGuard as AsyncMutexGuard};
use txn_types::{Key, Lock};

use super::lock_table::LockTable;

thread_local! {
    /// Arcs retired from `KeyHandleGuard` on this thread, not yet freed.
    ///
    /// Freeing the last `Arc<KeyHandle>` **inside** `Drop` of the guard is
    /// undefined under Stacked/Tree Borrows (the guard value protects the
    /// Mutex allocation for the whole drop frame). Safe API
    /// `Arc::new(...).lock().await` + `drop` must not cause UB — so we retire
    /// the Arc and free it on a later `lock()` on this thread (or leak at TLS
    /// teardown), same idea as `tikv_util::range_latch`.
    static RETIRED_HANDLES: RefCell<Vec<Arc<KeyHandle>>> =
        const { RefCell::new(Vec::new()) };
}

/// An entry in the in-memory table providing functions related to a specific
/// key.
pub struct KeyHandle {
    pub key: Key,
    table: UnsafeCell<Option<LockTable>>,
    mutex: AsyncMutex<()>,
    lock_store: Mutex<Option<Lock>>,
}

impl KeyHandle {
    pub fn new(key: Key) -> Self {
        KeyHandle {
            key,
            table: UnsafeCell::new(None),
            mutex: AsyncMutex::new(()),
            lock_store: Mutex::new(None),
        }
    }

    fn reclaim_retired() {
        let _ = RETIRED_HANDLES.try_with(|r| r.borrow_mut().clear());
    }

    pub async fn lock(self: Arc<Self>) -> KeyHandleGuard {
        // Free Arcs retired by earlier guard drops on this thread (outside any
        // guard Drop-protect frame).
        Self::reclaim_retired();

        // Safety: `_mutex_guard` is dropped (unlocked) before the Arc is freed
        // (Arc is retired, not freed, inside Guard Drop; free is deferred).
        let mutex_guard = unsafe {
            #[allow(clippy::missing_transmute_annotations)]
            mem::transmute(self.mutex.lock().await)
        };
        KeyHandleGuard {
            _mutex_guard: ManuallyDrop::new(mutex_guard),
            handle: ManuallyDrop::new(self),
        }
    }

    pub fn with_lock<T>(&self, f: impl FnOnce(&Option<Lock>) -> T) -> T {
        f(&self.lock_store.lock())
    }

    /// Set the LockTable that the KeyHandle is in.
    ///
    /// This method is not thread safe. Make sure that no other threads access
    /// `table` at the same time.
    pub(crate) unsafe fn set_table(&self, table: LockTable) {
        *self.table.get() = Some(table);
    }
}

impl Drop for KeyHandle {
    fn drop(&mut self) {
        // SAFETY: `&mut self` ensures it's the only thread that can access `table`.
        unsafe {
            if let Some(table) = &*self.table.get() {
                table.remove(&self.key);
            }
        }
    }
}

unsafe impl Sync for KeyHandle {}

/// A `KeyHandle` with its mutex locked.
pub struct KeyHandleGuard {
    // Dropped explicitly in `Drop` before the Arc is retired.
    _mutex_guard: ManuallyDrop<AsyncMutexGuard<'static, ()>>,
    // Dropped via retire list — never free last Arc inside this struct's Drop.
    handle: ManuallyDrop<Arc<KeyHandle>>,
}

impl KeyHandleGuard {
    pub fn key(&self) -> &Key {
        &self.handle.key
    }

    pub fn with_lock<T>(&self, f: impl FnOnce(&mut Option<Lock>) -> T) -> T {
        f(&mut self.handle.lock_store.lock())
    }

    pub(crate) fn handle(&self) -> &Arc<KeyHandle> {
        &self.handle
    }
}

impl Drop for KeyHandleGuard {
    fn drop(&mut self) {
        // We only keep the lock in memory until the write to the underlying
        // store finishes.
        *self.handle.lock_store.lock() = None;
        // Unlock while Arc still keeps the Mutex alive.
        unsafe { ManuallyDrop::drop(&mut self._mutex_guard) };
        // Retire Arc: free outside this Drop-protect frame (next lock() or TLS end).
        let arc = unsafe { ManuallyDrop::take(&mut self.handle) };
        let mut hold = Some(arc);
        let parked = RETIRED_HANDLES.try_with(|r| {
            if let Some(a) = hold.take() {
                r.borrow_mut().push(a);
            }
        });
        if parked.is_err() {
            // TLS gone (thread teardown): leak rather than free under protect.
            if let Some(a) = hold.take() {
                mem::forget(a);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicUsize, Ordering},
        time::Duration,
    };

    use tokio::time::sleep;

    use super::*;

    #[tokio::test]
    async fn test_key_mutex() {
        let key_handle = Arc::new(KeyHandle::new(Key::from_raw(b"k")));

        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();
        for _ in 0..100 {
            let key_handle = key_handle.clone();
            let counter = counter.clone();
            let handle = tokio::spawn(async move {
                let _guard = key_handle.lock().await;
                // Modify an atomic counter with a mutex guard. The value of the counter
                // should remain unchanged if the mutex works.
                let counter_val = counter.fetch_add(1, Ordering::SeqCst) + 1;
                sleep(Duration::from_millis(1)).await;
                assert_eq!(counter.load(Ordering::SeqCst), counter_val);
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.await.unwrap();
        }
        assert_eq!(counter.load(Ordering::SeqCst), 100);
    }

    #[tokio::test]
    async fn test_ref_count() {
        let table = LockTable::default();

        let k = Key::from_raw(b"k");

        let handle = Arc::new(KeyHandle::new(k.clone()));
        table.0.insert(k.clone(), Arc::downgrade(&handle));
        unsafe {
            handle.set_table(table.clone());
        }
        let lock_ref1 = table.get(&k).unwrap();
        let lock_ref2 = table.get(&k).unwrap();
        drop(handle);
        drop(lock_ref1);
        assert!(table.get(&k).is_some());
        drop(lock_ref2);
        assert!(table.get(&k).is_none());
    }

    /// Sole-owner Arc in the guard: must not UB under Miri (Drop-protect free).
    ///
    /// Safe API only: `Arc::new(...).lock().await` + drop. Pre-fix free of the
    /// last Arc inside Guard Drop fails Miri SB/TB. Same class as RangeLatch R1.
    ///
    /// Production uses `LockTable::lock_key` (Weak in map); see
    /// `lock_table::test::miri_soundness_lock_key_last_arc`.
    ///
    /// ```text
    /// cargo +nightly miri test -p concurrency_manager --lib \
    ///   key_handle::tests::miri_soundness_key_handle_last_arc
    /// ```
    #[tokio::test]
    async fn miri_soundness_key_handle_last_arc() {
        for _ in 0..8 {
            let g = Arc::new(KeyHandle::new(Key::from_raw(b"solo"))).lock().await;
            drop(g);
        }
        // Reclaim on a subsequent lock on this thread.
        let keep = Arc::new(KeyHandle::new(Key::from_raw(b"keep")));
        let g = keep.clone().lock().await;
        drop(g);
        drop(keep);
    }
}
