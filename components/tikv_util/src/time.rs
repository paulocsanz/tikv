// Copyright 2017 TiKV Project Authors. Licensed under Apache-2.0.

// Re-export duration.
pub use std::time::Duration;
use std::{
    cell::RefCell,
    cmp::Ordering,
    ops::{Add, AddAssign, Sub, SubAssign},
    sync::{
        Once,
        mpsc::{self, Sender},
    },
    thread::{self, Builder, JoinHandle},
    time::{SystemTime, UNIX_EPOCH},
};

use async_speed_limit::clock::{BlockingClock, Clock, StandardClock};
use time::Duration as TimeDuration;

/// Returns the monotonic raw time since some unspecified starting point.
pub use self::inner::monotonic_raw_now;
pub use self::inner::{monotonic_coarse_now, monotonic_now};

/// DST logical clock controls (only available with `dst` feature).
#[cfg(feature = "dst")]
pub use self::inner::{
    dst_advance, dst_now_nanos, dst_reset, dst_set_logical_nanos, dst_set_manual_only, dst_set_step,
    dst_start_hybrid_driver, dst_step,
};

use crate::{sys::thread::StdThreadBuildWrapper, thread_name_prefix::TIME_MONITOR_THREAD};

const NANOSECONDS_PER_SECOND: u64 = 1_000_000_000;
const MILLISECONDS_PER_SECOND: u64 = 1_000;
const MICROSECONDS_PER_SECOND: u64 = 1_000_000;
const NANOSECONDS_PER_MILLISECOND: u64 = 1_000_000;
const NANOSECONDS_PER_MICROSECOND: u64 = 1_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Timespec {
    pub sec: i64,
    pub nsec: i32,
}

impl Timespec {
    pub fn new(sec: i64, nsec: i32) -> Self {
        Self::from_total_nanos((sec as i128) * NANOSECONDS_PER_SECOND as i128 + nsec as i128)
    }

    fn from_total_nanos(total_nanos: i128) -> Self {
        let sec = total_nanos.div_euclid(NANOSECONDS_PER_SECOND as i128);
        let nsec = total_nanos.rem_euclid(NANOSECONDS_PER_SECOND as i128) as i32;
        Self {
            sec: sec as i64,
            nsec,
        }
    }

    fn total_nanos(self) -> i128 {
        (self.sec as i128) * NANOSECONDS_PER_SECOND as i128 + self.nsec as i128
    }
}

impl Add<TimeDuration> for Timespec {
    type Output = Timespec;

    fn add(self, rhs: TimeDuration) -> Self::Output {
        let delta = rhs.whole_nanoseconds();
        Self::from_total_nanos(self.total_nanos() + delta)
    }
}

impl Sub<TimeDuration> for Timespec {
    type Output = Timespec;

    fn sub(self, rhs: TimeDuration) -> Self::Output {
        let delta = rhs.whole_nanoseconds();
        Self::from_total_nanos(self.total_nanos() - delta)
    }
}

impl Sub<Timespec> for Timespec {
    type Output = TimeDuration;

    fn sub(self, rhs: Timespec) -> Self::Output {
        let sec = self
            .sec
            .checked_sub(rhs.sec)
            .expect("overflow when subtracting timespec seconds");
        let nsec = i64::from(self.nsec) - i64::from(rhs.nsec);
        TimeDuration::seconds(sec)
            .checked_add(TimeDuration::nanoseconds(nsec))
            .expect("overflow when subtracting timespecs")
    }
}

/// Converts Duration to milliseconds.
#[inline]
pub fn duration_to_ms(d: Duration) -> u64 {
    let nanos = u64::from(d.subsec_nanos());
    // If Duration is too large, the result may be overflow.
    d.as_secs() * MILLISECONDS_PER_SECOND + (nanos / NANOSECONDS_PER_MILLISECOND)
}

/// Converts Duration to seconds.
#[inline]
pub fn duration_to_sec(d: Duration) -> f64 {
    let nanos = f64::from(d.subsec_nanos());
    d.as_secs() as f64 + (nanos / NANOSECONDS_PER_SECOND as f64)
}

pub fn nanos_to_secs(nanos: u64) -> f64 {
    nanos as f64 / NANOSECONDS_PER_SECOND as f64
}

/// Converts Duration to microseconds.
#[inline]
pub fn duration_to_us(d: Duration) -> u64 {
    let nanos = u64::from(d.subsec_nanos());
    // If Duration is too large, the result may be overflow.
    d.as_secs() * MICROSECONDS_PER_SECOND + (nanos / NANOSECONDS_PER_MICROSECOND)
}

/// Converts TimeSpec to nanoseconds
#[inline]
pub fn timespec_to_ns(t: Timespec) -> u64 {
    (t.sec as u64) * NANOSECONDS_PER_SECOND + t.nsec as u64
}

pub fn get_time() -> Timespec {
    let dur = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    Timespec::new(dur.as_secs() as i64, dur.subsec_nanos() as i32)
}

fn to_std_duration(duration: TimeDuration) -> Duration {
    duration.try_into().unwrap()
}

fn to_time_duration(duration: Duration) -> TimeDuration {
    duration.try_into().unwrap()
}

/// Converts Duration to nanoseconds.
#[inline]
pub fn duration_to_ns(d: Duration) -> u64 {
    let nanos = u64::from(d.subsec_nanos());
    // If Duration is too large, the result may be overflow.
    d.as_secs() * NANOSECONDS_PER_SECOND + nanos
}

pub trait InstantExt {
    fn saturating_elapsed(&self) -> Duration;
}

impl InstantExt for std::time::Instant {
    #[inline]
    fn saturating_elapsed(&self) -> Duration {
        std::time::Instant::now().saturating_duration_since(*self)
    }
}

/// A time in seconds since the start of the Unix epoch.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct UnixSecs(u64);

impl UnixSecs {
    pub fn now() -> UnixSecs {
        UnixSecs(
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        )
    }

    pub fn zero() -> UnixSecs {
        UnixSecs(0)
    }

    pub fn into_inner(self) -> u64 {
        self.0
    }

    pub fn is_zero(self) -> bool {
        self.0 == 0
    }
}

pub struct SlowTimer {
    slow_time: Duration,
    t: Instant,
}

impl SlowTimer {
    pub fn new() -> SlowTimer {
        SlowTimer::default()
    }

    pub fn from(slow_time: Duration) -> SlowTimer {
        SlowTimer {
            slow_time,
            t: Instant::now_coarse(),
        }
    }

    pub fn from_secs(secs: u64) -> SlowTimer {
        SlowTimer::from(Duration::from_secs(secs))
    }

    pub fn from_millis(millis: u64) -> SlowTimer {
        SlowTimer::from(Duration::from_millis(millis))
    }

    pub fn saturating_elapsed(&self) -> Duration {
        self.t.saturating_elapsed()
    }

    pub fn is_slow(&self) -> bool {
        self.saturating_elapsed() >= self.slow_time
    }
}

const DEFAULT_SLOW_SECS: u64 = 1;

impl Default for SlowTimer {
    fn default() -> SlowTimer {
        SlowTimer::from_secs(DEFAULT_SLOW_SECS)
    }
}

const DEFAULT_WAIT_MS: u64 = 100;

pub struct Monitor {
    tx: Sender<bool>,
    handle: Option<JoinHandle<()>>,
}

impl Monitor {
    pub fn new<D, N>(on_jumped: D, now: N) -> Monitor
    where
        D: Fn() + Send + 'static,
        N: Fn() -> SystemTime + Send + 'static,
    {
        let props = crate::thread_group::current_properties();
        let (tx, rx) = mpsc::channel();
        let h = Builder::new()
            .name(thd_name!(TIME_MONITOR_THREAD))
            .spawn_wrapper(move || {
                crate::thread_group::set_properties(props);

                while rx.try_recv().is_err() {
                    let before = now();
                    thread::sleep(Duration::from_millis(DEFAULT_WAIT_MS));

                    let after = now();
                    if let Err(e) = after.duration_since(before) {
                        error!(
                            "system time jumped back";
                            "before" => ?before,
                            "after" => ?after,
                            "err" => ?e,
                        );
                        on_jumped()
                    }
                }
            })
            .unwrap();

        Monitor {
            tx,
            handle: Some(h),
        }
    }
}

impl Default for Monitor {
    fn default() -> Monitor {
        Monitor::new(|| {}, SystemTime::now)
    }
}

impl Drop for Monitor {
    fn drop(&mut self) {
        let h = self.handle.take();
        if h.is_none() {
            return;
        }

        if let Err(e) = self.tx.send(true) {
            error!("send quit message for time monitor worker failed"; "err" => ?e);
            return;
        }

        if let Err(e) = h.unwrap().join() {
            error!("join time monitor worker failed"; "err" => ?e);
        }
    }
}

#[cfg(feature = "dst")]
mod inner {
    //! Deterministic logical clock for DST (Peça 2 foundation).
    //!
    //! Design (revised after lease breakage with per-read advance):
    //! - `Instant::now()` returns the current logical time WITHOUT advancing.
    //! - Time advances only via `dst_advance(nanos)` (explicit, test-driven).
    //! - `dst_reset()` never jumps backward — it leaps to a fresh epoch so
    //!   existing Instant values remain ≤ current time (no "jumped back" panics).
    //! - Optional hybrid driver: advances logical time ~1:1 with wall clock so
    //!   Raft leases and SteadyTimer continue to work while Instant remains
    //!   seed-resettable and monotonic.
    use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
    use std::thread;
    use std::time::Duration;

    use super::Timespec;

    const NANOS_PER_SEC: i64 = 1_000_000_000;
    /// Leap forward on reset so no existing Instant is in the future.
    const RESET_EPOCH_NANOS: i64 = 3_600 * NANOS_PER_SEC; // 1 hour

    static LOGICAL_NANOS: AtomicI64 = AtomicI64::new(0);
    /// Step used by hybrid driver per wall-clock tick (default 1ms).
    static STEP_NANOS: AtomicI64 = AtomicI64::new(1_000_000);
    static DRIVER_RUNNING: AtomicBool = AtomicBool::new(false);
    /// When true, hybrid driver is never auto-started. Time advances only via
    /// `dst_advance` (pure step-driven / virtual timer mode).
    static MANUAL_ONLY: AtomicBool = AtomicBool::new(false);

    /// Set hybrid driver step (nanos of logical time per driver tick).
    pub fn dst_set_step(nanos: i64) {
        STEP_NANOS.store(nanos, Ordering::Relaxed);
    }

    /// Enable pure step-driven mode: no hybrid wall-clock driver.
    /// Call before any `Instant::now()` / `dst_advance` in the test.
    /// SteadyTimer delays fire when logical time is advanced past their deadline
    /// (timer thread polls every 1ms under feature `dst`).
    pub fn dst_set_manual_only(manual: bool) {
        MANUAL_ONLY.store(manual, Ordering::SeqCst);
    }

    /// Manually advance the logical clock by a fixed amount.
    /// Under feature `dst`, SteadyTimer re-checks within ~1ms and fires due delays.
    /// Also syncs the value to `/tmp/dst_clock` for the C LD_PRELOAD bridge.
    pub fn dst_advance(nanos: i64) {
        if nanos > 0 {
            LOGICAL_NANOS.fetch_add(nanos, Ordering::SeqCst);
            crate::det_clock_bridge::sync();
        }
    }

    /// Advance by `n` steps of the configured step size (default 1ms each).
    pub fn dst_step(n: u64) {
        let step = STEP_NANOS.load(Ordering::Relaxed);
        if step > 0 && n > 0 {
            let total = step.saturating_mul(n as i64);
            LOGICAL_NANOS.fetch_add(total, Ordering::SeqCst);
            crate::det_clock_bridge::sync();
        }
    }

    /// Start a new logical epoch without going backward.
    /// Existing Instants remain ≤ new now, so the time monitor never panics.
    pub fn dst_reset() {
        LOGICAL_NANOS.fetch_add(RESET_EPOCH_NANOS, Ordering::SeqCst);
        crate::det_clock_bridge::sync();
    }

    /// Set the logical clock to an absolute value.
    /// The caller must ensure `nanos` is ≥ any previously read `Instant::now()`
    /// so the time monitor never panics.  Use to fix the clock *phase* relative
    /// to Raft tick intervals across repeated scenario invocations.
    pub fn dst_set_logical_nanos(nanos: i64) {
        LOGICAL_NANOS.store(nanos, Ordering::SeqCst);
        crate::det_clock_bridge::sync();
    }

    /// Current logical time in nanos (read-only).
    pub fn dst_now_nanos() -> i64 {
        LOGICAL_NANOS.load(Ordering::SeqCst)
    }

    /// Start a hybrid driver thread: advances logical time by `step` every
    /// `wall_interval`. Call once per process; subsequent calls are no-ops.
    ///
    /// This keeps Raft leases and SteadyTimer roughly synchronized with real
    /// time while preserving a process-global monotonic logical clock that
    /// resets to a fresh epoch via `dst_reset()`.
    pub fn dst_start_hybrid_driver(wall_interval: Duration) {
        if MANUAL_ONLY.load(Ordering::SeqCst) {
            return;
        }
        if DRIVER_RUNNING
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return;
        }
        let step = STEP_NANOS.load(Ordering::Relaxed);
        thread::Builder::new()
            .name("dst-hybrid-clock".into())
            .spawn(move || {
                loop {
                    thread::sleep(wall_interval);
                    if MANUAL_ONLY.load(Ordering::Relaxed) {
                        // Pause advancing while in manual mode (driver may
                        // already have been started before manual was set).
                        continue;
                    }
                    LOGICAL_NANOS.fetch_add(step, Ordering::SeqCst);
                    // Keep C LD_PRELOAD bridge in lockstep with hybrid driver.
                    crate::det_clock_bridge::sync();
                }
            })
            .expect("failed to start dst hybrid clock driver");
    }

    fn ensure_driver() {
        if MANUAL_ONLY.load(Ordering::Relaxed) {
            return;
        }
        // Auto-start on first Instant::now() so tests that enable
        // tikv_util/dst without calling dst_init still make progress.
        // Without this, Instant freezes and Raft leases hang forever.
        if !DRIVER_RUNNING.load(Ordering::Relaxed) {
            dst_start_hybrid_driver(Duration::from_millis(1));
        }
    }

    fn read_now() -> Timespec {
        ensure_driver();
        let nanos = LOGICAL_NANOS.load(Ordering::SeqCst);
        Timespec::new(nanos / NANOS_PER_SEC, (nanos % NANOS_PER_SEC) as i32)
    }

    #[inline]
    pub fn monotonic_raw_now() -> Timespec {
        read_now()
    }

    #[inline]
    pub fn monotonic_now() -> Timespec {
        read_now()
    }

    #[inline]
    pub fn monotonic_coarse_now() -> Timespec {
        read_now()
    }
}

#[cfg(all(not(feature = "dst"), not(target_os = "linux")))]
mod inner {
    use std::{sync::OnceLock, time::Instant};

    use super::Timespec;

    #[inline]
    fn monotonic_elapsed() -> Timespec {
        static MONOTONIC_ORIGIN: OnceLock<Instant> = OnceLock::new();

        // `time::precise_time_ns()` became a `SystemTime` compatibility shim in
        // time 0.2, so keep a process-local monotonic origin on non-Linux.
        let elapsed = MONOTONIC_ORIGIN.get_or_init(Instant::now).elapsed();
        Timespec::new(
            i64::try_from(elapsed.as_secs()).expect("monotonic clock overflow"),
            elapsed.subsec_nanos() as i32,
        )
    }

    pub fn monotonic_raw_now() -> Timespec {
        // TODO Add monotonic raw clock time impl for macos and windows
        monotonic_elapsed()
    }

    pub fn monotonic_now() -> Timespec {
        // TODO Add monotonic clock time impl for macos and windows
        monotonic_elapsed()
    }

    pub fn monotonic_coarse_now() -> Timespec {
        // TODO Add monotonic coarse clock time impl for macos and windows
        monotonic_elapsed()
    }
}

#[cfg(all(not(feature = "dst"), target_os = "linux"))]
mod inner {
    use std::io;

    use super::Timespec;

    #[inline]
    pub fn monotonic_raw_now() -> Timespec {
        get_time(libc::CLOCK_MONOTONIC_RAW)
    }

    #[inline]
    pub fn monotonic_now() -> Timespec {
        get_time(libc::CLOCK_MONOTONIC)
    }

    #[inline]
    pub fn monotonic_coarse_now() -> Timespec {
        get_time(libc::CLOCK_MONOTONIC_COARSE)
    }

    #[inline]
    fn get_time(clock: libc::clockid_t) -> Timespec {
        let mut t = libc::timespec {
            tv_sec: 0,
            tv_nsec: 0,
        };
        let errno = unsafe { libc::clock_gettime(clock, &mut t) };
        if errno != 0 {
            panic!(
                "failed to get clocktime, err {}",
                io::Error::last_os_error()
            );
        }
        Timespec::new(t.tv_sec, t.tv_nsec as _)
    }
}

/// A measurement of a monotonically increasing clock.
/// It's similar and meant to replace `std::time::Instant`,
/// for providing extra features.
#[derive(Copy, Clone, Debug, Eq)]
pub enum Instant {
    Monotonic(Timespec),
    MonotonicCoarse(Timespec),
}

impl Instant {
    pub fn now() -> Instant {
        // Under feature `dst`, both now() and now_coarse() read from the same
        // logical clock. This closes all 705 std Instant::now() call sites
        // that would otherwise bypass the deterministic clock.
        #[cfg(feature = "dst")]
        {
            Instant::MonotonicCoarse(monotonic_coarse_now())
        }
        #[cfg(not(feature = "dst"))]
        {
            Instant::Monotonic(monotonic_now())
        }
    }

    pub fn now_coarse() -> Instant {
        Instant::MonotonicCoarse(monotonic_coarse_now())
    }

    pub fn saturating_elapsed(&self) -> Duration {
        match *self {
            Instant::Monotonic(t) => {
                let now = monotonic_now();
                Instant::saturating_elapsed_duration(now, t)
            }
            Instant::MonotonicCoarse(t) => {
                let now = monotonic_coarse_now();
                Instant::saturating_elapsed_duration_coarse(now, t)
            }
        }
    }

    // This function may panic if the current time is earlier than this
    // instant. Deprecated.
    // pub fn elapsed_secs(&self) -> f64;

    pub fn saturating_elapsed_secs(&self) -> f64 {
        duration_to_sec(self.saturating_elapsed())
    }

    pub fn duration_since(&self, earlier: Instant) -> Duration {
        match (*self, earlier) {
            (Instant::Monotonic(later), Instant::Monotonic(earlier)) => {
                Instant::elapsed_duration(later, earlier)
            }
            (Instant::MonotonicCoarse(later), Instant::MonotonicCoarse(earlier)) => {
                Instant::saturating_elapsed_duration_coarse(later, earlier)
            }
            _ => {
                panic!("duration between different types of Instants");
            }
        }
    }

    pub fn saturating_duration_since(&self, earlier: Instant) -> Duration {
        match (*self, earlier) {
            (Instant::Monotonic(later), Instant::Monotonic(earlier)) => {
                Instant::saturating_elapsed_duration(later, earlier)
            }
            (Instant::MonotonicCoarse(later), Instant::MonotonicCoarse(earlier)) => {
                Instant::saturating_elapsed_duration_coarse(later, earlier)
            }
            _ => {
                panic!("duration between different types of Instants");
            }
        }
    }

    /// It is similar to `duration_since`, but it won't panic when `self` is
    /// less than `other`, and `None` will be returned in this case.
    ///
    /// Callers need to ensure that `self` and `other` are same type of
    /// Instants.
    pub fn checked_sub(&self, other: Instant) -> Option<Duration> {
        if *self >= other {
            Some(self.duration_since(other))
        } else {
            None
        }
    }

    pub(crate) fn elapsed_duration(later: Timespec, earlier: Timespec) -> Duration {
        if later >= earlier {
            to_std_duration(later - earlier)
        } else {
            panic!(
                "monotonic time jumped back, {:.9} -> {:.9}",
                earlier.sec as f64 + f64::from(earlier.nsec) / NANOSECONDS_PER_SECOND as f64,
                later.sec as f64 + f64::from(later.nsec) / NANOSECONDS_PER_SECOND as f64
            );
        }
    }

    pub(crate) fn saturating_elapsed_duration(later: Timespec, earlier: Timespec) -> Duration {
        if later >= earlier {
            to_std_duration(later - earlier)
        } else {
            error!(
                "monotonic time jumped back, {:.3} -> {:.3}",
                earlier.sec as f64 + f64::from(earlier.nsec) / NANOSECONDS_PER_SECOND as f64,
                later.sec as f64 + f64::from(later.nsec) / NANOSECONDS_PER_SECOND as f64
            );
            Duration::from_millis(0)
        }
    }

    // It is different from `elapsed_duration`, the resolution here is millisecond.
    // The processors in an SMP system do not start all at exactly the same time
    // and therefore the timer registers are typically running at an offset.
    // Use millisecond resolution for ignoring the error.
    // See more: https://linux.die.net/man/2/clock_gettime
    pub(crate) fn saturating_elapsed_duration_coarse(
        later: Timespec,
        earlier: Timespec,
    ) -> Duration {
        let later_ms = later.sec * MILLISECONDS_PER_SECOND as i64
            + i64::from(later.nsec) / NANOSECONDS_PER_MILLISECOND as i64;
        let earlier_ms = earlier.sec * MILLISECONDS_PER_SECOND as i64
            + i64::from(earlier.nsec) / NANOSECONDS_PER_MILLISECOND as i64;
        let dur = later_ms - earlier_ms;
        if dur >= 0 {
            Duration::from_millis(dur as u64)
        } else {
            debug!(
                "coarse time jumped back, {:.3} -> {:.3}",
                earlier.sec as f64 + f64::from(earlier.nsec) / NANOSECONDS_PER_SECOND as f64,
                later.sec as f64 + f64::from(later.nsec) / NANOSECONDS_PER_SECOND as f64
            );
            Duration::from_millis(0)
        }
    }
}

impl PartialEq for Instant {
    fn eq(&self, other: &Instant) -> bool {
        match (*self, *other) {
            (Instant::Monotonic(this), Instant::Monotonic(other))
            | (Instant::MonotonicCoarse(this), Instant::MonotonicCoarse(other)) => this.eq(&other),
            _ => false,
        }
    }
}

impl PartialOrd for Instant {
    fn partial_cmp(&self, other: &Instant) -> Option<Ordering> {
        match (*self, *other) {
            (Instant::Monotonic(this), Instant::Monotonic(other))
            | (Instant::MonotonicCoarse(this), Instant::MonotonicCoarse(other)) => {
                this.partial_cmp(&other)
            }
            // The Order of different types of Instants is meaningless.
            _ => None,
        }
    }
}

impl Add<Duration> for Instant {
    type Output = Instant;

    fn add(self, other: Duration) -> Instant {
        match self {
            Instant::Monotonic(t) => Instant::Monotonic(t + to_time_duration(other)),
            Instant::MonotonicCoarse(t) => Instant::MonotonicCoarse(t + to_time_duration(other)),
        }
    }
}

impl AddAssign<Duration> for Instant {
    fn add_assign(&mut self, rhs: Duration) {
        *self = self.add(rhs)
    }
}

impl Sub<Duration> for Instant {
    type Output = Instant;

    fn sub(self, other: Duration) -> Instant {
        match self {
            Instant::Monotonic(t) => Instant::Monotonic(t - to_time_duration(other)),
            Instant::MonotonicCoarse(t) => Instant::MonotonicCoarse(t - to_time_duration(other)),
        }
    }
}

impl SubAssign<Duration> for Instant {
    fn sub_assign(&mut self, rhs: Duration) {
        *self = self.sub(rhs)
    }
}

impl Sub<Instant> for Instant {
    type Output = Duration;

    // TODO: For safety in production code, `sub` actually does saturating_sub.
    // We should remove this operator from public scope.
    fn sub(self, other: Instant) -> Duration {
        self.saturating_duration_since(other)
    }
}

/// A coarse clock for `async_speed_limit`.
#[derive(Copy, Clone, Default, Debug)]
pub struct CoarseClock;

impl Clock for CoarseClock {
    type Instant = Instant;
    type Delay = <StandardClock as Clock>::Delay;

    fn now(&self) -> Self::Instant {
        Instant::now_coarse()
    }

    fn sleep(&self, dur: Duration) -> Self::Delay {
        StandardClock.sleep(dur)
    }
}

impl BlockingClock for CoarseClock {
    fn blocking_sleep(&self, dur: Duration) {
        StandardClock.blocking_sleep(dur);
    }
}

/// A limiter which uses the coarse clock for measurement.
pub type Limiter = async_speed_limit::Limiter<CoarseClock>;
pub type Consume = async_speed_limit::limiter::Consume<CoarseClock, ()>;

/// ReadId to judge whether the read requests come from the same GRPC stream.
#[derive(PartialEq, Clone, Debug)]
pub struct ThreadReadId {
    sequence: u64,
    pub create_time: Timespec,
}

thread_local!(static READ_SEQUENCE: RefCell<u64> = const { RefCell::new(0) });

impl ThreadReadId {
    pub fn new() -> ThreadReadId {
        let sequence = READ_SEQUENCE.with(|s| {
            let seq = *s.borrow() + 1;
            *s.borrow_mut() = seq;
            seq
        });
        ThreadReadId {
            sequence,
            create_time: monotonic_raw_now(),
        }
    }
}

impl Default for ThreadReadId {
    fn default() -> Self {
        Self::new()
    }
}

/// Default duration cost for spinning one round.
///
/// Heuristically, spin duration for one round is about 3～4ns.
static mut DEFAULT_DURATION_SPIN_ONE_ROUND: u64 = 1;

/// Setup the default ratio for spin duration.
pub fn setup_for_spin_interval() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        let inspect_duration = Duration::from_millis(10);
        let start = Instant::now();
        let mut count = 0;
        // Spin for a while to get the duration for one round.
        for _ in 0..2_097_152 {
            count += 1;
            if count % 1024 == 0 && start.saturating_elapsed() >= inspect_duration {
                break;
            }
        }
        let elapsed_one_round = start.saturating_elapsed().as_nanos() as u64 / count;
        if elapsed_one_round > 0 {
            unsafe {
                DEFAULT_DURATION_SPIN_ONE_ROUND = elapsed_one_round;
            }
        }
        debug!("setup duration for spinning one round: {}ns", unsafe {
            DEFAULT_DURATION_SPIN_ONE_ROUND
        });
    });
}

/// Wait for at least `elaspsed` duration synchronously by looping.
///
/// Attention, this function is only suitable for short-time spinning, so
/// the `elaspsed` should be small, like 1ms. And the caller should not
/// rely on it to guarantee the exact time to sleep.
pub fn spin_at_least(elaspsed: Duration) {
    // Initialize default spin loop interval.
    setup_for_spin_interval();

    let rounds = unsafe { elaspsed.as_nanos() as u64 / DEFAULT_DURATION_SPIN_ONE_ROUND };
    let now = Instant::now();
    for i in 1..=rounds {
        if i % 100 == 0 && now.saturating_elapsed() >= elaspsed {
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        ops::Sub,
        sync::{
            Arc,
            atomic::{AtomicBool, Ordering},
        },
        thread,
        time::{Duration, SystemTime},
    };

    use test::Bencher;

    use super::*;

    #[test]
    fn test_time_monitor() {
        let jumped = Arc::new(AtomicBool::new(false));
        let triggered = AtomicBool::new(false);
        let now = move || {
            if !triggered.load(Ordering::SeqCst) {
                triggered.store(true, Ordering::SeqCst);
                SystemTime::now()
            } else {
                SystemTime::now().sub(Duration::from_secs(2))
            }
        };

        let jumped2 = Arc::clone(&jumped);
        let on_jumped = move || {
            jumped2.store(true, Ordering::SeqCst);
        };

        let _m = Monitor::new(on_jumped, now);
        thread::sleep(Duration::from_secs(1));

        assert_eq!(jumped.load(Ordering::SeqCst), true);
    }

    #[test]
    fn test_duration_to() {
        let tbl = vec![0, 100, 1_000, 5_000, 9999, 1_000_000, 1_000_000_000];
        for ms in tbl {
            let d = Duration::from_millis(ms);
            assert_eq!(ms, duration_to_ms(d));
            let exp_sec = ms as f64 / 1000.0;
            let act_sec = duration_to_sec(d);
            assert!((act_sec - exp_sec).abs() < f64::EPSILON);
            assert_eq!(ms * 1_000, duration_to_us(d));
            assert_eq!(ms * 1_000_000, duration_to_ns(d));
        }
    }

    #[test]
    fn test_nanos_to_secs() {
        assert_eq!(nanos_to_secs(0), 0.0);
        assert_eq!(nanos_to_secs(1), 1e-9);
        assert_eq!(nanos_to_secs(NANOSECONDS_PER_SECOND), 1.0);
        assert_eq!(nanos_to_secs(1_500_000_000), 1.5);
        // Test with a large number of nanoseconds (e.g., 10 billion ns = 10 seconds)
        assert_eq!(nanos_to_secs(10 * NANOSECONDS_PER_SECOND), 10.0);
    }

    #[test]
    fn test_timespec_sub_large_span() {
        let later = Timespec::new(10_000_000_000, 123);
        let earlier = Timespec::new(0, 456);
        let expected =
            TimeDuration::seconds(9_999_999_999) + TimeDuration::nanoseconds(999_999_667);
        assert_eq!(later - earlier, expected);
    }

    #[test]
    fn test_now() {
        let pairs = vec![
            (monotonic_raw_now(), monotonic_raw_now()),
            (monotonic_now(), monotonic_now()),
            (monotonic_coarse_now(), monotonic_coarse_now()),
        ];
        for (early_time, late_time) in pairs {
            // The monotonic clocktime must be strictly monotonic increasing.
            assert!(
                late_time >= early_time,
                "expect late time {:?} >= early time {:?}",
                late_time,
                early_time
            );
        }
    }

    #[test]
    #[allow(clippy::eq_op)]
    fn test_instant() {
        Instant::now().saturating_elapsed();
        Instant::now_coarse().saturating_elapsed();

        // Ordering.
        let early_raw = Instant::now();
        let late_raw = Instant::now();
        assert!(early_raw <= late_raw);
        assert!(late_raw >= early_raw);

        assert_eq!(early_raw, early_raw);
        assert!(early_raw >= early_raw);
        assert!(early_raw <= early_raw);

        let early_coarse = Instant::now_coarse();
        let late_coarse = Instant::now_coarse();
        assert!(late_coarse >= early_coarse);
        assert!(early_coarse <= late_coarse);

        assert_eq!(early_coarse, early_coarse);
        assert!(early_coarse >= early_coarse);
        assert!(early_coarse <= early_coarse);

        let zero = Duration::new(0, 0);
        // Sub Instant.
        assert!(late_raw.duration_since(early_raw) >= zero);
        assert!(late_coarse.duration_since(early_coarse) >= zero);

        // Sub Duration.
        assert_eq!(late_raw - zero, late_raw);
        assert_eq!(late_coarse - zero, late_coarse);

        // Sub assign Duration
        let mut tmp_late_row = late_raw;
        tmp_late_row -= zero;
        assert_eq!(tmp_late_row, late_raw);

        // checked_sub Duration.
        assert!(late_raw.checked_sub(early_raw).unwrap() >= zero);
        // It's either `None` or `Some(zero)`(if they are equal).
        assert_eq!(early_raw.checked_sub(late_raw).unwrap_or(zero), zero);

        let mut tmp_late_coarse = late_coarse;
        tmp_late_coarse -= zero;
        assert_eq!(tmp_late_coarse, late_coarse);

        // Add Duration.
        assert_eq!(late_raw + zero, late_raw);
        assert_eq!(late_coarse + zero, late_coarse);

        // add assign
        let mut tmp_late_row = late_raw;
        tmp_late_row += zero;
        assert_eq!(tmp_late_row, late_raw);

        let mut tmp_coarse = late_coarse;
        tmp_coarse += zero;
        assert_eq!(tmp_coarse, late_coarse);

        // PartialEq and PartialOrd
        let ts = Timespec::new(1, 1);
        let now1 = Instant::Monotonic(ts);
        let now2 = Instant::MonotonicCoarse(ts);
        assert_ne!(now1, now2);
        assert_eq!(now1.partial_cmp(&now2), None);
    }

    #[test]
    fn test_coarse_instant_on_smp() {
        let zero = Duration::from_millis(0);
        for i in 0..1_000_000 {
            let now = Instant::now();
            let now_coarse = Instant::now_coarse();
            if i % 100 == 0 {
                thread::yield_now();
            }
            assert!(now.saturating_elapsed() >= zero);
            assert!(now_coarse.saturating_elapsed() >= zero);
        }
    }

    #[test]
    fn test_wait_at_least() {
        setup_for_spin_interval();

        let start = Instant::now();
        spin_at_least(Duration::from_micros(500));
        assert!(start.saturating_elapsed() >= Duration::from_micros(100));
    }

    #[bench]
    fn bench_instant_now(b: &mut Bencher) {
        b.iter(|| {
            let _now = Instant::now();
        });
    }

    #[bench]
    fn bench_instant_now_coarse(b: &mut Bencher) {
        b.iter(|| {
            let _now = Instant::now_coarse();
        });
    }
}
