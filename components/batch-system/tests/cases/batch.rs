// Copyright 2020 TiKV Project Authors. Licensed under Apache-2.0.

use std::{
    sync::{Arc, atomic::AtomicUsize},
    thread::sleep,
    time::Duration,
};

#[cfg(feature = "dst")]
use std::sync::Mutex;

use batch_system::{test_runner::*, *};
use kvproto::resource_manager::{GroupMode, GroupRawResourceSettings, ResourceGroup};
use resource_control::ResourceGroupManager;
use tikv_util::mpsc;

#[test]
fn test_batch() {
    let (control_tx, control_fsm) = Runner::new(10);
    let (router, mut system) =
        batch_system::create_system(&Config::default(), control_tx, control_fsm, None);
    let builder = Builder::new();
    let metrics = builder.metrics.clone();
    system.spawn("test".to_owned(), builder);
    let mut expected_metrics = HandleMetrics::default();
    assert_eq!(*metrics.lock().unwrap(), expected_metrics);
    let (tx, rx) = mpsc::unbounded();
    let tx_ = tx.clone();
    let r = router.clone();
    router
        .send_control(Message::Callback(Box::new(
            move |_: &Handler, _: &mut Runner| {
                let (tx, runner) = Runner::new(10);
                let mailbox = BasicMailbox::new(tx, runner, Arc::default());
                r.register(1, mailbox);
                tx_.send(1).unwrap();
            },
        )))
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(1));
    // sleep to wait Batch-System to finish calling end().
    sleep(Duration::from_millis(20));
    router
        .send(
            1,
            Message::Callback(Box::new(move |_: &Handler, _: &mut Runner| {
                tx.send(2).unwrap();
            })),
        )
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(2));
    system.shutdown();
    expected_metrics.control = 1;
    expected_metrics.normal = 1;
    expected_metrics.begin = 2;
    assert_eq!(*metrics.lock().unwrap(), expected_metrics);
}

#[test]
fn test_priority() {
    let (control_tx, control_fsm) = Runner::new(10);
    let (router, mut system) =
        batch_system::create_system(&Config::default(), control_tx, control_fsm, None);
    let builder = Builder::new();
    system.spawn("test".to_owned(), builder);
    let (tx, rx) = mpsc::unbounded();
    let tx_ = tx.clone();
    let r = router.clone();
    let state_cnt = Arc::new(AtomicUsize::new(0));
    router
        .send_control(Message::Callback(Box::new(
            move |_: &Handler, _: &mut Runner| {
                let (tx, runner) = Runner::new(10);
                r.register(1, BasicMailbox::new(tx, runner, state_cnt.clone()));
                let (tx2, mut runner2) = Runner::new(10);
                runner2.set_priority(Priority::Low);
                r.register(2, BasicMailbox::new(tx2, runner2, state_cnt));
                tx_.send(1).unwrap();
            },
        )))
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(1));

    let tx_ = tx.clone();
    router
        .send(
            1,
            Message::Callback(Box::new(move |h: &Handler, r: &mut Runner| {
                assert_eq!(h.get_priority(), Priority::Normal);
                assert_eq!(h.get_priority(), r.get_priority());
                tx_.send(2).unwrap();
            })),
        )
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(2));

    router
        .send(
            2,
            Message::Callback(Box::new(move |h: &Handler, r: &mut Runner| {
                assert_eq!(h.get_priority(), Priority::Low);
                assert_eq!(h.get_priority(), r.get_priority());
                tx.send(3).unwrap();
            })),
        )
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(3));
}

#[test]
fn test_resource_group() {
    let (control_tx, control_fsm) = Runner::new(10);
    let resource_manager = ResourceGroupManager::default();

    let get_group = |name: &str, read_tokens: u64, write_tokens: u64| -> ResourceGroup {
        let mut group = ResourceGroup::new();
        group.set_name(name.to_string());
        group.set_mode(GroupMode::RawMode);
        let mut resource_setting = GroupRawResourceSettings::new();
        resource_setting
            .mut_cpu()
            .mut_settings()
            .set_fill_rate(read_tokens);
        resource_setting
            .mut_io_write()
            .mut_settings()
            .set_fill_rate(write_tokens);
        group.set_raw_resource_settings(resource_setting);
        group
    };

    resource_manager.add_resource_group(get_group("group1", 10, 10));
    resource_manager.add_resource_group(get_group("group2", 100, 100));

    let mut cfg = Config::default();
    cfg.pool_size = 1;
    let (router, mut system) = batch_system::create_system(
        &cfg,
        control_tx,
        control_fsm,
        Some(resource_manager.derive_controller("test".to_string(), false)),
    );
    let builder = Builder::new();
    system.spawn("test".to_owned(), builder);
    let (tx, rx) = mpsc::unbounded();
    let tx_ = tx.clone();
    let r = router.clone();
    let state_cnt = Arc::new(AtomicUsize::new(0));
    router
        .send_control(Message::Callback(Box::new(
            move |_: &Handler, _: &mut Runner| {
                let (tx, runner) = Runner::new(10);
                r.register(1, BasicMailbox::new(tx, runner, state_cnt.clone()));
                let (tx2, runner2) = Runner::new(10);
                r.register(2, BasicMailbox::new(tx2, runner2, state_cnt));
                tx_.send(0).unwrap();
            },
        )))
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(0));

    let tx_ = tx.clone();
    let (tx1, rx1) = std::sync::mpsc::sync_channel(0);
    // block the thread
    router
        .send_control(Message::Callback(Box::new(
            move |_: &Handler, _: &mut Runner| {
                tx_.send(0).unwrap();
                tx1.send(0).unwrap();
            },
        )))
        .unwrap();
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(0));

    router
        .send(1, Message::Resource("group1".to_string(), 1))
        .unwrap();
    let tx_ = tx.clone();
    router
        .send(
            1,
            Message::Callback(Box::new(move |_: &Handler, _: &mut Runner| {
                tx_.send(1).unwrap();
            })),
        )
        .unwrap();

    router
        .send(2, Message::Resource("group2".to_string(), 1))
        .unwrap();
    router
        .send(
            2,
            Message::Callback(Box::new(move |_: &Handler, _: &mut Runner| {
                tx.send(2).unwrap();
            })),
        )
        .unwrap();

    // pause the blocking thread
    assert_eq!(rx1.recv_timeout(Duration::from_secs(3)), Ok(0));

    // should recv from group2 first, because group2 has more tokens and it would be
    // handled with higher priority.
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(2));
    assert_eq!(rx.recv_timeout(Duration::from_secs(3)), Ok(1));
}

/// Pure-Rust DST harness: manual drive + logical clock + ordered callbacks.
///
/// Same seed ⇒ same event log across two independent BatchSystem lives.
/// Requires `--features dst` (cooperative pollers + tikv_util logical Instant).
/// Run with `--test-threads=1` — executor state is process-global.
#[cfg(feature = "dst")]
#[test]
fn test_dst_manual_drive_bitstable() {
    struct ManualGuard;
    impl Drop for ManualGuard {
        fn drop(&mut self) {
            end_manual_scenario();
        }
    }

    fn run_scenario(seed: u64) -> (Vec<u64>, HandleMetrics) {
        begin_manual_scenario(seed);
        let _guard = ManualGuard;

        let (control_tx, control_fsm) = Runner::new(64);
        let mut cfg = Config::default();
        cfg.pool_size = 2;
        cfg.low_priority_pool_size = 0;
        let (router, mut system) =
            batch_system::create_system(&cfg, control_tx, control_fsm, None);
        let builder = Builder::new();
        let metrics = builder.metrics.clone();
        system.spawn("dst-manual".to_owned(), builder);

        // Wait until both normal pollers are registered with the global executor.
        for _ in 0..200 {
            if live_count() >= 2 {
                break;
            }
            let _ = step_all_once();
        }
        assert!(
            live_count() >= 2,
            "expected ≥2 pollers registered, got {}",
            live_count()
        );

        let log: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
        let log_reg = Arc::clone(&log);
        let r = router.clone();

        // Register three normal FSMs via control; record ids in log.
        router
            .send_control(Message::Callback(Box::new(
                move |_: &Handler, _: &mut Runner| {
                    for id in 1u64..=3 {
                        let (tx, runner) = Runner::new(64);
                        r.register(id, BasicMailbox::new(tx, runner, Arc::default()));
                        log_reg.lock().unwrap().push(100 + id);
                    }
                },
            )))
            .unwrap();

        // Drive until registration callbacks run.
        for _ in 0..64 {
            let _ = drive_once();
            if log.lock().unwrap().len() >= 3 {
                break;
            }
        }
        assert_eq!(
            log.lock().unwrap().as_slice(),
            &[101, 102, 103],
            "registration order must be deterministic"
        );

        // Enqueue work on each FSM: Loop work + tagged callbacks.
        for id in 1u64..=3 {
            router.send(id, Message::Loop(50)).unwrap();
            let log_cb = Arc::clone(&log);
            router
                .send(
                    id,
                    Message::Callback(Box::new(move |_: &Handler, _: &mut Runner| {
                        log_cb.lock().unwrap().push(id * 10);
                    })),
                )
                .unwrap();
        }

        // Step-driven until all tags appear (or budget exhausted).
        for _ in 0..128 {
            let _ = drive_once();
            if log.lock().unwrap().len() >= 6 {
                break;
            }
        }

        let events = log.lock().unwrap().clone();
        let m = *metrics.lock().unwrap();
        system.shutdown();
        // end via Drop guard
        (events, m)
    }

    let (e1, m1) = run_scenario(0xD57_001);
    let (e2, m2) = run_scenario(0xD57_001);
    assert_eq!(e1, e2, "event log must bit-match across seed replays");
    assert_eq!(e1.len(), 6, "register(3) + tags(3): {e1:?}");
    assert_eq!(&e1[..3], &[101, 102, 103]);
    // Tag set must be complete (order is part of the bitstable claim via e1==e2).
    let mut tags = e1[3..].to_vec();
    tags.sort_unstable();
    assert_eq!(tags, vec![10, 20, 30]);
    assert_eq!(m1, m2, "handle metrics must match: {m1:?} vs {m2:?}");
    assert!(m1.control >= 1);
    assert!(m1.normal >= 3);
}

/// `drive_once` advances logical Instant between poller rounds.
#[cfg(feature = "dst")]
#[test]
fn test_dst_drive_once_advances_logical_clock() {
    begin_manual_scenario(7);
    struct ManualGuard;
    impl Drop for ManualGuard {
        fn drop(&mut self) {
            end_manual_scenario();
        }
    }
    let _guard = ManualGuard;
    let t0 = tikv_util::time::Instant::now_coarse();
    let _ = drive_once();
    let _ = drive_once();
    let t1 = tikv_util::time::Instant::now_coarse();
    let ms = t1.saturating_duration_since(t0).as_millis();
    // Two drive_once → two dst_tick (1ms each) ≈ 2ms logical.
    assert!(
        ms >= 1 && ms <= 5,
        "expected ~2ms logical advance, got {ms}ms"
    );
}
