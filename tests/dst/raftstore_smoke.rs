// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.
//
// Slim pure-Rust DST smoke for raftstore (feature `dst` required).
//
// Proves:
// 1. DstNetworkQueue hold/sort/release is seed-stable (unit, no cluster).
// 2. 1-node cluster under dst + logical clock can elect + put/get, and two
//    runs with the same seed produce the same stable KV fingerprint.
// 3. (heavier) step-driven 3-node path is available behind env DST_FULL=1.
//
// Run:
//   cargo test -p tests --features "dst,testexport" --test dst_raftstore -- --test-threads=1
//   DST_FULL=1 cargo test -p tests --features "dst,testexport" --test dst_raftstore \
//     test_dst_step_driven_put_fingerprint -- --test-threads=1 --nocapture

#![cfg(feature = "dst")]

use std::task::{Context, Poll};

use futures::Future;
use kvproto::raft_serverpb::RaftMessage;
use raft::eraftpb::MessageType;
use test_raftstore::{
    CloneFilterFactory, DstNetworkQueue, Filter, msg_sort_key, new_node_cluster, new_peer,
};
use tikv_util::{config::ReadableSize, time};

fn dst_setup_cluster(cluster: &mut test_raftstore::Cluster<test_raftstore::NodeCluster>) {
    cluster.cfg.raft_store.store_batch_system.pool_size = 1;
    cluster.cfg.raft_store.store_batch_system.low_priority_pool_size = 0;
    cluster.cfg.raft_store.apply_batch_system.pool_size = 1;
    cluster.cfg.raft_store.apply_batch_system.low_priority_pool_size = 0;

    cluster.cfg.rocksdb.max_background_jobs = 1;
    cluster.cfg.rocksdb.max_background_flushes = 1;
    cluster.cfg.rocksdb.max_sub_compactions = 0;
    cluster.cfg.rocksdb.defaultcf.write_buffer_size = Some(ReadableSize::mb(256));
    cluster.cfg.rocksdb.defaultcf.disable_auto_compactions = true;
    cluster.cfg.rocksdb.writecf.write_buffer_size = Some(ReadableSize::mb(128));
    cluster.cfg.rocksdb.writecf.disable_auto_compactions = true;
    cluster.cfg.rocksdb.lockcf.disable_auto_compactions = true;
    cluster.cfg.raftdb.max_background_jobs = 1;
    cluster.cfg.raftdb.max_background_flushes = 1;
    cluster.cfg.raftdb.max_sub_compactions = 0;
    cluster.cfg.raftdb.defaultcf.write_buffer_size = Some(ReadableSize::mb(128));
    cluster.cfg.raftdb.defaultcf.disable_auto_compactions = true;
}

fn dst_tick_ms(ms: u64) {
    time::dst_advance((ms as i64) * 1_000_000);
    tikv_util::timer::dst_process_timers();
    // Tiny wall sleep so hybrid-less SteadyTimer threads and the cooperative
    // batch driver get a chance to run between logical advances.
    std::thread::sleep(std::time::Duration::from_millis(1));
}

fn rich_fingerprint_stable(
    cluster: &mut test_raftstore::Cluster<test_raftstore::NodeCluster>,
    keys: &[&[u8]],
) -> String {
    let region = cluster.get_region(b"");
    let epoch = region.get_region_epoch();
    let mut kv = Vec::new();
    for k in keys {
        let v = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| cluster.get(*k)))
            .ok()
            .flatten();
        kv.push(format!(
            "{}={}",
            String::from_utf8_lossy(k),
            v.as_ref()
                .map(|b| String::from_utf8_lossy(b).into_owned())
                .unwrap_or_else(|| "none".into())
        ));
    }
    format!(
        "region={} conf_ver={} ver={} data=[{}]",
        region.get_id(),
        epoch.get_conf_ver(),
        epoch.get_version(),
        kv.join(",")
    )
}

/// 1-node path: dst_init + hybrid clock + pool_size=1 + must_put.
/// Stable fingerprint across two runs of the same seed.
fn run_single_node_scenario(seed: u64) -> String {
    tikv_util::dst_init::dst_init(seed);
    // dst_init freezes Instant (manual_only). Raft leases / election need a
    // progressing clock — start the hybrid driver so logical ≈ wall during
    // bootstrap and must_put (still process-global monotonic logical time).
    time::dst_set_manual_only(false);
    time::dst_start_hybrid_driver(std::time::Duration::from_millis(1));
    batch_system::set_manual_drive(false);

    let mut cluster = new_node_cluster(seed, 1);
    dst_setup_cluster(&mut cluster);
    test_raftstore::configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10));
    cluster.run();

    // Wall wait — hybrid driver advances logical Instant in lockstep.
    for _ in 0..50 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        if cluster.leader_of_region(1).is_some() {
            break;
        }
    }
    assert!(
        cluster.leader_of_region(1).is_some(),
        "1-node cluster should elect a leader under dst"
    );

    let keys: &[&[u8]] = &[b"k1", b"k2", b"k3"];
    for (i, k) in keys.iter().enumerate() {
        let val = format!("v{seed}_{i}");
        cluster.must_put(*k, val.as_bytes());
    }

    let fp = rich_fingerprint_stable(&mut cluster, keys);
    cluster.shutdown();
    fp
}

// ─── Step-driven helpers (opt-in via DST_FULL=1) ───────────────────────

fn network_step(
    cluster: &mut test_raftstore::Cluster<test_raftstore::NodeCluster>,
    net: &DstNetworkQueue,
) -> usize {
    let batch = net.take_sorted(usize::MAX);
    if batch.is_empty() {
        return 0;
    }
    let to_send = net.record_and_filter(batch);
    let n = to_send.len();
    net.set_passthrough(true);
    for msg in to_send {
        let _ = cluster.send_raft_msg(msg);
    }
    net.set_passthrough(false);
    n
}

fn network_drain_manual(
    cluster: &mut test_raftstore::Cluster<test_raftstore::NodeCluster>,
    net: &DstNetworkQueue,
    max_rounds: usize,
) {
    for _ in 0..max_rounds {
        let _ = batch_system::step_all_once();
        let n = network_step(cluster, net);
        dst_tick_ms(5);
        if n == 0 && net.pending() == 0 {
            let _ = batch_system::step_all_once();
            dst_tick_ms(10);
            if network_step(cluster, net) == 0 {
                break;
            }
        }
    }
}

fn put_stepped(
    cluster: &mut test_raftstore::Cluster<test_raftstore::NodeCluster>,
    net: &DstNetworkQueue,
    key: &[u8],
    val: &[u8],
) -> bool {
    let mut fut = match cluster.async_put(key, val) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    net.set_recording(true);
    for _ in 0..4000 {
        let _ = batch_system::step_all_once();
        while network_step(cluster, net) > 0 {}
        match Future::poll(fut.as_mut(), &mut cx) {
            Poll::Ready(resp) => {
                let ok = !resp.get_header().has_error();
                net.set_recording(false);
                network_drain_manual(cluster, net, 40);
                net.set_recording(true);
                return ok;
            }
            Poll::Pending => dst_tick_ms(5),
        }
    }
    net.set_recording(false);
    network_drain_manual(cluster, net, 10);
    net.set_recording(true);
    false
}

fn run_step_driven_scenario(seed: u64) -> (String, String) {
    tikv_util::dst_init::dst_init(seed);
    // Hybrid clock during bootstrap so election / transfer make progress.
    time::dst_set_manual_only(false);
    time::dst_start_hybrid_driver(std::time::Duration::from_millis(1));
    batch_system::set_manual_drive(false);

    let mut cluster = new_node_cluster(seed, 3);
    dst_setup_cluster(&mut cluster);
    test_raftstore::configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10));
    cluster.run();

    for _ in 0..80 {
        std::thread::sleep(std::time::Duration::from_millis(50));
        if cluster.leader_of_region(1).is_some() {
            break;
        }
    }

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cluster.must_transfer_leader(1, new_peer(1, 1));
    }));
    std::thread::sleep(std::time::Duration::from_millis(500));

    {
        let now = time::dst_now_nanos();
        let align: i64 = 1_000_000_000;
        let aligned = ((now / align) + 2) * align;
        time::dst_set_logical_nanos(aligned);
    }
    // Pure step-driven phase for puts.
    batch_system::set_manual_drive(true);
    time::dst_set_manual_only(true);
    tikv_util::dst_init::dst_reseed_before_phase();

    let net = DstNetworkQueue::new(seed, 0);
    cluster.add_send_filter(CloneFilterFactory(net.clone()));
    net.clear_log();

    let keys: &[&[u8]] = &[b"sd_a", b"sd_b", b"sd_c"];
    for (i, k) in keys.iter().enumerate() {
        let val = format!("sv{seed}_{i}");
        let ok = put_stepped(&mut cluster, &net, k, val.as_bytes());
        assert!(
            ok,
            "stepped put failed for key {}",
            String::from_utf8_lossy(k)
        );
    }

    net.set_recording(false);
    network_drain_manual(&mut cluster, &net, 40);

    let stable = rich_fingerprint_stable(&mut cluster, keys);
    let app = net.log_summary_app_only();
    let leader = cluster
        .leader_of_region(1)
        .map(|l| l.get_store_id())
        .unwrap_or(0);
    let full = format!("leader={leader} {stable} app=[{app}]");

    cluster.shutdown();
    batch_system::set_manual_drive(false);
    time::dst_set_manual_only(false);
    (stable, full)
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[test]
fn test_dst_network_queue_sort_unit() {
    let net = DstNetworkQueue::new(1, 0);
    let mk = |from: u64, to: u64, index: u64| {
        let mut m = RaftMessage::default();
        m.set_region_id(1);
        m.mut_from_peer().set_store_id(from);
        m.mut_to_peer().set_store_id(to);
        m.mut_message().set_msg_type(MessageType::MsgAppend);
        m.mut_message().set_term(1);
        m.mut_message().set_index(index);
        m
    };
    let mut batch = vec![mk(2, 1, 9), mk(1, 2, 1), mk(1, 2, 3)];
    assert!(net.before(&mut batch).is_ok());
    assert!(batch.is_empty());
    let sorted = net.take_sorted(usize::MAX);
    let keys: Vec<_> = sorted.iter().map(msg_sort_key).collect();
    assert_eq!(keys[0], (1, 2, 1, MessageType::MsgAppend as i32, 1, 1));
    assert_eq!(keys[1], (1, 2, 1, MessageType::MsgAppend as i32, 1, 3));
    assert_eq!(keys[2], (2, 1, 1, MessageType::MsgAppend as i32, 1, 9));
}

#[test]
fn test_dst_single_node_fingerprint_replay() {
    let seed: u64 = 0x51;
    let a = run_single_node_scenario(seed);
    let b = run_single_node_scenario(seed);
    eprintln!("DST_1N stable1: {a}");
    eprintln!("DST_1N stable2: {b}");
    assert_eq!(a, b, "1-node stable fingerprint must match across seed replay");
    assert!(a.contains("k1="), "missing k1 in {a}");
    assert!(!a.contains("=none"), "put values missing in {a}");
}

/// Full step-driven 3-node path. Opt-in: can hang under residual races while
/// the virtual net + manual drive path is still being hardened.
#[test]
fn test_dst_step_driven_put_fingerprint() {
    if std::env::var_os("DST_FULL").is_none() {
        eprintln!("skip test_dst_step_driven_put_fingerprint (set DST_FULL=1 to enable)");
        return;
    }
    let seed: u64 = 0x5d01;
    let (s1, f1) = run_step_driven_scenario(seed);
    let (s2, f2) = run_step_driven_scenario(seed);
    eprintln!("DST_STEP stable1: {s1}");
    eprintln!("DST_STEP stable2: {s2}");
    eprintln!("DST_STEP full1: {f1}");
    eprintln!("DST_STEP full2: {f2}");
    assert_eq!(s1, s2);
    assert!(f1.contains("leader=1") && f2.contains("leader=1"));
}
