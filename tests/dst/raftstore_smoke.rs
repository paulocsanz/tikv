// Copyright 2026 TiKV Project Authors. Licensed under Apache-2.0.
//
// Slim pure-Rust DST smoke for raftstore (feature `dst` required).
//
// Proves:
// 1. DstNetworkQueue hold/sort/release is seed-stable (unit, no cluster).
// 2. 1-node: same seed → same KV fingerprint (hybrid clock).
// 3. 3-node: DstNetworkQueue batch_size=1 + hybrid clock → same KV fingerprint.
// 4. 3-node pure-hold + manual drive → KV + MsgApp ops sequence bit-match.
//
// Run:
//   cargo test -p tests --features "dst,testexport" --test dst_raftstore -- --test-threads=1

#![cfg(feature = "dst")]

use std::{
    task::{Context, Poll},
    time::{Duration, Instant as WallInstant},
};

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

/// Wait until region 1 has a leader (hybrid clock + wall sleep).
fn wait_leader(
    cluster: &mut test_raftstore::Cluster<test_raftstore::NodeCluster>,
    max_iters: usize,
) -> bool {
    for _ in 0..max_iters {
        if cluster.leader_of_region(1).is_some() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    cluster.leader_of_region(1).is_some()
}

/// Bootstrap shared by 3-node scenarios: hybrid clock, pool_size=1, elect, force leader 1.
fn bootstrap_3node(seed: u64) -> test_raftstore::Cluster<test_raftstore::NodeCluster> {
    tikv_util::dst_init::dst_init(seed);
    time::dst_set_manual_only(false);
    time::dst_start_hybrid_driver(Duration::from_millis(1));
    batch_system::set_manual_drive(false);

    let mut cluster = new_node_cluster(seed, 3);
    dst_setup_cluster(&mut cluster);
    test_raftstore::configure_for_lease_read(&mut cluster.cfg, Some(50), Some(10));
    cluster.run();

    assert!(
        wait_leader(&mut cluster, 100),
        "3-node cluster failed to elect a leader under dst"
    );

    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cluster.must_transfer_leader(1, new_peer(1, 1));
    }));
    // Settle transfer under hybrid + auto driver.
    for _ in 0..20 {
        std::thread::sleep(Duration::from_millis(50));
        if cluster
            .leader_of_region(1)
            .map(|p| p.get_store_id() == 1)
            .unwrap_or(false)
        {
            break;
        }
    }
    cluster
}

/// 3-node + DstNetworkQueue(batch_size=1): every send path sorts then releases.
/// Hybrid clock + auto poller driver. Proves ordered virtual net + dst seed
/// yield a stable KV fingerprint without pure-hold (which needs manual drive).
fn run_ordered_net_scenario(seed: u64) -> (String, String) {
    let mut cluster = bootstrap_3node(seed);

    let net = DstNetworkQueue::new(seed, 1);
    cluster.add_send_filter(CloneFilterFactory(net.clone()));
    net.clear_log();

    let keys: &[&[u8]] = &[b"on_a", b"on_b", b"on_c"];
    for (i, k) in keys.iter().enumerate() {
        let val = format!("ov{seed}_{i}");
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            cluster.must_put(*k, val.as_bytes());
        }));
        std::thread::sleep(Duration::from_millis(20));
    }
    std::thread::sleep(Duration::from_millis(200));

    let stable = rich_fingerprint_stable(&mut cluster, keys);
    let app = net.log_summary_app_only();
    let leader = cluster
        .leader_of_region(1)
        .map(|l| l.get_store_id())
        .unwrap_or(0);
    let full = format!("leader={leader} {stable} app=[{app}]");

    cluster.clear_send_filters();
    cluster.shutdown();
    (stable, full)
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
    wall_deadline: WallInstant,
) -> bool {
    let mut fut = match cluster.async_put(key, val) {
        Ok(f) => f,
        Err(_) => return false,
    };
    let waker = futures::task::noop_waker();
    let mut cx = Context::from_waker(&waker);
    net.set_recording(true);
    // Cap iterations + hard wall budget so a stuck pure-hold cannot hang CI.
    for iter in 0..2000 {
        if WallInstant::now() > wall_deadline {
            eprintln!(
                "put_stepped wall deadline hit for key={} after {iter} iters (pending={})",
                String::from_utf8_lossy(key),
                net.pending()
            );
            break;
        }
        let _ = batch_system::step_all_once();
        // Flush production aggressively each poller round.
        for _ in 0..8 {
            if network_step(cluster, net) == 0 {
                break;
            }
        }
        match Future::poll(fut.as_mut(), &mut cx) {
            Poll::Ready(resp) => {
                let ok = !resp.get_header().has_error();
                net.set_recording(false);
                network_drain_manual(cluster, net, 40);
                net.set_recording(true);
                return ok;
            }
            Poll::Pending => {
                // Advance logical time enough for a Raft tick (~50ms lease cfg).
                dst_tick_ms(10);
            }
        }
    }
    net.set_recording(false);
    network_drain_manual(cluster, net, 10);
    net.set_recording(true);
    false
}

/// Returns (stable_kv, full_trace, app_summary, ops_sequence).
fn run_step_driven_scenario(seed: u64) -> (String, String, String, String) {
    let mut cluster = bootstrap_3node(seed);

    // Phase-align clock + freeze hybrid + pause background poller driver.
    {
        let now = time::dst_now_nanos();
        let align: i64 = 1_000_000_000;
        let aligned = ((now / align) + 2) * align;
        time::dst_set_logical_nanos(aligned);
    }
    batch_system::set_manual_drive(true);
    time::dst_set_manual_only(true);
    tikv_util::dst_init::dst_reseed_before_phase();

    // Pure hold — only network_step delivers.
    let net = DstNetworkQueue::new(seed, 0);
    cluster.add_send_filter(CloneFilterFactory(net.clone()));
    net.clear_log();

    let wall_deadline = WallInstant::now() + Duration::from_secs(45);
    let keys: &[&[u8]] = &[b"sd_a", b"sd_b", b"sd_c"];
    for (i, k) in keys.iter().enumerate() {
        let val = format!("sv{seed}_{i}");
        let ok = put_stepped(&mut cluster, &net, k, val.as_bytes(), wall_deadline);
        assert!(
            ok,
            "stepped put failed for key {} (pending={} live_pollers={})",
            String::from_utf8_lossy(k),
            net.pending(),
            batch_system::live_count()
        );
    }

    net.set_recording(false);
    network_drain_manual(&mut cluster, &net, 40);

    let stable = rich_fingerprint_stable(&mut cluster, keys);
    let app = net.log_summary_app_only();
    let ops = net.log_ops_sequence();
    let leader = cluster
        .leader_of_region(1)
        .map(|l| l.get_store_id())
        .unwrap_or(0);
    let full = format!("leader={leader} {stable} app=[{app}] ops_len={}", ops.len());

    cluster.clear_send_filters();
    cluster.shutdown();
    batch_system::set_manual_drive(false);
    time::dst_set_manual_only(false);
    (stable, full, app, ops)
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

/// 3-node + ordered virtual net (batch_size=1). Always-on claim.
#[test]
fn test_dst_ordered_net_3node_fingerprint() {
    let seed: u64 = 0x03d3; // ordered-net
    let (s1, f1) = run_ordered_net_scenario(seed);
    let (s2, f2) = run_ordered_net_scenario(seed);
    eprintln!("DST_ORD stable1: {s1}");
    eprintln!("DST_ORD stable2: {s2}");
    eprintln!("DST_ORD full1: {f1}");
    eprintln!("DST_ORD full2: {f2}");
    assert_eq!(
        s1, s2,
        "3-node ordered-net stable fingerprint must match across seed replay"
    );
    assert!(
        !s1.contains("=none"),
        "puts must land in fingerprint: {s1}"
    );
    // Leader forced to 1 when transfer succeeds; if not, still require data match.
    if f1.contains("leader=1") {
        assert!(f2.contains("leader=1"));
    }
}

/// Pure-hold + manual drive 3-node: KV + MsgApp path summary bit-match.
/// Wall-capped put_stepped so a regression cannot hang CI forever.
#[test]
fn test_dst_step_driven_put_fingerprint() {
    let seed: u64 = 0x5d01;
    let (s1, f1, app1, ops1) = run_step_driven_scenario(seed);
    let (s2, f2, app2, ops2) = run_step_driven_scenario(seed);
    eprintln!("DST_STEP stable1: {s1}");
    eprintln!("DST_STEP stable2: {s2}");
    eprintln!("DST_STEP app1: {app1}");
    eprintln!("DST_STEP app2: {app2}");
    eprintln!("DST_STEP ops_len1={} ops_len2={}", ops1.len(), ops2.len());
    eprintln!("DST_STEP full1: {f1}");
    eprintln!("DST_STEP full2: {f2}");
    assert_eq!(s1, s2, "step-driven stable fingerprint must match");
    assert!(!s1.contains("=none"), "puts missing: {s1}");
    assert!(f1.contains("leader=1") && f2.contains("leader=1"));
    // Stronger: client replication path counts under pure-hold + manual drive.
    assert_eq!(
        app1, app2,
        "MsgApp/AppResp summary must match under pure step-driven schedule"
    );
    assert_eq!(
        ops1, ops2,
        "MsgApp/AppResp ops sequence must match under pure step-driven schedule"
    );
}
