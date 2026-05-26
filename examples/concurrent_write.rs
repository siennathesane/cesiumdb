use std::thread;

use cesiumdb::{
    Db,
    DbOptions,
};

fn main() {
    let opts = DbOptions::new();
    let db = Db::open(opts).unwrap();

    let key = vec![0u8; 32];
    let val = vec![0u8; 256];

    // Warmup
    for _ in 0..10000 {
        db.put(&key, &val).unwrap();
    }

    let num_threads = [1, 2, 4, 8];
    let ops_per_thread = 100000;

    for &n in &num_threads {
        let db = db.clone();
        let start = std::time::Instant::now();

        let handles: Vec<_> = (0..n)
            .map(|i| {
                let db = db.clone();
                let key = key.clone();
                let val = val.clone();
                thread::spawn(move || {
                    for j in 0..ops_per_thread {
                        let mut k = key.clone();
                        k[0..8].copy_from_slice(&(i as u64).to_le_bytes());
                        k[8..16].copy_from_slice(&(j as u64).to_le_bytes());
                        db.put(&k, &val).unwrap();
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let elapsed = start.elapsed();
        let total_ops = n * ops_per_thread;
        let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();
        println!(
            "{} threads: {:.0} ops/sec ({:?} total)",
            n, ops_per_sec, elapsed
        );
    }
}
