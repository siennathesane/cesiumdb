//! CesiumDB benchmark binary — db_bench equivalent
//!
//! Usage:
//!   cargo run --release --bin bench -- --benchmarks=fillrandom,stats
//! --db=/tmp/cesiumdb_bench   cargo run --release --bin bench --
//! --benchmarks=readwhilewriting --db=/tmp/cesiumdb_bench --duration=60
//! --threads=8

use std::{
    env,
    fs,
    io::Write,
    ops::Bound,
    path::PathBuf,
    sync::{
        Arc,
        Mutex,
        atomic::{
            AtomicBool,
            AtomicU64,
            Ordering,
        },
    },
    thread,
    time::{
        Duration,
        Instant,
        SystemTime,
    },
};

use cesiumdb::{
    Db,
    DbOptions,
};
use rand::{
    Rng,
    SeedableRng,
    rngs::StdRng,
};

// ============================================================================
// CLI Arguments
// ============================================================================

struct Args {
    benchmarks: Vec<String>,
    db: String,
    num: u64,
    key_size: usize,
    value_size: usize,
    threads: usize,
    duration: u64,
    writes: u64,
    use_existing_db: bool,
    sync: bool,
    seed: u64,
    report_file: Option<String>,
    memtable_size: u64,
    max_memtables: u64,
    block_size: u64,
    target_segment_size: u64,
    target_file_size_multiplier: u64,
    l0_trigger: usize,
    l0_stop: usize,
    max_background_jobs: usize,
    max_db_size_gb: u64,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            benchmarks: vec!["fillseq".to_string()],
            db: "/tmp/cesiumdb_bench".to_string(),
            num: 1_000_000,
            key_size: 20,
            value_size: 400,
            threads: 1,
            duration: 0,
            writes: 0,
            use_existing_db: false,
            sync: false,
            seed: 0,
            report_file: None,
            memtable_size: 64 * 1024 * 1024,
            max_memtables: 8,
            block_size: 4096,
            target_segment_size: 64 * 1024 * 1024,
            target_file_size_multiplier: 1,
            l0_trigger: 8,
            l0_stop: 16,
            max_background_jobs: 8,
            max_db_size_gb: 0,
        }
    }
}

fn next_arg(i: &mut usize) -> String {
    *i += 1;
    env::args().nth(*i).unwrap_or_else(|| {
        eprintln!("Expected value after argument");
        std::process::exit(1);
    })
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let mut i = 1;
    while i < env::args().len() {
        let raw = env::args().nth(i).unwrap();
        // Support both --key=value and --key value
        let (key, mut value) = if let Some(eq) = raw.find('=') {
            (raw[..eq].to_string(), Some(raw[eq + 1..].to_string()))
        } else {
            (raw.clone(), None)
        };

        let get_val = |v: &mut Option<String>, i: &mut usize| -> String {
            v.take().unwrap_or_else(|| next_arg(i))
        };

        match key.as_str() {
            | "--benchmarks" => {
                let v = get_val(&mut value, &mut i);
                args.benchmarks = v.split(',').map(|s| s.to_string()).collect();
            },
            | "--db" => args.db = get_val(&mut value, &mut i),
            | "--num" => args.num = get_val(&mut value, &mut i).parse().unwrap(),
            | "--key_size" => args.key_size = get_val(&mut value, &mut i).parse().unwrap(),
            | "--value_size" => args.value_size = get_val(&mut value, &mut i).parse().unwrap(),
            | "--threads" => args.threads = get_val(&mut value, &mut i).parse().unwrap(),
            | "--duration" => args.duration = get_val(&mut value, &mut i).parse().unwrap(),
            | "--writes" => args.writes = get_val(&mut value, &mut i).parse().unwrap(),
            | "--use_existing_db" => {
                args.use_existing_db = get_val(&mut value, &mut i) == "1";
            },
            | "--sync" => args.sync = get_val(&mut value, &mut i) == "1",
            | "--seed" => args.seed = get_val(&mut value, &mut i).parse().unwrap(),
            | "--report_file" => args.report_file = Some(get_val(&mut value, &mut i)),
            | "--memtable_size" => {
                args.memtable_size = get_val(&mut value, &mut i).parse().unwrap();
            },
            | "--max_memtables" => {
                args.max_memtables = get_val(&mut value, &mut i).parse().unwrap();
            },
            | "--block_size" => args.block_size = get_val(&mut value, &mut i).parse().unwrap(),
            | "--target_segment_size" => {
                args.target_segment_size = get_val(&mut value, &mut i).parse().unwrap();
            },
            | "--target_file_size_multiplier" => {
                args.target_file_size_multiplier = get_val(&mut value, &mut i).parse().unwrap();
            },
            | "--l0_trigger" => args.l0_trigger = get_val(&mut value, &mut i).parse().unwrap(),
            | "--l0_stop" => args.l0_stop = get_val(&mut value, &mut i).parse().unwrap(),
            | "--max_background_jobs" => {
                args.max_background_jobs = get_val(&mut value, &mut i).parse().unwrap();
            },
            | "--max_db_size_gb" => {
                args.max_db_size_gb = get_val(&mut value, &mut i).parse().unwrap();
            },
            | _ => {
                eprintln!("Unknown argument: {}", raw);
                std::process::exit(1);
            },
        }
        i += 1;
    }
    args
}

// ============================================================================
// Key / Value Generation
// ============================================================================

fn generate_key(key_size: usize, idx: u64) -> Vec<u8> {
    let mut key = format!("user{}", idx).into_bytes();
    key.resize(key_size, b'0');
    key
}

fn generate_value(value_size: usize, seed: u64) -> Vec<u8> {
    let mut data = vec![0u8; value_size];
    let mut s = seed;
    for byte in data.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        *byte = (s >> 32) as u8;
    }
    data
}

// ============================================================================
// Benchmark Metrics
// ============================================================================

struct BenchmarkResult {
    name: String,
    ops: u64,
    seconds: f64,
    bytes: u64,
    latencies: Vec<u64>, // microseconds
    /// For mixed workloads (e.g. readwhilewriting), track read ops separately
    /// so we can report reads/sec matching RocksDB semantics.
    read_ops: u64,
}

impl BenchmarkResult {
    fn ops_per_sec(&self) -> f64 {
        self.ops as f64 / self.seconds.max(0.0001)
    }

    fn mb_per_sec(&self) -> f64 {
        (self.bytes as f64 / (1024.0 * 1024.0)) / self.seconds.max(0.0001)
    }

    fn micros_per_op(&self) -> f64 {
        (self.seconds * 1_000_000.0) / self.ops.max(1) as f64
    }

    fn p50(&self) -> f64 {
        self.percentile(0.50)
    }

    fn p99(&self) -> f64 {
        self.percentile(0.99)
    }

    fn p999(&self) -> f64 {
        self.percentile(0.999)
    }

    fn p9999(&self) -> f64 {
        self.percentile(0.9999)
    }

    fn percentile(&self, p: f64) -> f64 {
        if self.latencies.is_empty() {
            return 0.0;
        }
        let idx = ((self.latencies.len() as f64 - 1.0) * p) as usize;
        self.latencies[idx] as f64
    }

    fn print(&self) {
        // For readwhilewriting, report reads/sec to match standard benchmark semantics.
        let (ops_label, ops_display, rate_label) = if self.read_ops > 0 {
            (
                self.read_ops,
                format!("{} reads + {} writes", self.read_ops, self.ops - self.read_ops),
                "reads/sec",
            )
        } else {
            (self.ops, format!("{}", self.ops), "ops/sec")
        };
        let ops_per_sec = ops_label as f64 / self.seconds.max(0.0001);
        println!(
            "{:>14} : {:>10.3} micros/op {:>10.0} {} {:>10.3} seconds {:>15} operations; {:>10.1} MB/s",
            self.name,
            self.seconds * 1_000_000.0 / ops_label.max(1) as f64,
            ops_per_sec,
            rate_label,
            self.seconds,
            ops_display,
            self.mb_per_sec()
        );
        if !self.latencies.is_empty() {
            println!(
                "               P50: {:.1} us  P99: {:.1} us  P99.9: {:.1} us  P99.99: {:.1} us",
                self.p50(),
                self.p99(),
                self.p999(),
                self.p9999()
            );
        }
    }
}

fn sample_latency(latencies: &Mutex<Vec<u64>>, micros: u64, max_samples: usize) {
    let mut vec = latencies.lock().unwrap();
    if vec.len() < max_samples {
        vec.push(micros);
    } else {
        // Reservoir sampling: replace random element with decreasing probability
        let idx = (micros as usize) % vec.len();
        if idx < max_samples {
            vec[idx] = micros;
        }
    }
}

fn sort_latencies(latencies: &Mutex<Vec<u64>>) -> Vec<u64> {
    let mut vec = latencies.lock().unwrap().clone();
    vec.sort_unstable();
    vec
}

// ============================================================================
// Benchmark Runners
// ============================================================================

fn open_db(args: &Args) -> Arc<Db> {
    let mut opts = DbOptions::default();
    opts.data_dir(PathBuf::from(&args.db))
        .memtable_size(args.memtable_size)
        .max_memtables(args.max_memtables)
        .target_segment_size(args.target_segment_size)
        .target_file_size_multiplier(args.target_file_size_multiplier)
;

    let mut scheduler = cesiumdb::compaction::SchedulerConfig::default();
    scheduler.l0_compaction_trigger = args.l0_trigger;
    scheduler.l0_stop_writes_trigger = args.l0_stop;
    scheduler.max_concurrent_jobs = args.max_background_jobs;
    scheduler.target_segment_size = args.target_segment_size;
    scheduler.target_file_size_multiplier = args.target_file_size_multiplier;
    opts.scheduler_config(scheduler);

    // Note: block_size isn't directly exposed on DbOptions yet.
    Db::open(opts).unwrap()
}

fn run_fillseq(db: &Arc<Db>, args: &Args) -> BenchmarkResult {
    let num = if args.writes > 0 {
        args.writes
    } else {
        args.num
    };
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(1_000_000)));
    let start = Instant::now();

    for i in 0..num {
        let key = generate_key(args.key_size, i);
        let value = generate_value(args.value_size, i);
        let op_start = Instant::now();
        db.put(&key, &value).unwrap();
        if args.sync {
            db.sync().unwrap();
        }
        sample_latency(&latencies, op_start.elapsed().as_micros() as u64, 1_000_000);
    }

    let elapsed = start.elapsed().as_secs_f64();
    BenchmarkResult {
        name: "fillseq".to_string(),
        ops: num,
        seconds: elapsed,
        bytes: num * args.value_size as u64,
        latencies: sort_latencies(&latencies),
        read_ops: 0,
    }
}

fn run_fillrandom(db: &Arc<Db>, args: &Args) -> BenchmarkResult {
    let num = if args.writes > 0 {
        args.writes
    } else {
        args.num
    };
    let threads = args.threads.max(1);
    let ops_per_thread = num / threads as u64;
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(1_000_000)));
    let start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..threads)
        .map(|tid| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let latencies = latencies.clone();
            let key_size = args.key_size;
            let value_size = args.value_size;
            let seed = args.seed.wrapping_add(tid as u64);
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut local = Vec::with_capacity(100_000);
                for _ in 0..ops_per_thread {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let idx = rng.random::<u64>() % num;
                    let key = generate_key(key_size, idx);
                    let value = generate_value(value_size, rng.random::<u64>());
                    let op_start = Instant::now();
                    db.put(&key, &value).unwrap();
                    local.push(op_start.elapsed().as_micros() as u64);
                    if local.len() >= 100_000 {
                        sample_batch(&latencies, &mut local);
                    }
                }
                if !local.is_empty() {
                    sample_batch(&latencies, &mut local);
                }
            })
        })
        .collect();

    if args.duration > 0 {
        thread::sleep(Duration::from_secs(args.duration));
        shutdown.store(true, Ordering::Relaxed);
    }

    for w in workers {
        let _ = w.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let actual_ops = if args.duration > 0 {
        // Approximate: we don't track exact ops when duration-limited in multi-thread
        (elapsed * (ops_per_thread * threads as u64) as f64 / args.duration.max(1) as f64) as u64
    } else {
        num
    };

    BenchmarkResult {
        name: "fillrandom".to_string(),
        ops: actual_ops,
        seconds: elapsed,
        bytes: actual_ops * args.value_size as u64,
        latencies: sort_latencies(&latencies),
        read_ops: 0,
    }
}

fn sample_batch(latencies: &Mutex<Vec<u64>>, local: &mut Vec<u64>) {
    let mut vec = latencies.lock().unwrap();
    for &micros in local.iter() {
        if vec.len() < 1_000_000 {
            vec.push(micros);
        } else {
            let idx = (micros as usize) % vec.len();
            if idx < 1_000_000 {
                vec[idx] = micros;
            }
        }
    }
    local.clear();
}

fn run_overwrite(db: &Arc<Db>, args: &Args) -> BenchmarkResult {
    // Overwrite is the same as fillrandom but on existing keys
    run_fillrandom(db, args)
}

fn run_readrandom(db: &Arc<Db>, args: &Args) -> BenchmarkResult {
    let num = args.num;
    let threads = args.threads.max(1);
    let ops_per_thread = num / threads as u64;
    let found = Arc::new(AtomicU64::new(0));
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(1_000_000)));
    let start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..threads)
        .map(|tid| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let found = found.clone();
            let latencies = latencies.clone();
            let key_size = args.key_size;
            let seed = args.seed.wrapping_add(tid as u64);
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut local = Vec::with_capacity(100_000);
                for _ in 0..ops_per_thread {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let idx = rng.random::<u64>() % num;
                    let key = generate_key(key_size, idx);
                    let op_start = Instant::now();
                    if let Ok(Some(_)) = db.get(&key) {
                        found.fetch_add(1, Ordering::Relaxed);
                    }
                    local.push(op_start.elapsed().as_micros() as u64);
                    if local.len() >= 100_000 {
                        sample_batch(&latencies, &mut local);
                    }
                }
                if !local.is_empty() {
                    sample_batch(&latencies, &mut local);
                }
            })
        })
        .collect();

    if args.duration > 0 {
        thread::sleep(Duration::from_secs(args.duration));
        shutdown.store(true, Ordering::Relaxed);
    }

    for w in workers {
        let _ = w.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let actual_ops = if args.duration > 0 {
        (elapsed * (ops_per_thread * threads as u64) as f64 / args.duration.max(1) as f64) as u64
    } else {
        num
    };

    BenchmarkResult {
        name: "readrandom".to_string(),
        ops: actual_ops,
        seconds: elapsed,
        bytes: actual_ops * args.value_size as u64,
        latencies: sort_latencies(&latencies),
        read_ops: 0,
    }
}

fn run_readwhilewriting(db: &Arc<Db>, args: &Args) -> BenchmarkResult {
    let num = args.num;
    let threads = args.threads.max(2);
    let read_threads = threads / 2;
    let write_threads = threads - read_threads;
    let duration = if args.duration > 0 { args.duration } else { 30 };

    let read_ops = Arc::new(AtomicU64::new(0));
    let write_ops = Arc::new(AtomicU64::new(0));
    let found = Arc::new(AtomicU64::new(0));
    let shutdown = Arc::new(AtomicBool::new(false));
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(1_000_000)));

    let start = Instant::now();

    // Spawn readers
    let mut workers = Vec::new();
    for tid in 0..read_threads {
        let db = db.clone();
        let shutdown = shutdown.clone();
        let read_ops = read_ops.clone();
        let found = found.clone();
        let latencies = latencies.clone();
        let key_size = args.key_size;
        let seed = args.seed.wrapping_add(tid as u64).wrapping_add(1_000_000);
        workers.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(seed);
            let mut local = Vec::with_capacity(100_000);
            while !shutdown.load(Ordering::Relaxed) {
                let idx = rng.random::<u64>() % num;
                let key = generate_key(key_size, idx);
                let op_start = Instant::now();
                if let Ok(Some(_)) = db.get(&key) {
                    found.fetch_add(1, Ordering::Relaxed);
                }
                read_ops.fetch_add(1, Ordering::Relaxed);
                local.push(op_start.elapsed().as_micros() as u64);
                if local.len() >= 100_000 {
                    sample_batch(&latencies, &mut local);
                }
            }
            if !local.is_empty() {
                sample_batch(&latencies, &mut local);
            }
        }));
    }

    // Spawn writers
    for tid in 0..write_threads {
        let db = db.clone();
        let shutdown = shutdown.clone();
        let write_ops = write_ops.clone();
        let key_size = args.key_size;
        let value_size = args.value_size;
        let seed = args.seed.wrapping_add(tid as u64).wrapping_add(2_000_000);
        workers.push(thread::spawn(move || {
            let mut rng = StdRng::seed_from_u64(seed);
            while !shutdown.load(Ordering::Relaxed) {
                let idx = rng.random::<u64>() % num;
                let key = generate_key(key_size, idx);
                let value = generate_value(value_size, rng.random::<u64>());
                let _ = db.put(&key, &value);
                write_ops.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }

    thread::sleep(Duration::from_secs(duration));
    shutdown.store(true, Ordering::Relaxed);

    for w in workers {
        let _ = w.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_ops = read_ops.load(Ordering::Relaxed) + write_ops.load(Ordering::Relaxed);

    BenchmarkResult {
        name: "readwhilewriting".to_string(),
        ops: total_ops,
        seconds: elapsed,
        bytes: write_ops.load(Ordering::Relaxed) * args.value_size as u64,
        latencies: sort_latencies(&latencies),
        read_ops: read_ops.load(Ordering::Relaxed),
    }
}

fn run_seekrandom(db: &Arc<Db>, args: &Args) -> BenchmarkResult {
    let num = args.num;
    let threads = args.threads.max(1);
    let ops_per_thread = num / threads as u64;
    let duration = args.duration;
    let latencies = Arc::new(Mutex::new(Vec::with_capacity(1_000_000)));
    let start = Instant::now();
    let shutdown = Arc::new(AtomicBool::new(false));

    let workers: Vec<_> = (0..threads)
        .map(|tid| {
            let db = db.clone();
            let shutdown = shutdown.clone();
            let latencies = latencies.clone();
            let key_size = args.key_size;
            let seed = args.seed.wrapping_add(tid as u64);
            thread::spawn(move || {
                let mut rng = StdRng::seed_from_u64(seed);
                let mut local = Vec::with_capacity(100_000);
                for _ in 0..ops_per_thread {
                    if shutdown.load(Ordering::Relaxed) {
                        break;
                    }
                    let idx = rng.random::<u64>() % num;
                    let key = generate_key(key_size, idx);
                    let op_start = Instant::now();
                    let _count = db
                        .scan(Bound::Included(&key), Bound::Unbounded)
                        .unwrap()
                        .take(10)
                        .count();
                    local.push(op_start.elapsed().as_micros() as u64);
                    if local.len() >= 100_000 {
                        sample_batch(&latencies, &mut local);
                    }
                }
                if !local.is_empty() {
                    sample_batch(&latencies, &mut local);
                }
            })
        })
        .collect();

    if duration > 0 {
        thread::sleep(Duration::from_secs(duration));
        shutdown.store(true, Ordering::Relaxed);
    }

    for w in workers {
        let _ = w.join();
    }

    let elapsed = start.elapsed().as_secs_f64();
    let actual_ops = if duration > 0 {
        (elapsed * (ops_per_thread * threads as u64) as f64 / duration.max(1) as f64) as u64
    } else {
        num
    };

    BenchmarkResult {
        name: "seekrandom".to_string(),
        ops: actual_ops,
        seconds: elapsed,
        bytes: actual_ops * args.value_size as u64,
        latencies: sort_latencies(&latencies),
        read_ops: 0,
    }
}

fn run_stats(db: &Arc<Db>) {
    println!("\n=== DB Statistics ===");
    let vstats = db.version_stats();
    println!("Version sequence: {}", vstats.sequence);
    println!("Total segments: {}", vstats.total_segments);
    println!(
        "Total size: {} bytes ({:.2} GB)",
        vstats.total_size,
        vstats.total_size as f64 / (1024.0 * 1024.0 * 1024.0)
    );
    println!("L0 segments: {}", vstats.l0_segments);
    println!("Num levels: {}", vstats.num_levels);

    if let Ok(stats) = db.compaction_stats() {
        println!("\n=== Compaction Statistics ===");
        println!("{}", stats);
    }

    let read_amp = db.read_amp_stats();
    println!("\n=== Read Amplification ===");
    println!("Total gets: {}", read_amp.total_gets);
    println!("L0 segments checked: {}", read_amp.l0_segments_checked);
    println!("LN segments checked: {}", read_amp.ln_segments_checked);
    if read_amp.total_gets > 0 {
        println!(
            "Avg L0 checks per get: {:.2}",
            read_amp.l0_segments_checked as f64 / read_amp.total_gets as f64
        );
        println!(
            "Avg LN checks per get: {:.2}",
            read_amp.ln_segments_checked as f64 / read_amp.total_gets as f64
        );
    }
    println!();
}

fn run_flush(db: &Arc<Db>) {
    println!("Flushing memtables...");
    let start = Instant::now();
    db.sync().unwrap();
    println!("Flush completed in {:.3}s", start.elapsed().as_secs_f64());
}

fn run_waitforcompaction(db: &Arc<Db>, args: &Args) {
    println!("Waiting for compactions to finish...");
    let timeout = if args.duration > 0 {
        args.duration
    } else {
        300
    };
    let start = Instant::now();
    loop {
        if let Ok(stats) = db.compaction_stats() {
            if stats.queued_jobs == 0 && stats.in_progress_jobs == 0 {
                println!(
                    "Compactions finished in {:.1}s",
                    start.elapsed().as_secs_f64()
                );
                return;
            }
        }
        if start.elapsed().as_secs() >= timeout {
            println!("Timeout waiting for compactions");
            return;
        }
        thread::sleep(Duration::from_secs(1));
    }
}

fn run_benchmark(bench_name: &str, db: &Arc<Db>, args: &Args) -> Option<BenchmarkResult> {
    match bench_name {
        | "fillseq" => Some(run_fillseq(db, args)),
        | "fillrandom" => Some(run_fillrandom(db, args)),
        | "overwrite" => Some(run_overwrite(db, args)),
        | "readrandom" => Some(run_readrandom(db, args)),
        | "readwhilewriting" => Some(run_readwhilewriting(db, args)),
        | "seekrandom" => Some(run_seekrandom(db, args)),
        | "stats" => {
            run_stats(db);
            None
        },
        | "flush" => {
            run_flush(db);
            None
        },
        | "waitforcompaction" => {
            run_waitforcompaction(db, args);
            None
        },
        | _ => {
            eprintln!("Unknown benchmark: {}", bench_name);
            None
        },
    }
}

// ============================================================================
// Report / TSV Output
// ============================================================================

fn write_report(report_file: &str, results: &[BenchmarkResult], args: &Args) {
    let mut file = fs::File::create(report_file).unwrap();
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    writeln!(file, "# CesiumDB benchmark report — epoch={}", now).unwrap();
    writeln!(
        file,
        "# db={} num={} key_size={} value_size={} threads={}",
        args.db, args.num, args.key_size, args.value_size, args.threads
    )
    .unwrap();
    writeln!(
        file,
        "# ops_sec\tmb_sec\tmicros_op\tp50\tp99\tp99.9\tp99.99\ttest"
    )
    .unwrap();

    for r in results {
        writeln!(
            file,
            "{:.0}\t{:.1}\t{:.3}\t{:.1}\t{:.1}\t{:.1}\t{:.1}\t{}",
            r.ops_per_sec(),
            r.mb_per_sec(),
            r.micros_per_op(),
            r.p50(),
            r.p99(),
            r.p999(),
            r.p9999(),
            r.name
        )
        .unwrap();
    }
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let mut args = parse_args();

    if !args.use_existing_db {
        let _ = fs::remove_dir_all(&args.db);
    }
    if args.max_db_size_gb > 0 {
        let max_bytes = args.max_db_size_gb * 1024 * 1024 * 1024;
        let bytes_per_key = (args.key_size + args.value_size) as u64;
        args.num = (max_bytes / bytes_per_key) * 90 / 100; // 10% safety margin for LSM overhead
        println!(
            "MAX_DB_SIZE_GB={} -> NUM_KEYS={} (value_size={} + key_size={})",
            args.max_db_size_gb, args.num, args.value_size, args.key_size
        );
    }

    let db = open_db(&args);

    println!("CesiumDB Benchmark");
    println!("DB: {}", args.db);
    println!(
        "Keys: {}  Key size: {}  Value size: {}",
        args.num, args.key_size, args.value_size
    );
    println!(
        "Threads: {}  Duration: {}s  Writes: {}",
        args.threads, args.duration, args.writes
    );
    println!();

    let mut results = Vec::new();
    for bench_name in &args.benchmarks {
        if let Some(result) = run_benchmark(bench_name, &db, &args) {
            result.print();
            results.push(result);
        }
    }

    if let Some(ref report_file) = args.report_file {
        write_report(report_file, &results, &args);
        println!("Report written to: {}", report_file);
    }

    // Clean up if not using existing DB
    if !args.use_existing_db {
        let _ = db.close();
    }
}
