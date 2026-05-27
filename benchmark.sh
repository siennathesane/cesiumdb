#!/usr/bin/env bash
# CesiumDB benchmark script — inspired by RocksDB's benchmark.sh
#
# Usage:
#   ./benchmark.sh <test>
#
# Available tests:
#   fillseq              Sequential fill (single-threaded)
#   fillrandom           Random fill (multi-threaded)
#   overwrite            Random overwrites on existing DB
#   readrandom           Random point reads (multi-threaded)
#   readwhilewriting     Concurrent reads + writes
#   seekrandom           Random range scans
#   stats                Print DB statistics
#   flush                Flush memtables
#   waitforcompaction    Wait for background compactions
#
# Environment variables:
#   DB_DIR               Database directory (default: /tmp/cesiumdb_bench)
#   OUTPUT_DIR           Output directory for reports (default: /tmp)
#   NUM_KEYS             Number of keys (default: 1000000)
#   KEY_SIZE             Key size in bytes (default: 20)
#   VALUE_SIZE           Value size in bytes (default: 400)
#   NUM_THREADS          Number of threads (default: 8)
#   DURATION             Duration in seconds for time-based tests (default: 0)
#   WRITES               Number of writes for count-based tests (default: 0)
#   MEMTABLE_SIZE_MB     Memtable size in MB (default: 64)
#   MAX_MEMTABLES        Max memtables before backpressure (default: 8)
#   L0_TRIGGER           L0 compaction trigger (default: 8)
#   L0_STOP              L0 stop-writes trigger (default: 16)
#   MAX_BACKGROUND_JOBS        Max concurrent compaction jobs (default: 8)
#   TARGET_SST_SIZE_MB         Target SST size in MB (default: 64)
#   TARGET_FILE_SIZE_MULTIPLIER Target file size multiplier per level (default: 1)
#   SEED                       Random seed (default: current time)
#   USE_EXISTING_DB            1 = use existing DB, 0 = clean start (default: 0)
#   BENCH_BIN                  Path to bench binary (default: ./target/release/bench)

set -euo pipefail

# Exit codes
EXIT_INVALID_ARGS=1
EXIT_INVALID_PATH=2

# Size constants
K=1024
M=$((1024 * K))
G=$((1024 * M))

function display_usage() {
    echo "usage: benchmark.sh [--help] <test>"
    echo ""
    echo "Available benchmark tests:"
    echo -e "\tfillseq"
    echo -e "\tfillrandom"
    echo -e "\toverwrite"
    echo -e "\treadrandom"
    echo -e "\treadwhilewriting"
    echo -e "\tseekrandom"
    echo -e "\tstats"
    echo -e "\tflush"
    echo -e "\twaitforcompaction"
    echo ""
    echo "Environment variables:"
    echo -e "\tDB_DIR\t\t\tDatabase directory (default: /tmp/cesiumdb_bench)"
    echo -e "\tOUTPUT_DIR\t\tOutput directory for reports (default: /tmp)"
    echo -e "\tNUM_KEYS\t\tNumber of keys (default: 1000000)"
    echo -e "\tKEY_SIZE\t\tKey size in bytes (default: 20)"
    echo -e "\tVALUE_SIZE\t\tValue size in bytes (default: 400)"
    echo -e "\tNUM_THREADS\t\tNumber of threads (default: 8)"
    echo -e "\tDURATION\t\tDuration in seconds (default: 0)"
    echo -e "\tWRITES\t\t\tNumber of writes (default: 0)"
    echo -e "\tMEMTABLE_SIZE_MB\tMemtable size in MB (default: 64)"
    echo -e "\tMAX_MEMTABLES\t\tMax memtables (default: 8)"
    echo -e "\tL0_TRIGGER\t\tL0 compaction trigger (default: 8)"
    echo -e "\tL0_STOP\t\t\tL0 stop-writes trigger (default: 16)"
    echo -e "\tMAX_BACKGROUND_JOBS\t\tMax concurrent compaction jobs (default: 8)"
    echo -e "\tTARGET_SST_SIZE_MB\t\tTarget SST size in MB (default: 64)"
    echo -e "\tTARGET_FILE_SIZE_MULTIPLIER\tTarget file size multiplier per level (default: 1)"
    echo -e "\tSEED\t\t\t\tRandom seed (default: current time)"
    echo -e "\tUSE_EXISTING_DB\t\t\t1 = use existing DB (default: 0)"
    echo -e "\tBENCH_BIN\t\t\tPath to bench binary (default: ./target/release/bench)"},{
}

if [ $# -lt 1 ]; then
    display_usage
    exit $EXIT_INVALID_ARGS
fi

bench_cmd=$1

if [[ "$bench_cmd" == "--help" ]]; then
    display_usage
    exit 0
fi

# Check for bench binary
bench_bin=${BENCH_BIN:-./target/release/bench}
if [ ! -x "$bench_bin" ]; then
    echo "Bench binary not found at $bench_bin"
    echo "Build it first with: cargo build --release --bin bench"
    exit $EXIT_INVALID_PATH
fi

# Defaults
db_dir=${DB_DIR:-/tmp/cesiumdb_bench}
output_dir=${OUTPUT_DIR:-/tmp}
num_keys=${NUM_KEYS:-1000000}
key_size=${KEY_SIZE:-20}
value_size=${VALUE_SIZE:-400}
num_threads=${NUM_THREADS:-8}
duration=${DURATION:-0}
writes=${WRITES:-0}
# Default to realistic settings that avoid pathological write amplification.
# TARGET_FILE_SIZE_MULTIPLIER=1 creates equal-sized levels and causes
# endless rewrite cycles; 10 is the standard leveled-compaction default.
memtable_size_mb=${MEMTABLE_SIZE_MB:-128}
max_memtables=${MAX_MEMTABLES:-4}
l0_trigger=${L0_TRIGGER:-4}
l0_stop=${L0_STOP:-24}
max_background_jobs=${MAX_BACKGROUND_JOBS:-8}
target_segment_size_mb=${TARGET_SST_SIZE_MB:-64}
target_file_size_multiplier=${TARGET_FILE_SIZE_MULTIPLIER:-10}
seed=${SEED:-$(date +%s)}
use_existing_db=${USE_EXISTING_DB:-0}
max_db_size_gb=${MAX_DB_SIZE_GB:-0}

memtable_size=$((memtable_size_mb * M))
target_segment_size=$((target_segment_size_mb * M))

# If MAX_DB_SIZE_GB is set, compute NUM_KEYS from it
if [ "$max_db_size_gb" -gt 0 ]; then
    max_db_size_bytes=$((max_db_size_gb * G))
    # Account for key overhead (~key_size bytes per key) + value
    bytes_per_key=$((key_size + value_size))
    computed_keys=$((max_db_size_bytes / bytes_per_key))
    # Add a small safety margin (90%) so we don't exceed target with LSM overhead
    computed_keys=$((computed_keys * 90 / 100))
    num_keys=$computed_keys
    echo "MAX_DB_SIZE_GB=${max_db_size_gb} -> NUM_KEYS=${num_keys} (value_size=${value_size}, key_size=${key_size})"
fi

# Common parameters
common_params="
  --db=$db_dir
  --num=$num_keys
  --key_size=$key_size
  --value_size=$value_size
  --threads=$num_threads
  --seed=$seed
  --use_existing_db=$use_existing_db
  --memtable_size=$memtable_size
  --max_memtables=$max_memtables
  --target_segment_size=$target_segment_size
  --target_file_size_multiplier=$target_file_size_multiplier
  --l0_trigger=$l0_trigger
  --l0_stop=$l0_stop
  --max_background_jobs=$max_background_jobs
"

if [ $duration -gt 0 ]; then
    common_params="$common_params --duration=$duration"
fi
if [ $writes -gt 0 ]; then
    common_params="$common_params --writes=$writes"
fi

mkdir -p "$output_dir"

function run_benchmark() {
    local name=$1
    local benchmarks=$2
    local extra_params=${3:-}
    local log_file="$output_dir/benchmark_${name}.log"

    echo "Running $name..."
    echo "Command: $bench_bin --benchmarks=$benchmarks $common_params $extra_params"
    echo "Command: $bench_bin --benchmarks=$benchmarks $common_params $extra_params" > "$log_file"

    $bench_bin \
        --benchmarks="$benchmarks" \
        $common_params \
        $extra_params \
        2>&1 | tee -a "$log_file"
}

# Run the requested benchmark
case $bench_cmd in
    fillseq)
        run_benchmark "fillseq" "fillseq,stats" "--threads=1"
        ;;
    fillrandom)
        run_benchmark "fillrandom" "fillrandom,stats"
        ;;
    overwrite)
        run_benchmark "overwrite" "overwrite,stats" "--use_existing_db=1"
        ;;
    readrandom)
        run_benchmark "readrandom" "readrandom,stats" "--use_existing_db=1"
        ;;
    readwhilewriting)
        run_benchmark "readwhilewriting" "readwhilewriting,stats" "--use_existing_db=1"
        ;;
    seekrandom)
        run_benchmark "seekrandom" "seekrandom,stats" "--use_existing_db=1"
        ;;
    stats)
        run_benchmark "stats" "stats" "--use_existing_db=1"
        ;;
    flush)
        run_benchmark "flush" "flush,stats" "--use_existing_db=1"
        ;;
    waitforcompaction)
        run_benchmark "waitforcompaction" "waitforcompaction,stats" "--use_existing_db=1"
        ;;
    fillseq-readrandom)
        # Combined: load sequentially, then read randomly
        run_benchmark "fillseq" "fillseq" "--threads=1 --use_existing_db=0"
        run_benchmark "readrandom" "readrandom,stats" "--use_existing_db=1"
        ;;
    fillrandom-readwhilewriting)
        # Combined: load randomly, then read-while-writing
        run_benchmark "fillrandom" "fillrandom" "--use_existing_db=0"
        run_benchmark "readwhilewriting" "readwhilewriting,stats" "--use_existing_db=1"
        ;;
    *)
        echo "Unknown benchmark: $bench_cmd"
        display_usage
        exit $EXIT_INVALID_ARGS
        ;;
esac

echo "Benchmark complete. Logs written to $output_dir/benchmark_*.log"
