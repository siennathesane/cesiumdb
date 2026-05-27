#!/usr/bin/env bash
# Comparative benchmark: CesiumDB vs RocksDB
# Uses realistic LSM-tree configurations that show good write performance.
#
# Usage: ./compare_bench.sh [fillrandom|overwrite|readrandom|readwhilewriting]

set -euo pipefail

BENCH_TYPE="${1:-fillrandom}"
DB_DIR_CESIUM="/tmp/cesiumdb_cmp"
DB_DIR_ROCKS="/tmp/rocksdb_cmp"

# ---------------------------------------------------------------------------
# Realistic configuration (avoids the pathological equal-size levels)
# ---------------------------------------------------------------------------
NUM_KEYS=11504376        # ~5 GiB with 20-byte keys + 400-byte values
KEY_SIZE=20
VALUE_SIZE=400
THREADS=8
SEED=1779858607

# Memtable settings — larger buffers amortise flush overhead
MEMTABLE_SIZE=134217728  # 128 MiB
MAX_MEMTABLES=4

# Level targeting — exponential growth (multiplier=10) is standard for
# leveled compaction.  Without this every level is the same size and data
# gets rewritten endlessly.
TARGET_SEGMENT_SIZE=67108864   # 64 MiB base file size
TARGET_FILE_SIZE_MULTIPLIER=10 # L2=640M, L3=6.4G, etc.

# L0 tuning — trigger compaction a bit earlier so we don't build a
# massive L0 backlog, but allow a generous stop limit to avoid stalls.
L0_TRIGGER=4
L0_STOP=24

MAX_BACKGROUND_JOBS=8

# Binaries
CESIUM_BENCH="./target/release/bench"
ROCKS_BENCH="../rocksdb/build/db_bench"

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

function check_binaries() {
    if [ ! -x "$CESIUM_BENCH" ]; then
        echo "CesiumDB bench not found. Building..."
        cargo build --release --bin bench
    fi

    if [ ! -x "$ROCKS_BENCH" ]; then
        echo "RocksDB db_bench not found at $ROCKS_BENCH"
        echo "Build it first: cd ../rocksdb && cmake --build build --target db_bench"
        exit 1
    fi
}

function run_cesium() {
    local bench="$1"
    local existing="${2:-0}"
    rm -rf "$DB_DIR_CESIUM"

    $CESIUM_BENCH \
        --benchmarks="${bench},stats" \
        --db="$DB_DIR_CESIUM" \
        --num="$NUM_KEYS" \
        --key_size="$KEY_SIZE" \
        --value_size="$VALUE_SIZE" \
        --threads="$THREADS" \
        --seed="$SEED" \
        --use_existing_db="$existing" \
        --memtable_size="$MEMTABLE_SIZE" \
        --max_memtables="$MAX_MEMTABLES" \
        --target_segment_size="$TARGET_SEGMENT_SIZE" \
        --target_file_size_multiplier="$TARGET_FILE_SIZE_MULTIPLIER" \
        --l0_trigger="$L0_TRIGGER" \
        --l0_stop="$L0_STOP" \
        --max_background_jobs="$MAX_BACKGROUND_JOBS" \
        2>&1 | tee /tmp/cesium_bench_${bench}.log
}

function run_rocksdb() {
    local bench="$1"
    local existing="${2:-0}"
    rm -rf "$DB_DIR_ROCKS"

    # RocksDB db_bench treats --num as PER THREAD, so we divide by thread count
    local rocks_num=$((NUM_KEYS / THREADS))

    local existing_flag=""
    if [ "$existing" -eq 1 ]; then
        existing_flag="--use_existing_db=1"
    fi

    $ROCKS_BENCH \
        --benchmarks="$bench" \
        --db="$DB_DIR_ROCKS" \
        --num="$rocks_num" \
        --key_size="$KEY_SIZE" \
        --value_size="$VALUE_SIZE" \
        --threads="$THREADS" \
        --seed="$SEED" \
        --write_buffer_size="$MEMTABLE_SIZE" \
        --max_write_buffer_number="$MAX_MEMTABLES" \
        --min_write_buffer_number_to_merge=2 \
        --target_file_size_base="$TARGET_SEGMENT_SIZE" \
        --target_file_size_multiplier="$TARGET_FILE_SIZE_MULTIPLIER" \
        --max_bytes_for_level_base=671088640 \
        --max_bytes_for_level_multiplier=10 \
        --level_compaction_dynamic_level_bytes=true \
        --level0_file_num_compaction_trigger="$L0_TRIGGER" \
        --level0_stop_writes_trigger="$L0_STOP" \
        --max_background_jobs="$MAX_BACKGROUND_JOBS" \
        --compression_type=none \
        $existing_flag \
        2>&1 | tee /tmp/rocksdb_bench_${bench}.log
}

function extract_metric() {
    local log="$1"
    local label="$2"
    grep "$label" "$log" | head -1 | awk '{for(i=1;i<=NF;i++) if($i=="ops/sec" || $i=="reads/sec") print $(i-1)}'
}

function extract_mbs() {
    local log="$1"
    local label="$2"
    grep "$label" "$log" | head -1 | grep -oE '[0-9]+\.[0-9]+ MB/s' | grep -oE '[0-9]+\.[0-9]+' || echo "0"
}

function extract_time() {
    local log="$1"
    local label="$2"
    grep "$label" "$log" | head -1 | grep -oE '[0-9]+\.[0-9]+ seconds' | grep -oE '[0-9]+\.[0-9]+' || echo "0"
}

function extract_p9999() {
    local log="$1"
    grep "P99.99:" "$log" | head -1 | grep -oE 'P99\.99: [0-9]+\.[0-9]+ us' | awk '{print $2}' || echo "0"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

check_binaries

echo "========================================"
echo "Benchmark: $BENCH_TYPE"
echo "Keys: $NUM_KEYS total  |  Key size: $KEY_SIZE  |  Value size: $VALUE_SIZE"
echo "Threads: $THREADS  |  Seed: $SEED"
echo "Memtable: $MEMTABLE_SIZE  |  Max memtables: $MAX_MEMTABLES"
echo "Target segment: $TARGET_SEGMENT_SIZE  |  Multiplier: $TARGET_FILE_SIZE_MULTIPLIER"
echo "L0 trigger: $L0_TRIGGER  |  L0 stop: $L0_STOP"
echo "========================================"
echo ""

# --- CesiumDB ---
echo ">>> Running CesiumDB..."
run_cesium "$BENCH_TYPE" 0
echo ""

# --- RocksDB ---
echo ">>> Running RocksDB..."
run_rocksdb "$BENCH_TYPE" 0
echo ""

# --- Extract & compare ---
CE_OPS=$(extract_metric /tmp/cesium_bench_${BENCH_TYPE}.log "fillrandom")
CE_MB=$(extract_mbs /tmp/cesium_bench_${BENCH_TYPE}.log "fillrandom")
CE_TIME=$(extract_time /tmp/cesium_bench_${BENCH_TYPE}.log "fillrandom")
CE_P9999=$(extract_p9999 /tmp/cesium_bench_${BENCH_TYPE}.log)

RK_OPS=$(extract_metric /tmp/rocksdb_bench_${BENCH_TYPE}.log "fillrandom")
RK_MB=$(extract_mbs /tmp/rocksdb_bench_${BENCH_TYPE}.log "fillrandom")
RK_TIME=$(extract_time /tmp/rocksdb_bench_${BENCH_TYPE}.log "fillrandom")
RK_P9999=$(extract_p9999 /tmp/rocksdb_bench_${BENCH_TYPE}.log)

# If the bench type isn't fillrandom, try the actual bench name
case "$BENCH_TYPE" in
    fillrandom|overwrite|readrandom|readwhilewriting|seekrandom)
        if [ -z "$CE_OPS" ]; then
            CE_OPS=$(extract_metric /tmp/cesium_bench_${BENCH_TYPE}.log "$BENCH_TYPE")
            CE_MB=$(extract_mbs /tmp/cesium_bench_${BENCH_TYPE}.log "$BENCH_TYPE")
            CE_TIME=$(extract_time /tmp/cesium_bench_${BENCH_TYPE}.log "$BENCH_TYPE")
        fi
        if [ -z "$RK_OPS" ]; then
            RK_OPS=$(extract_metric /tmp/rocksdb_bench_${BENCH_TYPE}.log "$BENCH_TYPE")
            RK_MB=$(extract_mbs /tmp/rocksdb_bench_${BENCH_TYPE}.log "$BENCH_TYPE")
            RK_TIME=$(extract_time /tmp/rocksdb_bench_${BENCH_TYPE}.log "$BENCH_TYPE")
        fi
        ;;
esac

echo "========================================"
echo "           RESULTS COMPARISON           "
echo "========================================"
printf "%-20s %15s %15s %10s\n" "Metric" "CesiumDB" "RocksDB" "Winner"
printf "%-20s %15s %15s %10s\n" "------" "--------" "-------" "------"
printf "%-20s %15s %15s %10s\n" "ops/sec" "$CE_OPS" "$RK_OPS" "$(awk -v c="$CE_OPS" -v r="$RK_OPS" 'BEGIN{if(c+0>r+0) print "CesiumDB"; else if(r+0>c+0) print "RocksDB"; else print "Tie"}')"
printf "%-20s %15s %15s %10s\n" "MB/s" "$CE_MB" "$RK_MB" "$(awk -v c="$CE_MB" -v r="$RK_MB" 'BEGIN{if(c+0>r+0) print "CesiumDB"; else if(r+0>c+0) print "RocksDB"; else print "Tie"}')"
printf "%-20s %15s %15s %10s\n" "Time (s)" "$CE_TIME" "$RK_TIME" "$(awk -v c="$CE_TIME" -v r="$RK_TIME" 'BEGIN{if(c+0<r+0) print "CesiumDB"; else if(r+0<c+0) print "RocksDB"; else print "Tie"}')"
printf "%-20s %15s %15s %10s\n" "P99.99 (us)" "$CE_P9999" "$RK_P9999" "$(awk -v c="$CE_P9999" -v r="$RK_P9999" 'BEGIN{if(c+0<r+0) print "CesiumDB"; else if(r+0<c+0) print "RocksDB"; else print "Tie"}')"
echo "========================================"
