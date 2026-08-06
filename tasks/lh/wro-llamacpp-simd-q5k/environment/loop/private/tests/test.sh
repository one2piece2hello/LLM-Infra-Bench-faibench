#!/usr/bin/env bash
# Verifier for wro-llamacpp-simd-q5k (ggml Q5_K×Q8_K CPU dot-product; arch/x86/quants.c).
# Type-2 Long-horizon, ACCELERATION lane, COMPILED-SCOPE (candidate quants.c is recompiled every run).
# Scope = ggml/src/ggml-cpu/arch/x86/quants.c.  reward = raw wall speedup (base_ms/cand_ms), noop~1.0.
# Correctness signature = deterministic block dot == 2304.0 (harness calls ggml_cpu_init for the f16 table).
set -uo pipefail
git config --global --add safe.directory '*' 2>/dev/null || true   # crane-appended images are root-owned
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO=/app/repo
BUILD=/app/build
QF="$REPO/ggml/src/ggml-cpu/arch/x86/quants.c"
HARNESS="${WRO_HARNESS:-}"; [ -z "$HARNESS" ] && { [ -f /opt/wro/harness.c ] && HARNESS=/opt/wro/harness.c || HARNESS="$TESTS_DIR/harness.c"; }
BASELINE_SRC="${WRO_BASELINE_QUANTS:-}"; [ -z "$BASELINE_SRC" ] && { [ -f /opt/wro/baseline_quants.c ] && BASELINE_SRC=/opt/wro/baseline_quants.c || BASELINE_SRC="$TESTS_DIR/baseline_quants.c"; }
# self-contained fallback: if no external harness.c is baked alongside, materialize the embedded one.
if [ ! -f "$HARNESS" ]; then
  HARNESS=/tmp/wro_harness.c
  cat > "$HARNESS" <<'WROHARNESS'
// wro-llamacpp-simd-q5k harness. Times q5_K_q8_K (wall) + a deterministic correctness dot.
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
typedef struct { uint16_t d; uint16_t dmin; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; } block_q5_K;
typedef struct { float d; int8_t qs[256]; int16_t bsums[16]; } block_q8_K;
extern void ggml_vec_dot_q5_K_q8_K(int n, float *s, size_t bs, const void *vx, size_t bx,
                 const void *vy, size_t by, int nrc);
extern void ggml_cpu_init(void);   // loads ggml_table_f32_f16 (f16->f32 decode)
// Deterministic block: fixed integer/exact bytes so scalar (generic) and SIMD paths agree bit-for-bit.
static float correctness_dot(void) {
    block_q5_K x; block_q8_K y;
    memset(&x, 0, sizeof(x)); memset(&y, 0, sizeof(y));
    x.d = 0x3C00; x.dmin = 0x0000;
    for (int i = 0; i < 12; i++)  x.scales[i] = 0x11;
    for (int i = 0; i < 32; i++)  x.qh[i]     = 0x00; // 5th bit = 0
    for (int i = 0; i < 128; i++) x.qs[i]     = 0x11; // low 4 bits = 1
    y.d = 1.0f;
    for (int i = 0; i < 256; i++) y.qs[i] = 1;
    for (int i = 0; i < 16; i++)  y.bsums[i] = 16;
    float s = -12345.0f; ggml_vec_dot_q5_K_q8_K(256, &s, 0, &x, 0, &y, 0, 1); return s;
}
int main(int argc, char **argv) {
    ggml_cpu_init();
    int nb = argc > 1 ? atoi(argv[1]) : 8192; int iters = argc > 2 ? atoi(argv[2]) : 3000;
    float cdot = correctness_dot();
    block_q5_K *x = malloc((size_t)nb * sizeof(block_q5_K)); block_q8_K *y = malloc((size_t)nb * sizeof(block_q8_K));
    srand(4321);
    for (int b = 0; b < nb; b++) {
        x[b].d = 0x3C00; x[b].dmin = 0x3C00;
        for (int i = 0; i < 12; i++)  x[b].scales[i] = rand() & 0xFF;
        for (int i = 0; i < 32; i++)  x[b].qh[i]     = rand() & 0xFF;
        for (int i = 0; i < 128; i++) x[b].qs[i]     = rand() & 0xFF;
        y[b].d = 1.0f;
        int gs[16]; memset(gs,0,sizeof(gs));
        for (int i = 0; i < 256; i++) { int8_t q = (int8_t)(rand()&0xFF); y[b].qs[i]=q; gs[i/16]+=q; }
        for (int i = 0; i < 16; i++)  y[b].bsums[i] = (int16_t)gs[i];
    }
    float s = 0; volatile double sink = 0;
    for (int w = 0; w < 30; w++) { ggml_vec_dot_q5_K_q8_K(nb*256, &s, 0, x, 0, y, 0, 1); sink += s; }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int it = 0; it < iters; it++) { ggml_vec_dot_q5_K_q8_K(nb*256, &s, 0, x, 0, y, 0, 1); sink += s; }
    clock_gettime(CLOCK_MONOTONIC, &t1);
    double ms = (t1.tv_sec-t0.tv_sec)*1e3 + (t1.tv_nsec-t0.tv_nsec)/1e6;
    printf("WRO_T5Q5K {\"timing_ms\": %.4f, \"checksum\": %.4f}\n", ms, (double)cdot);
    free(x); free(y); return 0;
}
WROHARNESS
fi
GOLDEN_CHECKSUM=2304.0
LOG=/logs/verifier; mkdir -p "$LOG"
MODE="${KERNELBENCH_VERIFY_MODE:-candidate}"
SCOPE=("ggml/src/ggml-cpu/arch/x86/quants.c")

HARD=(); add_hard(){ HARD+=("$1"); }
correctness_ok=true; scope_ok=true; import_origin_ok=true; benchmark_ok=true
speedup=0.0; ref_speedup=1.0; base_ms=-1; cand_ms=-1; cand_checksum=-1

cd "$REPO" 2>/dev/null || { add_hard "repo_missing"; correctness_ok=false; }
if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  git -C "$REPO" init -q && git -C "$REPO" -c user.email=r@wro -c user.name=wro add -A \
    && git -C "$REPO" -c user.email=r@wro -c user.name=wro commit -qm "ggml degraded baseline" 2>/dev/null || true
fi
mkdir -p "$REPO/.git/info" 2>/dev/null || true
printf '%s\n' 'ggml/tests/' 'ggml/examples/' 'ggml/ggml.pc.in' > "$REPO/.git/info/exclude" 2>/dev/null || true

if [ "$MODE" = "oracle" ]   && [ -n "${KERNELBENCH_ORACLE_PATCH:-}" ];   then
  git -C "$REPO" apply -p1 "$KERNELBENCH_ORACLE_PATCH"   2>"$LOG/oracle_apply.log"   || add_hard "oracle_apply_failed"; fi
if [ "$MODE" = "negative" ] && [ -n "${KERNELBENCH_NEGATIVE_PATCH:-}" ]; then
  git -C "$REPO" apply -p1 "$KERNELBENCH_NEGATIVE_PATCH" 2>"$LOG/negative_apply.log" || true; fi

# ---- scope gate: only compiled-source/CMake outside quants.c hard-fails (ignore non-compiled base noise) ----
git -C "$REPO" status --porcelain 2>/dev/null | awk '{print $2}' > "$LOG/changed_files.txt"
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    ggml/src/ggml-cpu/arch/x86/quants.c) : ;;
    *.c|*.h|*.cpp|*.cc|*.cu|*.cuh|*CMakeLists.txt|*.cmake) scope_ok=false; add_hard "out_of_scope_edit:$f" ;;
    *) : ;;
  esac
done < "$LOG/changed_files.txt"

# ---- configure ggml once (idempotent); build dir persists across submissions ----
if [ ! -f "$BUILD/CMakeCache.txt" ]; then
  mkdir -p "$REPO/ggml/tests" "$REPO/ggml/examples"
  [ -f "$REPO/ggml/tests/CMakeLists.txt" ]    || echo '# empty' > "$REPO/ggml/tests/CMakeLists.txt"
  [ -f "$REPO/ggml/examples/CMakeLists.txt" ] || echo '# empty' > "$REPO/ggml/examples/CMakeLists.txt"
  [ -f "$REPO/ggml/ggml.pc.in" ] || printf 'Name: ggml\nDescription: ggml\nVersion: @GGML_INSTALL_VERSION@\nLibs: -lggml\n' > "$REPO/ggml/ggml.pc.in"
  cmake -S "$REPO/ggml" -B "$BUILD" -DGGML_STANDALONE=OFF -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF \
    -DGGML_AVX2=ON -DGGML_NATIVE=OFF -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DGGML_BACKEND_DL=OFF \
    -DGGML_CCACHE=OFF -DCMAKE_BUILD_TYPE=Release > "$LOG/cmake_cfg.log" 2>&1 || { add_hard "cmake_configure_failed"; correctness_ok=false; }
fi

build_link_run() {  # -> echoes "<checksum> <timing_ms>" or "BUILD_FAIL"/"LINK_FAIL"
  cmake --build "$BUILD" --target ggml-cpu -j"$(nproc)" > "$LOG/build.log" 2>&1 || { echo "BUILD_FAIL"; return; }
  local LIBS; LIBS=$(find "$BUILD" -name 'libggml*.a' 2>/dev/null | tr '\n' ' ')
  gcc "$HARNESS" -I"$REPO/ggml/include" -o /tmp/wro_harness \
      -Wl,--start-group $LIBS -Wl,--end-group -lm -lstdc++ -mavx2 -mfma -mf16c -pthread 2> "$LOG/link.log" \
      || { echo "LINK_FAIL"; return; }
  local OUT; OUT=$(/tmp/wro_harness 8192 3000 2>>"$LOG/run.log")
  echo "$OUT" >> "$LOG/harness_raw.log"
  local CK TM
  CK=$(printf '%s' "$OUT" | sed -n 's/.*"checksum": *\([0-9.eE+-]*\).*/\1/p')
  TM=$(printf '%s' "$OUT" | sed -n 's/.*"timing_ms": *\([0-9.eE+-]*\).*/\1/p')
  echo "${CK:-NaN} ${TM:--1}"
}

if [ "$correctness_ok" = true ] && [ "$scope_ok" = true ]; then
  read -r cand_checksum cand_ms < <(build_link_run)
  if [ "$cand_checksum" = "BUILD_FAIL" ] || [ "$cand_checksum" = "LINK_FAIL" ]; then
    add_hard "candidate_${cand_checksum}"; correctness_ok=false; cand_ms=-1; cand_checksum=-1
  else
    ok=$(python3 -c "print(1 if abs(float('${cand_checksum}')-(${GOLDEN_CHECKSUM}))<1e-2 else 0)" 2>/dev/null || echo 0)
    [ "$ok" = "1" ] || { correctness_ok=false; add_hard "checksum_mismatch:${cand_checksum}!=${GOLDEN_CHECKSUM}"; }
  fi
fi

# ---- baseline: pristine degraded scalar (restore from the baked git baseline), recompiled the same way ----
if [ "$correctness_ok" = true ] && { [ "${#HARD[@]}" -eq 0 ] 2>/dev/null || [ -z "${HARD[*]:-}" ]; }; then
  cp "$QF" /tmp/wro_cand_quants.c 2>/dev/null || true
  git -C "$REPO" checkout -q -- "${SCOPE[@]}" 2>/dev/null || { [ -f "$BASELINE_SRC" ] && cp "$BASELINE_SRC" "$QF"; }
  read -r _bck base_ms < <(build_link_run)
  cp /tmp/wro_cand_quants.c "$QF" 2>/dev/null || true
  if [ "$(python3 -c "print(1 if ${cand_ms}>0 and ${base_ms}>0 else 0)" 2>/dev/null || echo 0)" = "1" ]; then
    speedup=$(python3 -c "print(round(${base_ms}/${cand_ms},6))")
  else benchmark_ok=false; add_hard "timing_invalid"; fi
fi

[ -f /opt/verifier-correctness-manifest.json ] && \
  ref_speedup=$(python3 -c "import json;print(json.load(open('/opt/verifier-correctness-manifest.json')).get('ref_speedup',1.0))" 2>/dev/null || echo 1.0)

reward=0.0; nhard="${#HARD[@]}"
if [ "$nhard" -eq 0 ] && [ "$correctness_ok" = true ]; then reward="$speedup"; fi
hard_str=""; [ "$nhard" -gt 0 ] && hard_str="${HARD[*]}"
export WRO_MODE="$MODE" WRO_REWARD="$reward" WRO_SPEEDUP="$speedup" WRO_BASE_MS="$base_ms" WRO_CAND_MS="$cand_ms" \
       WRO_CK="$cand_checksum" WRO_REF="$ref_speedup" WRO_HARD="$hard_str" WRO_SCOPE="$scope_ok" \
       WRO_IMP="$import_origin_ok" WRO_CORR="$correctness_ok" WRO_BENCH="$benchmark_ok"
python3 - <<'PY'
import os, json
def f(x):
    try: return float(x)
    except: return -1.0
ref=f(os.environ.get("WRO_REF","1")); sp=f(os.environ.get("WRO_SPEEDUP","0"))
print(json.dumps({"mode":os.environ.get("WRO_MODE"),"reward":f(os.environ.get("WRO_REWARD")),"speedup":sp,
  "baseline_ms":f(os.environ.get("WRO_BASE_MS")),"candidate_ms":f(os.environ.get("WRO_CAND_MS")),
  "checksum":f(os.environ.get("WRO_CK")),"ref_speedup":ref,
  "metadata":{"vs_oracle_ratio":(sp/ref) if ref>0 else None},"hard_fails":os.environ.get("WRO_HARD","").split(),
  "gates":{"scope_ok":os.environ.get("WRO_SCOPE")=="true","import_origin_ok":os.environ.get("WRO_IMP")=="true",
           "correctness_ok":os.environ.get("WRO_CORR")=="true","benchmark_ok":os.environ.get("WRO_BENCH")=="true"}}))
PY
