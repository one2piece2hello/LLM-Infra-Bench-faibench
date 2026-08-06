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
