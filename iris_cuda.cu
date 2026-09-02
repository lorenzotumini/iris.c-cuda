/*
 * Iris CUDA Backend Implementation
 *
 * Implements GPU runtime, memory management, cuBLAS/cuBLASLt matrix multiplication,
 * and kernel dispatching for Iris image generation models.
 */

#include "iris_cuda.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern "C" void iris_softmax_cpu(float *x, int rows, int cols);

/* Forward declarations of CUDA kernel entry points from iris_cuda_kernels.cu */
extern "C" {
void launch_rms_norm_f32(const float *x, const float *weight, float *out, int seq, int hidden, float eps, cudaStream_t stream);
void launch_rms_norm_bf16(const uint16_t *x, const uint16_t *weight, uint16_t *out, int seq, int hidden, float eps, cudaStream_t stream);
void launch_qk_rms_norm_f32(float *q, float *k, const float *q_weight, const float *k_weight, int seq, int heads, int head_dim, float eps, cudaStream_t stream);
void launch_qk_rms_norm_bf16(uint16_t *q, uint16_t *k, const uint16_t *q_weight, const uint16_t *k_weight, int seq, int heads, int head_dim, float eps, cudaStream_t stream);
void launch_head_rms_norm_bf16(uint16_t *x, const uint16_t *weight, int seq, int heads, int head_dim, float eps, cudaStream_t stream);

void launch_adaln_norm_f32(const float *x, const float *shift, const float *scale, float *out, int seq, int hidden, float eps, cudaStream_t stream);
void launch_adaln_norm_bf16(const uint16_t *x, const uint16_t *shift, const uint16_t *scale, uint16_t *out, int seq, int hidden, float eps, cudaStream_t stream);

void launch_silu_f32(float *x, int n, cudaStream_t stream);
void launch_silu_bf16(uint16_t *x, int n, cudaStream_t stream);
void launch_silu_mul_f32(float *gate, const float *up, int n, cudaStream_t stream);
void launch_silu_mul_bf16(uint16_t *gate, const uint16_t *up, int n, cudaStream_t stream);
void launch_gated_add_f32(float *out, const float *gate, const float *proj, int seq, int hidden, cudaStream_t stream);
void launch_gated_add_bf16(uint16_t *out, const uint16_t *gate, const uint16_t *proj, int seq, int hidden, cudaStream_t stream);
void launch_add_f32(float *out, const float *a, const float *b, int n, cudaStream_t stream);
void launch_add_bf16(uint16_t *out, const uint16_t *a, const uint16_t *b, int n, cudaStream_t stream);
void launch_add_bias_f32(float *out, const float *bias, int rows, int cols, cudaStream_t stream);
void launch_softmax_f32(float *x, int rows, int cols, cudaStream_t stream);
void launch_softmax_bf16(uint16_t *x, int rows, int cols, cudaStream_t stream);
void launch_f32_to_bf16(const float *in, uint16_t *out, int n, cudaStream_t stream);
void launch_bf16_to_f32(const uint16_t *in, float *out, int n, cudaStream_t stream);

void launch_split_qkv_mlp_f32(const float *fused, float *q, float *k, float *v, float *gate, float *up, int seq, int hidden, int mlp_hidden, cudaStream_t stream);
void launch_split_qkv_mlp_bf16(const uint16_t *fused, uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *gate, uint16_t *up, int seq, int hidden, int mlp_hidden, cudaStream_t stream);
void launch_concat_attn_mlp_f32(const float *attn, const float *mlp, float *out, int seq, int hidden, int mlp_hidden, cudaStream_t stream);
void launch_concat_attn_mlp_bf16(const uint16_t *attn, const uint16_t *mlp, uint16_t *out, int seq, int hidden, int mlp_hidden, cudaStream_t stream);
void launch_concat_seq_bf16(uint16_t *out, const uint16_t *a, const uint16_t *b, int seq_a, int seq_b, int hidden, cudaStream_t stream);
void launch_slice_seq_bf16(uint16_t *out, const uint16_t *in, int seq_out, int hidden, int start, cudaStream_t stream);
void launch_transpose_to_heads_bf16(const uint16_t *in, uint16_t *out, int seq, int heads, int head_dim, cudaStream_t stream);
void launch_transpose_from_heads_bf16(const uint16_t *in, uint16_t *out, int seq, int heads, int head_dim, cudaStream_t stream);

void launch_apply_rope_2d_f32(float *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim, cudaStream_t stream);
void launch_apply_rope_2d_bf16(uint16_t *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim, cudaStream_t stream);
void launch_apply_rope_unified_f32(float *x, const float *txt_cos, const float *txt_sin, const float *img_cos, const float *img_sin, int seq, int img_offset, int heads, int head_dim, int axis_dim, cudaStream_t stream);
void launch_apply_rope_unified_bf16(uint16_t *x, const float *txt_cos, const float *txt_sin, const float *img_cos, const float *img_sin, int seq, int img_offset, int heads, int head_dim, int axis_dim, cudaStream_t stream);
void launch_apply_rope_single_f32(float *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, cudaStream_t stream);
void launch_rope_text_bf16(uint16_t *q, uint16_t *k, const float *cos_cache, const float *sin_cache, int seq, int num_q_heads, int num_kv_heads, int head_dim, cudaStream_t stream);

int launch_attention_fused_f32(const float *Q, const float *K, const float *V, float *out, int seq_q, int seq_k, int num_heads, int head_dim, float scale, cudaStream_t stream);
int launch_attention_fused_bf16(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *out, int seq_q, int seq_k, int num_heads, int head_dim, float scale, cudaStream_t stream);
int launch_causal_attention_fused_bf16(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *out, const int *attention_mask, int seq, int num_q_heads, int num_kv_heads, int head_dim, float scale, cudaStream_t stream);

void launch_group_norm_f32(const float *x, const float *gamma, const float *beta, float *out, int batch, int channels, int spatial, int channels_per_group, float eps, cudaStream_t stream);
void launch_swish_f32(const float *x, float *out, int n, cudaStream_t stream);
void launch_nchw_to_nhwc_f32(const float *in, float *out, int batch, int channels, int spatial, cudaStream_t stream);
void launch_nhwc_to_nchw_f32(const float *in, float *out, int batch, int channels, int spatial, cudaStream_t stream);
void launch_upsample_nearest_2x_f32(const float *x, float *out, int channels, int in_h, int in_w, cudaStream_t stream);
void launch_conv2d_f32(const float *in, const float *weight, const float *bias, float *out, int batch, int in_ch, int out_ch, int in_h, int in_w, int out_h, int out_w, int kH, int kW, int stride, int padding, cudaStream_t stream);
int launch_im2col_f32(const float *in, float *col, int in_ch, int in_h, int in_w, int out_h, int out_w, int kH, int kW, int stride, int padding, int start, int count, cudaStream_t stream);
int launch_im2col_upsample2x_f32(const float *in, float *col,
                                 int in_ch, int in_h, int in_w,
                                 int out_h, int out_w, int kH, int kW,
                                 int padding, int start, int count,
                                 cudaStream_t stream);
void launch_add_bias_nchw_f32(float *out, const float *bias, int batch, int channels, int spatial, cudaStream_t stream);
} /* extern "C" */

/* ========================================================================
 * Global CUDA State
 * ======================================================================== */

static int g_initialized = 0;
static cudaStream_t g_stream = NULL;
static cublasHandle_t g_cublas = NULL;
static cublasLtHandle_t g_cublaslt = NULL;
static cudaMemPool_t g_mem_pool = NULL;
static int g_in_batch = 0;

/* Tensor Representation */
struct iris_gpu_tensor {
    void *device_ptr;
    size_t num_elements;
    size_t bytes;
    int is_f16;
    int is_persistent;
    int async_alloc;
};

/* Weight Cache Entry */
#define MAX_WEIGHT_CACHE 2048
typedef struct {
    const void *host_ptr;
    void *device_ptr;
    size_t bytes;
    int async_alloc;
    int streaming_retained;
} weight_cache_entry_t;

static weight_cache_entry_t g_weight_cache[MAX_WEIGHT_CACHE];
static int g_weight_cache_count = 0;
static size_t g_streaming_cache_bytes = 0;
static size_t g_streaming_cache_budget = 0;

typedef struct {
    void *ptr;
    int async_alloc;
} transient_buffer_t;

/* cudaMalloc/cudaFree introduce device-wide synchronization.  Prefer the
 * stream-ordered allocator and retain a compatibility fallback for runtimes
 * where memory pools are unavailable. */
static cudaError_t device_alloc(void **ptr, size_t bytes, int *async_alloc) {
    *ptr = NULL;
    *async_alloc = 0;
    cudaError_t err = cudaMallocAsync(ptr, bytes, g_stream);
    if (err == cudaSuccess) {
        *async_alloc = 1;
        return cudaSuccess;
    }
    cudaGetLastError();
    return cudaMalloc(ptr, bytes);
}

static void device_free(void *ptr, int async_alloc) {
    if (!ptr) return;
    if (async_alloc) cudaFreeAsync(ptr, g_stream);
    else cudaFree(ptr);
}

static transient_buffer_t upload_transient(const void *host_ptr, size_t bytes) {
    transient_buffer_t result = {NULL, 0};
    if (!host_ptr || bytes == 0) return result;
    if (device_alloc(&result.ptr, bytes, &result.async_alloc) != cudaSuccess) {
        result.ptr = NULL;
        return result;
    }
    if (cudaMemcpyAsync(result.ptr, host_ptr, bytes, cudaMemcpyHostToDevice, g_stream) != cudaSuccess) {
        device_free(result.ptr, result.async_alloc);
        result.ptr = NULL;
    }
    return result;
}

static void release_transient(transient_buffer_t *buffer) {
    if (!buffer) return;
    device_free(buffer->ptr, buffer->async_alloc);
    buffer->ptr = NULL;
}

static void *get_or_create_cached_weight(const void *host_ptr, size_t bytes) {
    if (!host_ptr || bytes == 0) return NULL;

    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].host_ptr == host_ptr &&
            g_weight_cache[i].bytes == bytes) {
            return g_weight_cache[i].device_ptr;
        }
    }

    if (g_weight_cache_count >= MAX_WEIGHT_CACHE) {
        /* Returning an untracked allocation here leaked it permanently. */
        return NULL;
    }

    void *device_ptr = NULL;
    int async_alloc = 0;
    cudaError_t err = device_alloc(&device_ptr, bytes, &async_alloc);
    if (err != cudaSuccess || !device_ptr) {
        return NULL;
    }

    if (cudaMemcpyAsync(device_ptr, host_ptr, bytes, cudaMemcpyHostToDevice, g_stream) != cudaSuccess) {
        device_free(device_ptr, async_alloc);
        return NULL;
    }

    g_weight_cache[g_weight_cache_count].host_ptr = host_ptr;
    g_weight_cache[g_weight_cache_count].device_ptr = device_ptr;
    g_weight_cache[g_weight_cache_count].bytes = bytes;
    g_weight_cache[g_weight_cache_count].async_alloc = async_alloc;
    g_weight_cache[g_weight_cache_count].streaming_retained = 0;
    g_weight_cache_count++;

    return device_ptr;
}

void iris_cuda_invalidate_weight(const void *host_ptr) {
    if (!host_ptr) return;
    for (int i = 0; i < g_weight_cache_count; ) {
        if (g_weight_cache[i].host_ptr != host_ptr) {
            i++;
            continue;
        }
        if (g_weight_cache[i].streaming_retained) {
            if (g_weight_cache[i].bytes <= g_streaming_cache_bytes)
                g_streaming_cache_bytes -= g_weight_cache[i].bytes;
            else
                g_streaming_cache_bytes = 0;
        }
        device_free(g_weight_cache[i].device_ptr, g_weight_cache[i].async_alloc);
        g_weight_cache[i] = g_weight_cache[g_weight_cache_count - 1];
        g_weight_cache_count--;
    }
}

void iris_cuda_release_streaming_weight(const void *host_ptr) {
    if (!host_ptr) return;
    for (int i = 0; i < g_weight_cache_count; i++) {
        weight_cache_entry_t *entry = &g_weight_cache[i];
        if (entry->host_ptr != host_ptr) continue;
        if (entry->streaming_retained) return;
        if (entry->bytes <= g_streaming_cache_budget -
                            g_streaming_cache_bytes) {
            entry->streaming_retained = 1;
            g_streaming_cache_bytes += entry->bytes;
            return;
        }
        break;
    }
    iris_cuda_invalidate_weight(host_ptr);
}

void iris_cuda_clear_streaming_weights(void) {
    for (int i = 0; i < g_weight_cache_count; ) {
        if (!g_weight_cache[i].streaming_retained) {
            i++;
            continue;
        }
        device_free(g_weight_cache[i].device_ptr,
                    g_weight_cache[i].async_alloc);
        g_weight_cache[i] = g_weight_cache[g_weight_cache_count - 1];
        g_weight_cache_count--;
    }
    g_streaming_cache_bytes = 0;
}

/* ========================================================================
 * Initialization and Lifecycle
 * ======================================================================== */

int iris_cuda_init(void) {
    if (g_initialized) return 1;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return 0;
    }

    cudaSetDevice(0);

    /* Retain at most one fifth of VRAM, capped at 1.5 GiB, for stable
     * mmap-backed transformer weights.  On an 8 GiB card this leaves ample
     * room for 1024px activations and the bounded attention workspace. */
    cudaDeviceProp props;
    if (cudaGetDeviceProperties(&props, 0) == cudaSuccess) {
        g_streaming_cache_budget = props.totalGlobalMem / 5;
        const size_t cap = 1536ULL * 1024 * 1024;
        if (g_streaming_cache_budget > cap)
            g_streaming_cache_budget = cap;
    } else {
        g_streaming_cache_budget = 0;
        cudaGetLastError();
    }

    if (cudaStreamCreateWithFlags(&g_stream, cudaStreamNonBlocking) != cudaSuccess) {
        return 0;
    }

    /* Keep freed stream-ordered allocations available across batch syncs.
     * Full model-stage resets explicitly trim the pool. */
    if (cudaDeviceGetDefaultMemPool(&g_mem_pool, 0) == cudaSuccess) {
        uint64_t threshold = UINT64_MAX;
        cudaMemPoolSetAttribute(g_mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    } else {
        g_mem_pool = NULL;
        cudaGetLastError();
    }

    if (cublasCreate(&g_cublas) != CUBLAS_STATUS_SUCCESS) {
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
        return 0;
    }
    cublasSetStream(g_cublas, g_stream);
    cublasSetMathMode(g_cublas, CUBLAS_DEFAULT_MATH);

    if (cublasLtCreate(&g_cublaslt) != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(g_cublas);
        cudaStreamDestroy(g_stream);
        g_cublas = NULL;
        g_stream = NULL;
        return 0;
    }

    g_initialized = 1;
    return 1;
}

int iris_metal_init(void) { return iris_cuda_init(); }

int iris_cuda_available(void) {
    if (!g_initialized) {
        iris_cuda_init();
    }
    return g_initialized;
}

int iris_metal_available(void) { return iris_cuda_available(); }
int iris_bf16_pipeline_available(void) { return iris_cuda_available(); }
int iris_cuda_shaders_available(void) { return iris_cuda_available(); }
int iris_metal_shaders_available(void) { return iris_cuda_available(); }
int iris_cuda_init_shaders(void) { return iris_cuda_init(); }
int iris_metal_init_shaders(void) { return iris_cuda_init(); }

void iris_cuda_cleanup(void) {
    if (!g_initialized) return;

    iris_cuda_reset();

    if (g_cublaslt) {
        cublasLtDestroy(g_cublaslt);
        g_cublaslt = NULL;
    }
    if (g_cublas) {
        cublasDestroy(g_cublas);
        g_cublas = NULL;
    }
    if (g_stream) {
        cudaStreamDestroy(g_stream);
        g_stream = NULL;
    }
    g_mem_pool = NULL;
    g_initialized = 0;
}

void iris_metal_cleanup(void) { iris_cuda_cleanup(); }

void iris_cuda_reset(void) {
    if (g_stream) {
        cudaStreamSynchronize(g_stream);
    }
    for (int i = 0; i < g_weight_cache_count; i++) {
        if (g_weight_cache[i].device_ptr) {
            device_free(g_weight_cache[i].device_ptr,
                        g_weight_cache[i].async_alloc);
        }
    }
    g_weight_cache_count = 0;
    g_streaming_cache_bytes = 0;
    if (g_stream) cudaStreamSynchronize(g_stream);
    if (g_mem_pool) cudaMemPoolTrimTo(g_mem_pool, 0);
}

void iris_metal_reset(void) { iris_cuda_reset(); }
void iris_metal_rope_cache_begin(void) {}
void iris_metal_reset_transient(void) {
    if (g_stream) cudaStreamSynchronize(g_stream);
    if (g_mem_pool) cudaMemPoolTrimTo(g_mem_pool, 0);
}

void iris_metal_clear_weight_cache_only(void) { iris_cuda_reset(); }
void iris_metal_clear_bf16_cache_only(void) { iris_cuda_reset(); }
void iris_metal_clear_f16_cache_only(void) { iris_cuda_reset(); }
void iris_metal_clear_activation_pool_only(void) {}

void iris_cuda_sync(void) {
    if (g_stream) {
        cudaError_t err = cudaStreamSynchronize(g_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "CUDA stream error: %s\n", cudaGetErrorString(err));
        }
    }
}

void iris_metal_sync(void) { iris_cuda_sync(); }
void iris_cuda_wait_idle(void) { iris_cuda_sync(); }
void iris_metal_wait_idle(void) { iris_cuda_sync(); }

void iris_cuda_begin_batch(void) { g_in_batch = 1; }
void iris_metal_begin_batch(void) { iris_cuda_begin_batch(); }

void iris_cuda_end_batch(void) {
    g_in_batch = 0;
    iris_cuda_sync();
}
void iris_metal_end_batch(void) { iris_cuda_end_batch(); }

int iris_cuda_in_batch(void) { return g_in_batch; }
int iris_metal_in_batch(void) { return iris_cuda_in_batch(); }

size_t iris_cuda_memory_used(void) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) {
        return total_b - free_b;
    }
    return 0;
}

size_t iris_cuda_memory_free(void) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) == cudaSuccess) return free_b;
    return 0;
}

size_t iris_metal_memory_used(void) { return iris_cuda_memory_used(); }

void iris_cuda_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    if (!iris_cuda_available()) return;
    get_or_create_cached_weight(bf16_weights, num_elements * sizeof(uint16_t));
}

void iris_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements) {
    iris_cuda_warmup_bf16(bf16_weights, num_elements);
}

void iris_cuda_warmup_bf16_buffer(const uint16_t *bf16_weights, size_t num_elements) {
    iris_cuda_warmup_bf16(bf16_weights, num_elements);
}

void iris_metal_warmup_bf16_buffer(const uint16_t *bf16_weights, size_t num_elements) {
    iris_cuda_warmup_bf16_buffer(bf16_weights, num_elements);
}

/* ========================================================================
 * GPU Tensor Allocation API
 * ======================================================================== */

iris_gpu_tensor_t iris_gpu_tensor_alloc(size_t num_elements) {
    if (!iris_cuda_available() || num_elements == 0) return NULL;

    iris_gpu_tensor_t tensor = (iris_gpu_tensor_t)calloc(1, sizeof(struct iris_gpu_tensor));
    if (!tensor) return NULL;

    tensor->num_elements = num_elements;
    tensor->bytes = num_elements * sizeof(float);
    tensor->is_f16 = 0;
    tensor->is_persistent = 0;
    tensor->async_alloc = 0;

    cudaError_t err = device_alloc(&tensor->device_ptr, tensor->bytes,
                                   &tensor->async_alloc);
    if (err != cudaSuccess || !tensor->device_ptr) {
        size_t free_bytes = 0, total_bytes = 0;
        cudaMemGetInfo(&free_bytes, &total_bytes);
        fprintf(stderr, "CUDA tensor allocation failed: %.1f MiB requested, "
                        "%.1f/%.1f MiB free (%s)\n",
                tensor->bytes / (1024.0 * 1024.0),
                free_bytes / (1024.0 * 1024.0),
                total_bytes / (1024.0 * 1024.0), cudaGetErrorString(err));
        free(tensor);
        return NULL;
    }
    return tensor;
}

iris_gpu_tensor_t iris_gpu_tensor_alloc_f16(size_t num_elements) {
    if (!iris_cuda_available() || num_elements == 0) return NULL;

    iris_gpu_tensor_t tensor = (iris_gpu_tensor_t)calloc(1, sizeof(struct iris_gpu_tensor));
    if (!tensor) return NULL;

    tensor->num_elements = num_elements;
    tensor->bytes = num_elements * sizeof(uint16_t);
    tensor->is_f16 = 1;
    tensor->is_persistent = 0;
    tensor->async_alloc = 0;

    cudaError_t err = device_alloc(&tensor->device_ptr, tensor->bytes,
                                   &tensor->async_alloc);
    if (err != cudaSuccess || !tensor->device_ptr) {
        free(tensor);
        return NULL;
    }
    return tensor;
}

iris_gpu_tensor_t iris_gpu_tensor_alloc_persistent(size_t num_elements) {
    iris_gpu_tensor_t t = iris_gpu_tensor_alloc(num_elements);
    if (t) t->is_persistent = 1;
    return t;
}

void iris_gpu_tensor_set_persistent(iris_gpu_tensor_t tensor, int persistent) {
    if (tensor) tensor->is_persistent = persistent;
}

iris_gpu_tensor_t iris_gpu_tensor_create(const float *data, size_t num_elements) {
    iris_gpu_tensor_t tensor = iris_gpu_tensor_alloc(num_elements);
    if (!tensor) return NULL;
    if (data) {
        cudaMemcpyAsync(tensor->device_ptr, data, tensor->bytes, cudaMemcpyHostToDevice, g_stream);
    }
    return tensor;
}

void iris_gpu_tensor_read(iris_gpu_tensor_t tensor, float *out) {
    if (!tensor || !tensor->device_ptr || !out) return;
    cudaMemcpyAsync(out, tensor->device_ptr, tensor->bytes, cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
}

void iris_gpu_tensor_write(iris_gpu_tensor_t tensor, const float *data) {
    if (!tensor || !tensor->device_ptr || !data) return;
    cudaMemcpyAsync(tensor->device_ptr, data, tensor->bytes, cudaMemcpyHostToDevice, g_stream);
}

float *iris_gpu_tensor_data(iris_gpu_tensor_t tensor) {
    /* Unlike Metal shared buffers, discrete CUDA device memory is not CPU
     * addressable.  Callers must use tensor_read/tensor_write. */
    (void)tensor;
    return NULL;
}

void iris_gpu_tensor_free(iris_gpu_tensor_t tensor) {
    if (!tensor) return;
    if (tensor->device_ptr) {
        device_free(tensor->device_ptr, tensor->async_alloc);
        tensor->device_ptr = NULL;
    }
    free(tensor);
}

size_t iris_gpu_tensor_size(iris_gpu_tensor_t tensor) {
    return tensor ? tensor->num_elements : 0;
}

int iris_gpu_tensor_is_f16(iris_gpu_tensor_t tensor) {
    return tensor ? tensor->is_f16 : 0;
}

void iris_gpu_sync(void) { iris_cuda_sync(); }
void iris_gpu_batch_begin(void) { iris_cuda_begin_batch(); }
void iris_gpu_batch_end(void) { iris_cuda_end_batch(); }
void iris_gpu_chain_begin(void) { iris_cuda_begin_batch(); }
void iris_gpu_chain_end(void) { iris_cuda_end_batch(); }
int iris_gpu_in_chain(void) { return iris_cuda_in_batch(); }

/* ========================================================================
 * SGEMM & Linear Projections
 * ======================================================================== */

void iris_cuda_sgemm(int transpose_a, int transpose_b,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc) {
    if (!iris_cuda_available()) return;

    /* Device buffers if on host */
    void *d_A = (void*)A;
    void *d_B = (void*)B;
    void *d_C = (void*)C;

    cudaPointerAttributes attr_A, attr_B, attr_C;
    int a_on_dev = (cudaPointerGetAttributes(&attr_A, A) == cudaSuccess && attr_A.type == cudaMemoryTypeDevice);
    int b_on_dev = (cudaPointerGetAttributes(&attr_B, B) == cudaSuccess && attr_B.type == cudaMemoryTypeDevice);
    int c_on_dev = (cudaPointerGetAttributes(&attr_C, C) == cudaSuccess && attr_C.type == cudaMemoryTypeDevice);

    void *tmp_A = NULL, *tmp_B = NULL, *tmp_C = NULL;

    size_t sz_A = (size_t)(transpose_a ? K * lda : M * lda) * sizeof(float);
    size_t sz_B = (size_t)(transpose_b ? N * ldb : K * ldb) * sizeof(float);
    size_t sz_C = (size_t)M * ldc * sizeof(float);

    if (!a_on_dev) {
        cudaMalloc(&tmp_A, sz_A);
        cudaMemcpyAsync(tmp_A, A, sz_A, cudaMemcpyHostToDevice, g_stream);
        d_A = tmp_A;
    }
    if (!b_on_dev) {
        cudaMalloc(&tmp_B, sz_B);
        cudaMemcpyAsync(tmp_B, B, sz_B, cudaMemcpyHostToDevice, g_stream);
        d_B = tmp_B;
    }
    if (!c_on_dev) {
        cudaMalloc(&tmp_C, sz_C);
        if (beta != 0.0f) {
            cudaMemcpyAsync(tmp_C, C, sz_C, cudaMemcpyHostToDevice, g_stream);
        }
        d_C = tmp_C;
    }

    cublasOperation_t op_A = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    /* BLAS Row-Major C[M, N] = A[M, K] * B[K, N] maps to Col-Major C^T[N, M] = B^T * A^T */
    cublasSgemm(g_cublas,
                op_B, op_A,
                N, M, K,
                &alpha,
                (const float *)d_B, ldb,
                (const float *)d_A, lda,
                &beta,
                (float *)d_C, ldc);

    if (!c_on_dev) {
        cudaMemcpyAsync(C, tmp_C, sz_C, cudaMemcpyDeviceToHost, g_stream);
        cudaStreamSynchronize(g_stream);
    }

    if (tmp_A) cudaFree(tmp_A);
    if (tmp_B) cudaFree(tmp_B);
    if (tmp_C) cudaFree(tmp_C);
}

void iris_metal_sgemm(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    iris_cuda_sgemm(transpose_a, transpose_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void iris_cuda_sgemm_cached(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    iris_cuda_sgemm(transpose_a, transpose_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void iris_metal_sgemm_cached(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const float *B, int ldb, float beta, float *C, int ldc) {
    iris_cuda_sgemm_cached(transpose_a, transpose_b, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
}

void iris_cuda_sgemm_bf16(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const uint16_t *B_bf16, int ldb, float beta, float *C, int ldc) {
    if (!iris_cuda_available()) return;

    void *d_B = get_or_create_cached_weight(B_bf16, (size_t)N * ldb * sizeof(uint16_t));
    if (!d_B) return;

    /* A to device if on host */
    cudaPointerAttributes attr_A, attr_C;
    int a_on_dev = (cudaPointerGetAttributes(&attr_A, A) == cudaSuccess && attr_A.type == cudaMemoryTypeDevice);
    int c_on_dev = (cudaPointerGetAttributes(&attr_C, C) == cudaSuccess && attr_C.type == cudaMemoryTypeDevice);

    void *d_A = (void *)A;
    void *d_C = (void *)C;
    void *tmp_A = NULL, *tmp_C = NULL;

    size_t sz_A = (size_t)M * lda * sizeof(float);
    size_t sz_C = (size_t)M * ldc * sizeof(float);

    if (!a_on_dev) {
        cudaMalloc(&tmp_A, sz_A);
        cudaMemcpyAsync(tmp_A, A, sz_A, cudaMemcpyHostToDevice, g_stream);
        d_A = tmp_A;
    }
    if (!c_on_dev) {
        cudaMalloc(&tmp_C, sz_C);
        if (beta != 0.0f) {
            cudaMemcpyAsync(tmp_C, C, sz_C, cudaMemcpyHostToDevice, g_stream);
        }
        d_C = tmp_C;
    }

    cublasOperation_t op_A = transpose_a ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasOperation_t op_B = transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;

    /* Mixed precision SGEMM with BF16 weights */
    cublasGemmEx(g_cublas,
                 op_B, op_A,
                 N, M, K,
                 &alpha,
                 d_B, CUDA_R_16BF, ldb,
                 d_A, CUDA_R_32F, lda,
                 &beta,
                 d_C, CUDA_R_32F, ldc,
                 CUBLAS_COMPUTE_32F_FAST_TF32,
                 CUBLAS_GEMM_DEFAULT);

    if (!c_on_dev) {
        cudaMemcpyAsync(C, tmp_C, sz_C, cudaMemcpyDeviceToHost, g_stream);
        cudaStreamSynchronize(g_stream);
    }

    if (tmp_A) cudaFree(tmp_A);
    if (tmp_C) cudaFree(tmp_C);
}

void iris_metal_sgemm_bf16(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, const uint16_t *B_bf16, int ldb, float beta, float *C, int ldc) {
    iris_cuda_sgemm_bf16(transpose_a, transpose_b, M, N, K, alpha, A, lda, B_bf16, ldb, beta, C, ldc);
}

int iris_gpu_linear_bf16_native_into(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    if (!iris_cuda_available() || !out || !x || !W_bf16) return 0;

    void *d_W = get_or_create_cached_weight(W_bf16, (size_t)out_dim * in_dim * sizeof(uint16_t));
    if (!d_W) return 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasStatus_t status = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, seq_len, in_dim,
        &alpha,
        d_W, CUDA_R_16BF, in_dim,
        x->device_ptr, CUDA_R_16BF, in_dim,
        &beta,
        out->device_ptr, CUDA_R_16BF, out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    return (status == CUBLAS_STATUS_SUCCESS);
}

iris_gpu_tensor_t iris_gpu_linear_bf16_native(iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc_f16(seq_len * out_dim);
    if (!out) return NULL;
    if (!iris_gpu_linear_bf16_native_into(out, x, W_bf16, seq_len, in_dim, out_dim)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}

int iris_gpu_linear_bf16_into(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    if (!iris_cuda_available() || !out || !x || !W_bf16) return 0;

    void *d_W = get_or_create_cached_weight(W_bf16, (size_t)out_dim * in_dim * sizeof(uint16_t));
    if (!d_W) return 0;

    float alpha = 1.0f;
    float beta = 0.0f;

    cudaDataType_t x_type = x->is_f16 ? CUDA_R_16BF : CUDA_R_32F;
    cudaDataType_t out_type = out->is_f16 ? CUDA_R_16BF : CUDA_R_32F;

    cublasStatus_t status = cublasGemmEx(
        g_cublas,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_dim, seq_len, in_dim,
        &alpha,
        d_W, CUDA_R_16BF, in_dim,
        x->device_ptr, x_type, in_dim,
        &beta,
        out->device_ptr, out_type, out_dim,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT);

    return (status == CUBLAS_STATUS_SUCCESS);
}

iris_gpu_tensor_t iris_gpu_linear_bf16(iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc(seq_len * out_dim);
    if (!out) return NULL;
    if (!iris_gpu_linear_bf16_into(out, x, W_bf16, seq_len, in_dim, out_dim)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}

iris_gpu_tensor_t iris_gpu_linear_bf16_bf16out(iris_gpu_tensor_t x, const uint16_t *W_bf16, int seq_len, int in_dim, int out_dim) {
    return iris_gpu_linear_bf16_native(x, W_bf16, seq_len, in_dim, out_dim);
}

iris_gpu_tensor_t iris_gpu_linear(iris_gpu_tensor_t x, const float *W, const float *b, int seq_len, int in_dim, int out_dim) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc(seq_len * out_dim);
    if (!out) return NULL;

    void *d_W = get_or_create_cached_weight(W, (size_t)out_dim * in_dim * sizeof(float));
    if (!d_W) { iris_gpu_tensor_free(out); return NULL; }

    float alpha = 1.0f;
    float beta = 0.0f;

    cublasSgemm(g_cublas,
                CUBLAS_OP_T, CUBLAS_OP_N,
                out_dim, seq_len, in_dim,
                &alpha,
                (const float *)d_W, in_dim,
                (const float *)x->device_ptr, in_dim,
                &beta,
                (float *)out->device_ptr, out_dim);

    if (b) {
        void *d_b = get_or_create_cached_weight(b, (size_t)out_dim * sizeof(float));
        if (!d_b) {
            iris_gpu_tensor_free(out);
            return NULL;
        }
        launch_add_bias_f32((float *)out->device_ptr, (const float *)d_b,
                            seq_len, out_dim, g_stream);
    }

    return out;
}

/* ========================================================================
 * Kernel Dispatch Wrappers
 * ======================================================================== */

void iris_gpu_rms_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *weight, int seq, int hidden, float eps) {
    transient_buffer_t d_w = upload_transient(weight, (size_t)hidden * sizeof(float));
    if (!d_w.ptr) return;
    launch_rms_norm_f32((const float *)x->device_ptr, (const float *)d_w.ptr, (float *)out->device_ptr, seq, hidden, eps, g_stream);
    release_transient(&d_w);
}

void iris_gpu_rms_norm_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t x, iris_gpu_tensor_t weight, int seq, int hidden, float eps) {
    launch_rms_norm_bf16((const uint16_t *)x->device_ptr, (const uint16_t *)weight->device_ptr, (uint16_t *)out->device_ptr, seq, hidden, eps, g_stream);
}

void iris_metal_rms_norm(float *out, const float *x, const float *weight, int seq_len, int hidden, float eps) {
    iris_gpu_tensor_t tx = iris_gpu_tensor_create(x, seq_len * hidden);
    iris_gpu_tensor_t tout = iris_gpu_tensor_alloc(seq_len * hidden);
    iris_gpu_rms_norm_f32(tout, tx, weight, seq_len, hidden, eps);
    iris_gpu_tensor_read(tout, out);
    iris_gpu_tensor_free(tx);
    iris_gpu_tensor_free(tout);
}

void iris_gpu_qk_rms_norm(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *q_weight, const float *k_weight, int seq, int heads, int head_dim, float eps) {
    transient_buffer_t d_qw = upload_transient(q_weight, (size_t)head_dim * sizeof(float));
    transient_buffer_t d_kw = upload_transient(k_weight, (size_t)head_dim * sizeof(float));
    if (d_qw.ptr && d_kw.ptr) {
        launch_qk_rms_norm_f32((float *)q->device_ptr, (float *)k->device_ptr, (const float *)d_qw.ptr, (const float *)d_kw.ptr, seq, heads, head_dim, eps, g_stream);
    }
    release_transient(&d_qw);
    release_transient(&d_kw);
}

void iris_gpu_qk_rms_norm_bf16(iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t q_weight_bf16, iris_gpu_tensor_t k_weight_bf16, int seq, int heads, int head_dim, float eps) {
    launch_qk_rms_norm_bf16((uint16_t *)q->device_ptr, (uint16_t *)k->device_ptr, (const uint16_t *)q_weight_bf16->device_ptr, (const uint16_t *)k_weight_bf16->device_ptr, seq, heads, head_dim, eps, g_stream);
}

void iris_metal_qk_rms_norm(float *q, float *k, const float *q_weight, const float *k_weight, int seq, int heads, int head_dim, float eps) {
    iris_gpu_tensor_t tq = iris_gpu_tensor_create(q, seq * heads * head_dim);
    iris_gpu_tensor_t tk = iris_gpu_tensor_create(k, seq * heads * head_dim);
    iris_gpu_qk_rms_norm(tq, tk, q_weight, k_weight, seq, heads, head_dim, eps);
    iris_gpu_tensor_read(tq, q);
    iris_gpu_tensor_read(tk, k);
    iris_gpu_tensor_free(tq);
    iris_gpu_tensor_free(tk);
}

int iris_gpu_head_rms_norm_bf16(iris_gpu_tensor_t x, iris_gpu_tensor_t weight_bf16, int seq, int heads, int head_dim, float eps) {
    launch_head_rms_norm_bf16((uint16_t *)x->device_ptr, (const uint16_t *)weight_bf16->device_ptr, seq, heads, head_dim, eps, g_stream);
    return 1;
}

void iris_gpu_adaln_norm(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *shift, const float *scale, int seq, int hidden, float eps) {
    transient_buffer_t d_sh = upload_transient(shift, (size_t)hidden * sizeof(float));
    transient_buffer_t d_sc = upload_transient(scale, (size_t)hidden * sizeof(float));
    if (d_sh.ptr && d_sc.ptr) {
        launch_adaln_norm_f32((const float *)x->device_ptr, (const float *)d_sh.ptr, (const float *)d_sc.ptr, (float *)out->device_ptr, seq, hidden, eps, g_stream);
    }
    release_transient(&d_sh);
    release_transient(&d_sc);
}

void iris_gpu_adaln_norm_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t x, iris_gpu_tensor_t shift_bf16, iris_gpu_tensor_t scale_bf16, int seq, int hidden, float eps) {
    launch_adaln_norm_bf16((const uint16_t *)x->device_ptr, (const uint16_t *)shift_bf16->device_ptr, (const uint16_t *)scale_bf16->device_ptr, (uint16_t *)out->device_ptr, seq, hidden, eps, g_stream);
}

void iris_metal_adaln_norm(float *out, const float *x, const float *shift, const float *scale, int seq_len, int hidden, float eps) {
    iris_gpu_tensor_t tx = iris_gpu_tensor_create(x, seq_len * hidden);
    iris_gpu_tensor_t tout = iris_gpu_tensor_alloc(seq_len * hidden);
    iris_gpu_adaln_norm(tout, tx, shift, scale, seq_len, hidden, eps);
    iris_gpu_tensor_read(tout, out);
    iris_gpu_tensor_free(tx);
    iris_gpu_tensor_free(tout);
}

void iris_metal_silu(float *x, int n) {
    iris_gpu_tensor_t tx = iris_gpu_tensor_create(x, n);
    launch_silu_f32((float *)tx->device_ptr, n, g_stream);
    iris_gpu_tensor_read(tx, x);
    iris_gpu_tensor_free(tx);
}

void iris_metal_silu_mul(float *gate, const float *up, int n) {
    iris_gpu_tensor_t tg = iris_gpu_tensor_create(gate, n);
    iris_gpu_tensor_t tu = iris_gpu_tensor_create(up, n);
    launch_silu_mul_f32((float *)tg->device_ptr, (const float *)tu->device_ptr, n, g_stream);
    iris_gpu_tensor_read(tg, gate);
    iris_gpu_tensor_free(tg);
    iris_gpu_tensor_free(tu);
}

void iris_gpu_silu_mul(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n) {
    launch_silu_mul_f32((float *)gate->device_ptr, (const float *)up->device_ptr, n, g_stream);
}

void iris_gpu_silu_mul_bf16(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n) {
    launch_silu_mul_bf16((uint16_t *)gate->device_ptr, (const uint16_t *)up->device_ptr, n, g_stream);
}

void iris_gpu_gated_add(iris_gpu_tensor_t out, const float *gate, iris_gpu_tensor_t proj, int seq, int hidden) {
    transient_buffer_t d_g = upload_transient(gate, (size_t)hidden * sizeof(float));
    if (!d_g.ptr) return;
    launch_gated_add_f32((float *)out->device_ptr, (const float *)d_g.ptr, (const float *)proj->device_ptr, seq, hidden, g_stream);
    release_transient(&d_g);
}

void iris_gpu_gated_add_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t gate_bf16, iris_gpu_tensor_t proj, int seq, int hidden) {
    launch_gated_add_bf16((uint16_t *)out->device_ptr, (const uint16_t *)gate_bf16->device_ptr, (const uint16_t *)proj->device_ptr, seq, hidden, g_stream);
}

void iris_gpu_add_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int n) {
    launch_add_bf16((uint16_t *)out->device_ptr, (const uint16_t *)a->device_ptr, (const uint16_t *)b->device_ptr, n, g_stream);
}

void iris_gpu_copy_bf16(iris_gpu_tensor_t dst, iris_gpu_tensor_t src, size_t n) {
    cudaMemcpyAsync(dst->device_ptr, src->device_ptr, n * sizeof(uint16_t), cudaMemcpyDeviceToDevice, g_stream);
}

void iris_metal_softmax(float *x, int rows, int cols) {
    iris_gpu_tensor_t tx = iris_gpu_tensor_create(x, (size_t)rows * (size_t)cols);
    if (!tx) {
        iris_softmax_cpu(x, rows, cols);
        return;
    }
    launch_softmax_f32((float *)tx->device_ptr, rows, cols, g_stream);
    iris_gpu_tensor_read(tx, x);
    iris_gpu_tensor_free(tx);
}

void iris_gpu_split_qkv_mlp(iris_gpu_tensor_t fused, iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v, iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int seq, int hidden, int mlp_hidden) {
    launch_split_qkv_mlp_f32((const float *)fused->device_ptr, (float *)q->device_ptr, (float *)k->device_ptr, (float *)v->device_ptr, (float *)gate->device_ptr, (float *)up->device_ptr, seq, hidden, mlp_hidden, g_stream);
}

void iris_gpu_split_qkv_mlp_bf16(iris_gpu_tensor_t fused, iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v, iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int seq, int hidden, int mlp_hidden) {
    launch_split_qkv_mlp_bf16((const uint16_t *)fused->device_ptr, (uint16_t *)q->device_ptr, (uint16_t *)k->device_ptr, (uint16_t *)v->device_ptr, (uint16_t *)gate->device_ptr, (uint16_t *)up->device_ptr, seq, hidden, mlp_hidden, g_stream);
}

void iris_gpu_concat_attn_mlp(iris_gpu_tensor_t attn, iris_gpu_tensor_t mlp, iris_gpu_tensor_t out, int seq, int hidden, int mlp_hidden) {
    launch_concat_attn_mlp_f32((const float *)attn->device_ptr, (const float *)mlp->device_ptr, (float *)out->device_ptr, seq, hidden, mlp_hidden, g_stream);
}

void iris_gpu_concat_attn_mlp_bf16(iris_gpu_tensor_t attn, iris_gpu_tensor_t mlp, iris_gpu_tensor_t out, int seq, int hidden, int mlp_hidden) {
    launch_concat_attn_mlp_bf16((const uint16_t *)attn->device_ptr, (const uint16_t *)mlp->device_ptr, (uint16_t *)out->device_ptr, seq, hidden, mlp_hidden, g_stream);
}

void iris_gpu_concat_seq_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int seq_a, int seq_b, int hidden) {
    launch_concat_seq_bf16((uint16_t *)out->device_ptr, (const uint16_t *)a->device_ptr, (const uint16_t *)b->device_ptr, seq_a, seq_b, hidden, g_stream);
}

void iris_gpu_slice_seq_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t in, int seq_out, int hidden, int start) {
    launch_slice_seq_bf16((uint16_t *)out->device_ptr, (const uint16_t *)in->device_ptr, seq_out, hidden, start, g_stream);
}

void iris_gpu_transpose_to_heads_bf16(iris_gpu_tensor_t in, iris_gpu_tensor_t out, int seq, int heads, int head_dim) {
    launch_transpose_to_heads_bf16((const uint16_t *)in->device_ptr, (uint16_t *)out->device_ptr, seq, heads, head_dim, g_stream);
}

void iris_gpu_transpose_from_heads_bf16(iris_gpu_tensor_t in, iris_gpu_tensor_t out, int seq, int heads, int head_dim) {
    launch_transpose_from_heads_bf16((const uint16_t *)in->device_ptr, (uint16_t *)out->device_ptr, seq, heads, head_dim, g_stream);
}

void iris_gpu_rope_2d(iris_gpu_tensor_t x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim) {
    void *d_cos = get_or_create_cached_weight(cos_freq, (size_t)seq * head_dim * sizeof(float));
    void *d_sin = get_or_create_cached_weight(sin_freq, (size_t)seq * head_dim * sizeof(float));
    launch_apply_rope_2d_f32((float *)x->device_ptr, (const float *)d_cos, (const float *)d_sin, seq, heads, head_dim, axis_dim, g_stream);
}

void iris_gpu_rope_2d_bf16(iris_gpu_tensor_t x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim) {
    void *d_cos = get_or_create_cached_weight(cos_freq, (size_t)seq * head_dim * sizeof(float));
    void *d_sin = get_or_create_cached_weight(sin_freq, (size_t)seq * head_dim * sizeof(float));
    launch_apply_rope_2d_bf16((uint16_t *)x->device_ptr, (const float *)d_cos, (const float *)d_sin, seq, heads, head_dim, axis_dim, g_stream);
}

void iris_gpu_rope_single_f32(iris_gpu_tensor_t x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim) {
    void *d_cos = get_or_create_cached_weight(cos_freq, (size_t)seq * head_dim * sizeof(float));
    void *d_sin = get_or_create_cached_weight(sin_freq, (size_t)seq * head_dim * sizeof(float));
    launch_apply_rope_single_f32((float *)x->device_ptr, (const float *)d_cos, (const float *)d_sin, seq, heads, head_dim, g_stream);
}

void iris_gpu_rope_single_pair_f32(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim) {
    iris_gpu_rope_single_f32(q, cos_freq, sin_freq, seq, heads, head_dim);
    iris_gpu_rope_single_f32(k, cos_freq, sin_freq, seq, heads, head_dim);
}

void iris_gpu_rope_unified(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *txt_cos, const float *txt_sin, const float *img_cos, const float *img_sin, int seq, int img_offset, int heads, int head_dim, int axis_dim) {
    void *d_tcos = get_or_create_cached_weight(txt_cos, (size_t)img_offset * head_dim * sizeof(float));
    void *d_tsin = get_or_create_cached_weight(txt_sin, (size_t)img_offset * head_dim * sizeof(float));
    void *d_icos = get_or_create_cached_weight(img_cos, (size_t)(seq - img_offset) * head_dim * sizeof(float));
    void *d_isin = get_or_create_cached_weight(img_sin, (size_t)(seq - img_offset) * head_dim * sizeof(float));

    launch_apply_rope_unified_f32((float *)q->device_ptr, (const float *)d_tcos, (const float *)d_tsin, (const float *)d_icos, (const float *)d_isin, seq, img_offset, heads, head_dim, axis_dim, g_stream);
    launch_apply_rope_unified_f32((float *)k->device_ptr, (const float *)d_tcos, (const float *)d_tsin, (const float *)d_icos, (const float *)d_isin, seq, img_offset, heads, head_dim, axis_dim, g_stream);
}

void iris_gpu_rope_unified_bf16(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *txt_cos, const float *txt_sin, const float *img_cos, const float *img_sin, int seq, int img_offset, int heads, int head_dim, int axis_dim) {
    void *d_tcos = get_or_create_cached_weight(txt_cos, (size_t)img_offset * head_dim * sizeof(float));
    void *d_tsin = get_or_create_cached_weight(txt_sin, (size_t)img_offset * head_dim * sizeof(float));
    void *d_icos = get_or_create_cached_weight(img_cos, (size_t)(seq - img_offset) * head_dim * sizeof(float));
    void *d_isin = get_or_create_cached_weight(img_sin, (size_t)(seq - img_offset) * head_dim * sizeof(float));

    launch_apply_rope_unified_bf16((uint16_t *)q->device_ptr, (const float *)d_tcos, (const float *)d_tsin, (const float *)d_icos, (const float *)d_isin, seq, img_offset, heads, head_dim, axis_dim, g_stream);
    launch_apply_rope_unified_bf16((uint16_t *)k->device_ptr, (const float *)d_tcos, (const float *)d_tsin, (const float *)d_icos, (const float *)d_isin, seq, img_offset, heads, head_dim, axis_dim, g_stream);
}

void iris_gpu_rope_text_bf16(iris_gpu_tensor_t q, iris_gpu_tensor_t k, const float *cos_cache, const float *sin_cache, int seq, int num_q_heads, int num_kv_heads, int head_dim) {
    void *d_cos = get_or_create_cached_weight(cos_cache, (size_t)seq * (head_dim / 2) * sizeof(float));
    void *d_sin = get_or_create_cached_weight(sin_cache, (size_t)seq * (head_dim / 2) * sizeof(float));
    launch_rope_text_bf16((uint16_t *)q->device_ptr, (uint16_t *)k->device_ptr, (const float *)d_cos, (const float *)d_sin, seq, num_q_heads, num_kv_heads, head_dim, g_stream);
}

void iris_metal_rope_2d(float *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim) {
    iris_gpu_tensor_t tx = iris_gpu_tensor_create(x, seq * heads * head_dim);
    iris_gpu_rope_2d(tx, cos_freq, sin_freq, seq, heads, head_dim, axis_dim);
    iris_gpu_tensor_read(tx, x);
    iris_gpu_tensor_free(tx);
}

/* ========================================================================
 * Attention
 * ======================================================================== */

/* Q/K/V use token-major [sequence, head, dimension] storage.  cuBLAS can
 * consume that layout directly by treating each head as a column-major
 * matrix whose leading dimension is the full hidden size.  This avoids three
 * explicit transposes and turns the two O(sequence^2) attention products into
 * Tensor Core GEMMs. */
static int attention_cublas_f32(float *out, const float *Q, const float *K,
                                const float *V, int seq_q, int seq_k,
                                int num_heads, int head_dim, float scale) {
    if (!out || !Q || !K || !V || seq_q <= 0 || seq_k <= 0 ||
        num_heads <= 0 || head_dim <= 0) return 0;

    /* Keep the score matrix bounded.  A full 1024px Flux attention matrix is
     * already around 1.8 GiB in f32, and grows quadratically with resolution.
     * Attention rows are independent, so process a query tile at a time while
     * retaining the same cuBLAS GEMMs and numerically identical softmax. */
    const size_t workspace_budget = 384ULL * 1024 * 1024;
    const size_t bytes_per_query =
        (size_t)num_heads * (size_t)seq_k * sizeof(float);
    int tile_q = bytes_per_query ? (int)(workspace_budget / bytes_per_query) : 0;
    if (tile_q < 1) tile_q = 1;
    if (tile_q > seq_q) tile_q = seq_q;
    if (tile_q > 128 && tile_q < seq_q) tile_q = (tile_q / 128) * 128;

    size_t score_elements = 0;
    int scores_async = 0;
    float *scores = NULL;
    while (tile_q >= 1) {
        score_elements = (size_t)num_heads * (size_t)tile_q * (size_t)seq_k;
        if (device_alloc((void **)&scores, score_elements * sizeof(float),
                         &scores_async) == cudaSuccess) break;
        tile_q /= 2;
        if (tile_q > 128) tile_q = (tile_q / 128) * 128;
    }
    if (!scores) return 0;

    const int hidden = num_heads * head_dim;
    const long long head_stride = head_dim;
    const float zero = 0.0f;
    const float one = 1.0f;

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int query_start = 0; query_start < seq_q &&
         status == CUBLAS_STATUS_SUCCESS; query_start += tile_q) {
        int query_count = seq_q - query_start;
        if (query_count > tile_q) query_count = tile_q;
        const long long score_stride = (long long)query_count * seq_k;
        const float *q_tile = Q + (size_t)query_start * hidden;
        float *out_tile = out + (size_t)query_start * hidden;

        /* scores[head, query, key] = scale * Q * K^T.  Scores are emitted as
         * column-major [key, query], the desired row-major [query, key]
         * softmax layout. */
        status = cublasGemmStridedBatchedEx(
            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            seq_k, query_count, head_dim,
            &scale,
            K, CUDA_R_32F, hidden, head_stride,
            q_tile, CUDA_R_32F, hidden, head_stride,
            &zero,
            scores, CUDA_R_32F, seq_k, score_stride,
            num_heads,
            CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);

        if (status == CUBLAS_STATUS_SUCCESS) {
            launch_softmax_f32(scores, num_heads * query_count, seq_k, g_stream);

            /* out^T = V^T * softmax(scores)^T, written directly into the
             * corresponding token-major output rows. */
            status = cublasGemmStridedBatchedEx(
                g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim, query_count, seq_k,
                &one,
                V, CUDA_R_32F, hidden, head_stride,
                scores, CUDA_R_32F, seq_k, score_stride,
                &zero,
                out_tile, CUDA_R_32F, hidden, head_stride,
                num_heads,
                CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT);
        }
    }

    int ok = status == CUBLAS_STATUS_SUCCESS &&
             cudaPeekAtLastError() == cudaSuccess;
    device_free(scores, scores_async);
    return ok;
}

static int attention_cublas_bf16(uint16_t *out, const uint16_t *Q,
                                 const uint16_t *K, const uint16_t *V,
                                 int seq_q, int seq_k, int num_heads,
                                 int head_dim, float scale) {
    if (!out || !Q || !K || !V || seq_q <= 0 || seq_k <= 0 ||
        num_heads <= 0 || head_dim <= 0) return 0;

    /* f32 logits plus bf16 probabilities use six bytes per score.  Bound the
     * workspace and tile queries so high-resolution generations stay on the
     * Tensor Core path instead of falling back to the scalar attention
     * kernel when a multi-gigabyte allocation is not possible. */
    const size_t workspace_budget = 384ULL * 1024 * 1024;
    const size_t bytes_per_query =
        (size_t)num_heads * (size_t)seq_k *
        (sizeof(float) + sizeof(uint16_t));
    int tile_q = bytes_per_query ? (int)(workspace_budget / bytes_per_query) : 0;
    if (tile_q < 1) tile_q = 1;
    if (tile_q > seq_q) tile_q = seq_q;
    if (tile_q > 128 && tile_q < seq_q) tile_q = (tile_q / 128) * 128;

    size_t score_elements = 0;
    int scores_async = 0, probs_async = 0;
    float *scores = NULL;
    uint16_t *probs = NULL;
    while (tile_q >= 1) {
        score_elements = (size_t)num_heads * (size_t)tile_q * (size_t)seq_k;
        if (device_alloc((void **)&scores, score_elements * sizeof(float),
                         &scores_async) == cudaSuccess &&
            device_alloc((void **)&probs, score_elements * sizeof(uint16_t),
                         &probs_async) == cudaSuccess) break;
        if (scores) device_free(scores, scores_async);
        if (probs) device_free(probs, probs_async);
        scores = NULL;
        probs = NULL;
        tile_q /= 2;
        if (tile_q > 128) tile_q = (tile_q / 128) * 128;
    }
    if (!scores || !probs) return 0;

    const int hidden = num_heads * head_dim;
    const long long head_stride = head_dim;
    const float zero = 0.0f;
    const float one = 1.0f;

    cublasStatus_t status = CUBLAS_STATUS_SUCCESS;
    for (int query_start = 0; query_start < seq_q &&
         status == CUBLAS_STATUS_SUCCESS; query_start += tile_q) {
        int query_count = seq_q - query_start;
        if (query_count > tile_q) query_count = tile_q;
        size_t tile_elements =
            (size_t)num_heads * (size_t)query_count * (size_t)seq_k;
        const long long score_stride = (long long)query_count * seq_k;
        const uint16_t *q_tile = Q + (size_t)query_start * hidden;
        uint16_t *out_tile = out + (size_t)query_start * hidden;

        /* Accumulate and normalize logits in f32.  Only normalized
         * probabilities are narrowed to bf16 for the Tensor Core V product. */
        status = cublasGemmStridedBatchedEx(
            g_cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            seq_k, query_count, head_dim,
            &scale,
            K, CUDA_R_16BF, hidden, head_stride,
            q_tile, CUDA_R_16BF, hidden, head_stride,
            &zero,
            scores, CUDA_R_32F, seq_k, score_stride,
            num_heads,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);

        if (status == CUBLAS_STATUS_SUCCESS) {
            launch_softmax_f32(scores, num_heads * query_count, seq_k, g_stream);
            launch_f32_to_bf16(scores, probs, (int)tile_elements, g_stream);

            status = cublasGemmStridedBatchedEx(
                g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                head_dim, query_count, seq_k,
                &one,
                V, CUDA_R_16BF, hidden, head_stride,
                probs, CUDA_R_16BF, seq_k, score_stride,
                &zero,
                out_tile, CUDA_R_16BF, hidden, head_stride,
                num_heads,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        }
    }

    int ok = status == CUBLAS_STATUS_SUCCESS &&
             cudaPeekAtLastError() == cudaSuccess;
    device_free(probs, probs_async);
    device_free(scores, scores_async);
    return ok;
}

int iris_gpu_attention_fused(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    if (attention_cublas_f32((float *)out->device_ptr,
                             (const float *)Q->device_ptr,
                             (const float *)K->device_ptr,
                             (const float *)V->device_ptr,
                             seq_q, seq_k, num_heads, head_dim, scale)) {
        return 1;
    }
    return launch_attention_fused_f32((const float *)Q->device_ptr, (const float *)K->device_ptr, (const float *)V->device_ptr, (float *)out->device_ptr, seq_q, seq_k, num_heads, head_dim, scale, g_stream);
}

int iris_gpu_attention_fused_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    if (attention_cublas_bf16((uint16_t *)out->device_ptr,
                              (const uint16_t *)Q->device_ptr,
                              (const uint16_t *)K->device_ptr,
                              (const uint16_t *)V->device_ptr,
                              seq_q, seq_k, num_heads, head_dim, scale)) {
        return 1;
    }
    return launch_attention_fused_bf16((const uint16_t *)Q->device_ptr, (const uint16_t *)K->device_ptr, (const uint16_t *)V->device_ptr, (uint16_t *)out->device_ptr, seq_q, seq_k, num_heads, head_dim, scale, g_stream);
}

int iris_gpu_attention_bf16_native(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    return iris_gpu_attention_fused_bf16(out, Q, K, V, seq_q, seq_k, num_heads, head_dim, scale);
}

int iris_gpu_attention_mps_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    return iris_gpu_attention_fused_bf16(out, Q, K, V, seq_q, seq_k, num_heads, head_dim, scale);
}

int iris_gpu_attention_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    return iris_gpu_attention_fused_bf16(out, Q, K, V, seq_q, seq_k, num_heads, head_dim, scale);
}

int iris_gpu_causal_attention_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V, const int *attention_mask, int seq, int num_q_heads, int num_kv_heads, int head_dim, float scale) {
    transient_buffer_t d_mask = {NULL, 0};
    if (attention_mask) d_mask = upload_transient(attention_mask, (size_t)seq * sizeof(int));
    if (attention_mask && !d_mask.ptr) return 0;
    int ok = launch_causal_attention_fused_bf16((const uint16_t *)Q->device_ptr, (const uint16_t *)K->device_ptr, (const uint16_t *)V->device_ptr, (uint16_t *)out->device_ptr, (const int *)d_mask.ptr, seq, num_q_heads, num_kv_heads, head_dim, scale, g_stream);
    release_transient(&d_mask);
    return ok;
}

int iris_metal_causal_attention(float *out, const float *Q, const float *K, const float *V, const int *attention_mask, int seq, int num_q_heads, int num_kv_heads, int head_dim, float scale) {
    return 0; /* Fall back to CPU or tensor path */
}

int iris_metal_attention_fused(float *out, const float *Q, const float *K, const float *V, int seq_q, int seq_k, int num_heads, int head_dim, float scale) {
    iris_gpu_tensor_t tq = iris_gpu_tensor_create(Q, seq_q * num_heads * head_dim);
    iris_gpu_tensor_t tk = iris_gpu_tensor_create(K, seq_k * num_heads * head_dim);
    iris_gpu_tensor_t tv = iris_gpu_tensor_create(V, seq_k * num_heads * head_dim);
    iris_gpu_tensor_t tout = iris_gpu_tensor_alloc(seq_q * num_heads * head_dim);

    int ok = iris_gpu_attention_fused(tout, tq, tk, tv, seq_q, seq_k, num_heads, head_dim, scale);
    if (ok) {
        iris_gpu_tensor_read(tout, out);
    }
    iris_gpu_tensor_free(tq);
    iris_gpu_tensor_free(tk);
    iris_gpu_tensor_free(tv);
    iris_gpu_tensor_free(tout);
    return ok;
}

void iris_metal_attention(float *out, const float *Q, const float *K, const float *V, float *scores_scratch, int heads, int seq_q, int seq_k, int head_dim, float scale) {
    (void)scores_scratch;
    iris_metal_attention_fused(out, Q, K, V, seq_q, seq_k, heads, head_dim, scale);
}

void iris_metal_attention_bf16(float *out, const float *Q, const float *K, const float *V, float *scores_scratch, int heads, int seq_q, int seq_k, int head_dim, float scale) {
    (void)scores_scratch;
    iris_metal_attention_fused(out, Q, K, V, seq_q, seq_k, heads, head_dim, scale);
}

/* ========================================================================
 * VAE & Elementwise Helpers
 * ======================================================================== */

void iris_gpu_group_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, const float *gamma, const float *beta, int batch, int channels, int spatial, int num_groups, float eps) {
    void *d_g = get_or_create_cached_weight(gamma, (size_t)channels * sizeof(float));
    void *d_b = get_or_create_cached_weight(beta, (size_t)channels * sizeof(float));
    int channels_per_group = channels / num_groups;
    launch_group_norm_f32((const float *)x->device_ptr, (const float *)d_g, (const float *)d_b, (float *)out->device_ptr, batch, channels, spatial, channels_per_group, eps, g_stream);
}

void iris_gpu_swish_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, int n) {
    launch_swish_f32((const float *)x->device_ptr, (float *)out->device_ptr, n, g_stream);
}

void iris_gpu_add_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int n) {
    launch_add_f32((float *)out->device_ptr, (const float *)a->device_ptr, (const float *)b->device_ptr, n, g_stream);
}

iris_gpu_tensor_t iris_gpu_upsample_nearest_2x_f32(iris_gpu_tensor_t x, int channels, int H, int W) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc((size_t)channels * (H * 2) * (W * 2));
    if (!out) return NULL;
    launch_upsample_nearest_2x_f32((const float *)x->device_ptr, (float *)out->device_ptr, channels, H, W, g_stream);
    return out;
}

/* Tensor-core-friendly convolution using a bounded im2col tile.  The output
 * remains in NCHW layout; viewed by cuBLAS it is a column-major [spatial,C]
 * matrix with leading dimension equal to the full spatial size. */
static int conv2d_im2col_f32(const float *d_in, const float *d_weight,
                             const float *d_bias, float *d_out,
                             int batch, int in_ch, int out_ch,
                             int H, int W, int out_h, int out_w,
                             int kH, int kW, int stride, int padding) {
    const int spatial = out_h * out_w;
    const int kernel_elems = in_ch * kH * kW;
    const size_t tile_budget = 128ULL * 1024 * 1024;
    int tile = (int)(tile_budget / ((size_t)kernel_elems * sizeof(float)));
    if (tile < 1) tile = 1;
    if (tile > spatial) tile = spatial;

    const int direct_1x1 = (kH == 1 && kW == 1 && stride == 1 && padding == 0);
    void *d_col = NULL;
    int col_async = 0;
    if (!direct_1x1 && device_alloc(&d_col,
            (size_t)kernel_elems * tile * sizeof(float), &col_async) != cudaSuccess) {
        return 0;
    }

    float alpha = 1.0f;
    float beta = 0.0f;
    int ok = 1;
    for (int b = 0; b < batch && ok; b++) {
        const float *in_b = d_in + (size_t)b * in_ch * H * W;
        float *out_b = d_out + (size_t)b * out_ch * spatial;
        for (int start = 0; start < spatial; start += tile) {
            int count = spatial - start;
            if (count > tile) count = tile;
            const float *matrix_a;
            int lda;
            if (direct_1x1) {
                matrix_a = in_b + start;
                lda = spatial;
            } else {
                if (!launch_im2col_f32(in_b, (float *)d_col,
                        in_ch, H, W, out_h, out_w, kH, kW,
                        stride, padding, start, count, g_stream)) {
                    ok = 0;
                    break;
                }
                matrix_a = (const float *)d_col;
                lda = count;
            }

            cublasStatus_t status = cublasGemmEx(
                g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                count, out_ch, kernel_elems,
                &alpha,
                matrix_a, CUDA_R_32F, lda,
                d_weight, CUDA_R_32F, kernel_elems,
                &beta,
                out_b + start, CUDA_R_32F, spatial,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                ok = 0;
                break;
            }
        }
    }

    if (d_col) device_free(d_col, col_async);
    if (ok && d_bias)
        launch_add_bias_nchw_f32(d_out, d_bias, batch, out_ch, spatial, g_stream);
    return ok;
}

/* Convolve a logical nearest-neighbor 2x upsample without allocating the
 * intermediate upsampled tensor. */
static int conv2d_upsample2x_im2col_f32(
        const float *d_in, const float *d_weight, const float *d_bias,
        float *d_out, int batch, int channels, int H, int W) {
    const int out_h = H * 2;
    const int out_w = W * 2;
    const int spatial = out_h * out_w;
    const int kernel_elems = channels * 3 * 3;
    const size_t tile_budget = 128ULL * 1024 * 1024;
    int tile = (int)(tile_budget / ((size_t)kernel_elems * sizeof(float)));
    if (tile < 1) tile = 1;
    if (tile > spatial) tile = spatial;

    void *d_col = NULL;
    int col_async = 0;
    if (device_alloc(&d_col, (size_t)kernel_elems * tile * sizeof(float),
                     &col_async) != cudaSuccess) return 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    int ok = 1;
    for (int b = 0; b < batch && ok; b++) {
        const float *in_b = d_in + (size_t)b * channels * H * W;
        float *out_b = d_out + (size_t)b * channels * spatial;
        for (int start = 0; start < spatial; start += tile) {
            int count = spatial - start;
            if (count > tile) count = tile;
            if (!launch_im2col_upsample2x_f32(
                    in_b, (float *)d_col, channels, H, W, out_h, out_w,
                    3, 3, 1, start, count, g_stream)) {
                ok = 0;
                break;
            }
            cublasStatus_t status = cublasGemmEx(
                g_cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                count, channels, kernel_elems,
                &alpha,
                d_col, CUDA_R_32F, count,
                d_weight, CUDA_R_32F, kernel_elems,
                &beta,
                out_b + start, CUDA_R_32F, spatial,
                CUBLAS_COMPUTE_32F_FAST_TF32,
                CUBLAS_GEMM_DEFAULT);
            if (status != CUBLAS_STATUS_SUCCESS) {
                ok = 0;
                break;
            }
        }
    }

    device_free(d_col, col_async);
    if (ok && d_bias)
        launch_add_bias_nchw_f32(d_out, d_bias, batch, channels, spatial, g_stream);
    return ok;
}

int iris_cuda_conv2d(float *out, const float *in, const float *weight, const float *bias, int batch, int in_ch, int out_ch, int H, int W, int kH, int kW, int stride, int padding) {
    int out_h = (H + 2 * padding - kH) / stride + 1;
    int out_w = (W + 2 * padding - kW) / stride + 1;

    iris_gpu_tensor_t tx = iris_gpu_tensor_create(in, (size_t)batch * in_ch * H * W);
    iris_gpu_tensor_t tout = iris_gpu_tensor_alloc((size_t)batch * out_ch * out_h * out_w);
    void *d_w = get_or_create_cached_weight(weight, (size_t)out_ch * in_ch * kH * kW * sizeof(float));
    void *d_b = bias ? get_or_create_cached_weight(bias, (size_t)out_ch * sizeof(float)) : NULL;

    if (!tx || !tout || !d_w || (bias && !d_b)) {
        iris_gpu_tensor_free(tx);
        iris_gpu_tensor_free(tout);
        return 0;
    }

    if (!conv2d_im2col_f32((const float *)tx->device_ptr, (const float *)d_w,
                           (const float *)d_b, (float *)tout->device_ptr,
                           batch, in_ch, out_ch, H, W, out_h, out_w,
                           kH, kW, stride, padding)) {
        launch_conv2d_f32((const float *)tx->device_ptr, (const float *)d_w,
                          (const float *)d_b, (float *)tout->device_ptr,
                          batch, in_ch, out_ch, H, W, out_h, out_w,
                          kH, kW, stride, padding, g_stream);
    }

    iris_gpu_tensor_read(tout, out);
    iris_gpu_tensor_free(tx);
    iris_gpu_tensor_free(tout);
    return 1;
}

int iris_metal_conv2d(float *out, const float *in, const float *weight, const float *bias, int batch, int in_ch, int out_ch, int H, int W, int kH, int kW, int stride, int padding) {
    return iris_cuda_conv2d(out, in, weight, bias, batch, in_ch, out_ch, H, W, kH, kW, stride, padding);
}

iris_gpu_tensor_t iris_gpu_conv2d_f32(iris_gpu_tensor_t x, const float *weight, const float *bias, int batch, int in_ch, int out_ch, int H, int W, int kH, int kW, int stride, int padding) {
    int out_h = (H + 2 * padding - kH) / stride + 1;
    int out_w = (W + 2 * padding - kW) / stride + 1;

    iris_gpu_tensor_t out = iris_gpu_tensor_alloc((size_t)batch * out_ch * out_h * out_w);
    if (!out) return NULL;

    void *d_w = get_or_create_cached_weight(weight, (size_t)out_ch * in_ch * kH * kW * sizeof(float));
    void *d_b = bias ? get_or_create_cached_weight(bias, (size_t)out_ch * sizeof(float)) : NULL;

    if (!d_w || (bias && !d_b)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }

    if (!conv2d_im2col_f32((const float *)x->device_ptr, (const float *)d_w,
                           (const float *)d_b, (float *)out->device_ptr,
                           batch, in_ch, out_ch, H, W, out_h, out_w,
                           kH, kW, stride, padding)) {
        launch_conv2d_f32((const float *)x->device_ptr, (const float *)d_w,
                          (const float *)d_b, (float *)out->device_ptr,
                          batch, in_ch, out_ch, H, W, out_h, out_w,
                          kH, kW, stride, padding, g_stream);
    }
    return out;
}

iris_gpu_tensor_t iris_gpu_upsample_conv2d_f32(
        iris_gpu_tensor_t x, const float *weight, const float *bias,
        int batch, int channels, int H, int W) {
    const int out_h = H * 2;
    const int out_w = W * 2;
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc(
        (size_t)batch * channels * out_h * out_w);
    if (!out) return NULL;

    void *d_w = get_or_create_cached_weight(
        weight, (size_t)channels * channels * 3 * 3 * sizeof(float));
    void *d_b = bias ? get_or_create_cached_weight(
        bias, (size_t)channels * sizeof(float)) : NULL;
    if (!d_w || (bias && !d_b) ||
        !conv2d_upsample2x_im2col_f32(
            (const float *)x->device_ptr, (const float *)d_w,
            (const float *)d_b, (float *)out->device_ptr,
            batch, channels, H, W)) {
        iris_gpu_tensor_free(out);
        return NULL;
    }
    return out;
}

iris_gpu_tensor_t iris_gpu_vae_attention_f32(
    iris_gpu_tensor_t x,
    const float *norm_weight, const float *norm_bias,
    const float *q_weight, const float *q_bias,
    const float *k_weight, const float *k_bias,
    const float *v_weight, const float *v_bias,
    const float *out_weight, const float *out_bias,
    int batch, int channels, int H, int W, int num_groups, float eps) {
    if (!x || batch <= 0 || channels <= 0 || H <= 0 || W <= 0 ||
        !norm_weight || !norm_bias || !q_weight || !k_weight || !v_weight ||
        !out_weight) return NULL;

    const int spatial = H * W;
    const size_t elements = (size_t)batch * channels * spatial;
    const size_t batch_elements = (size_t)channels * spatial;
    const float scale = 1.0f / sqrtf((float)channels);
    iris_gpu_tensor_t norm = NULL;
    iris_gpu_tensor_t projected_nchw = NULL;
    iris_gpu_tensor_t q = NULL;
    iris_gpu_tensor_t k = NULL;
    iris_gpu_tensor_t v = NULL;
    iris_gpu_tensor_t attn_nhwc = NULL;
    iris_gpu_tensor_t attn_nchw = NULL;
    iris_gpu_tensor_t result = NULL;

    norm = iris_gpu_tensor_alloc(elements);
    if (!norm) goto cleanup;
    iris_gpu_group_norm_f32(norm, x, norm_weight, norm_bias,
                            batch, channels, spatial, num_groups, eps);

    q = iris_gpu_tensor_alloc(elements);
    if (!q) goto cleanup;
    projected_nchw = iris_gpu_conv2d_f32(norm, q_weight, q_bias,
                                         batch, channels, channels,
                                         H, W, 1, 1, 1, 0);
    if (!projected_nchw) goto cleanup;
    launch_nchw_to_nhwc_f32((const float *)projected_nchw->device_ptr,
                            (float *)q->device_ptr,
                            batch, channels, spatial, g_stream);
    iris_gpu_tensor_free(projected_nchw);
    projected_nchw = NULL;

    k = iris_gpu_tensor_alloc(elements);
    if (!k) goto cleanup;
    projected_nchw = iris_gpu_conv2d_f32(norm, k_weight, k_bias,
                                         batch, channels, channels,
                                         H, W, 1, 1, 1, 0);
    if (!projected_nchw) goto cleanup;
    launch_nchw_to_nhwc_f32((const float *)projected_nchw->device_ptr,
                            (float *)k->device_ptr,
                            batch, channels, spatial, g_stream);
    iris_gpu_tensor_free(projected_nchw);
    projected_nchw = NULL;

    v = iris_gpu_tensor_alloc(elements);
    if (!v) goto cleanup;
    projected_nchw = iris_gpu_conv2d_f32(norm, v_weight, v_bias,
                                         batch, channels, channels,
                                         H, W, 1, 1, 1, 0);
    if (!projected_nchw) goto cleanup;
    launch_nchw_to_nhwc_f32((const float *)projected_nchw->device_ptr,
                            (float *)v->device_ptr,
                            batch, channels, spatial, g_stream);
    iris_gpu_tensor_free(projected_nchw);
    projected_nchw = NULL;
    iris_gpu_tensor_free(norm);
    norm = NULL;

    attn_nhwc = iris_gpu_tensor_alloc(elements);
    if (!attn_nhwc) goto cleanup;
    for (int b = 0; b < batch; b++) {
        const float *q_b = (const float *)q->device_ptr + b * batch_elements;
        const float *k_b = (const float *)k->device_ptr + b * batch_elements;
        const float *v_b = (const float *)v->device_ptr + b * batch_elements;
        float *out_b = (float *)attn_nhwc->device_ptr + b * batch_elements;
        if (!attention_cublas_f32(out_b, q_b, k_b, v_b,
                                  spatial, spatial, 1, channels, scale) &&
            !launch_attention_fused_f32(q_b, k_b, v_b, out_b,
                                        spatial, spatial, 1, channels,
                                        scale, g_stream)) {
            goto cleanup;
        }
    }
    iris_gpu_tensor_free(q); q = NULL;
    iris_gpu_tensor_free(k); k = NULL;
    iris_gpu_tensor_free(v); v = NULL;

    attn_nchw = iris_gpu_tensor_alloc(elements);
    if (!attn_nchw) goto cleanup;
    launch_nhwc_to_nchw_f32((const float *)attn_nhwc->device_ptr,
                            (float *)attn_nchw->device_ptr,
                            batch, channels, spatial, g_stream);
    iris_gpu_tensor_free(attn_nhwc);
    attn_nhwc = NULL;

    result = iris_gpu_conv2d_f32(attn_nchw, out_weight, out_bias,
                                  batch, channels, channels,
                                  H, W, 1, 1, 1, 0);
    if (result)
        iris_gpu_add_f32(result, x, result, (int)elements);

cleanup:
    if (norm) iris_gpu_tensor_free(norm);
    if (projected_nchw) iris_gpu_tensor_free(projected_nchw);
    if (q) iris_gpu_tensor_free(q);
    if (k) iris_gpu_tensor_free(k);
    if (v) iris_gpu_tensor_free(v);
    if (attn_nhwc) iris_gpu_tensor_free(attn_nhwc);
    if (attn_nchw) iris_gpu_tensor_free(attn_nchw);
    return result;
}

void iris_gpu_copy_f32(iris_gpu_tensor_t dst, iris_gpu_tensor_t src, size_t n) {
    cudaMemcpyAsync(dst->device_ptr, src->device_ptr, n * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);
}

void iris_gpu_copy_region_f32(iris_gpu_tensor_t dst, size_t dst_offset, iris_gpu_tensor_t src, size_t src_offset, size_t n) {
    float *d_dst = (float *)dst->device_ptr + dst_offset;
    float *d_src = (float *)src->device_ptr + src_offset;
    cudaMemcpyAsync(d_dst, d_src, n * sizeof(float), cudaMemcpyDeviceToDevice, g_stream);
}

int iris_gpu_convert_f32_to_bf16_into(iris_gpu_tensor_t bf16_out, iris_gpu_tensor_t f32_in) {
    launch_f32_to_bf16((const float *)f32_in->device_ptr, (uint16_t *)bf16_out->device_ptr, f32_in->num_elements, g_stream);
    return 1;
}

int iris_gpu_convert_bf16_to_f32_into(iris_gpu_tensor_t f32_out, iris_gpu_tensor_t bf16_in) {
    launch_bf16_to_f32((const uint16_t *)bf16_in->device_ptr, (float *)f32_out->device_ptr, bf16_in->num_elements, g_stream);
    return 1;
}

iris_gpu_tensor_t iris_gpu_tensor_f32_to_bf16(iris_gpu_tensor_t f32_tensor) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc_f16(f32_tensor->num_elements);
    if (!out) return NULL;
    iris_gpu_convert_f32_to_bf16_into(out, f32_tensor);
    return out;
}

iris_gpu_tensor_t iris_gpu_tensor_bf16_to_f32(iris_gpu_tensor_t bf16_tensor) {
    iris_gpu_tensor_t out = iris_gpu_tensor_alloc(bf16_tensor->num_elements);
    if (!out) return NULL;
    iris_gpu_convert_bf16_to_f32_into(out, bf16_tensor);
    return out;
}

void iris_cuda_sgemm_batch(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, int stride_a, const float *B, int ldb, int stride_b, float beta, float *C, int ldc, int stride_c, int batch_count) {
    for (int i = 0; i < batch_count; i++) {
        iris_cuda_sgemm(transpose_a, transpose_b, M, N, K, alpha,
                        A + (size_t)i * stride_a, lda,
                        B + (size_t)i * stride_b, ldb, beta,
                        C + (size_t)i * stride_c, ldc);
    }
}

void iris_metal_sgemm_batch(int transpose_a, int transpose_b, int M, int N, int K, float alpha, const float *A, int lda, int stride_a, const float *B, int ldb, int stride_b, float beta, float *C, int ldc, int stride_c, int batch_count) {
    iris_cuda_sgemm_batch(transpose_a, transpose_b, M, N, K, alpha, A, lda, stride_a, B, ldb, stride_b, beta, C, ldc, stride_c, batch_count);
}
