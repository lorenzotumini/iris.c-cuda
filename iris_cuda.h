/*
 * Iris CUDA Acceleration
 *
 * GPU-accelerated matrix operations and compute kernels using NVIDIA CUDA and cuBLAS.
 * Provides acceleration on NVIDIA GPUs (e.g. Ampere, RTX 3070 Ti, etc.).
 */

#ifndef IRIS_CUDA_H
#define IRIS_CUDA_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize CUDA acceleration.
 * Returns 1 on success, 0 if CUDA is not available.
 * Safe to call multiple times.
 */
int iris_cuda_init(void);
int iris_metal_init(void);

/*
 * Check if CUDA acceleration is available and initialized.
 */
int iris_cuda_available(void);
int iris_metal_available(void);

/*
 * Cleanup CUDA resources.
 */
void iris_cuda_cleanup(void);
void iris_metal_cleanup(void);

/*
 * Reset all GPU state (caches, pools, pending streams).
 */
void iris_cuda_reset(void);
void iris_metal_reset(void);
/* Remove cached copies associated with a host allocation.  CUDA uses this to
 * stream mmap-backed layer weights without flushing unrelated cache entries. */
void iris_cuda_invalidate_weight(const void *host_ptr);
void iris_metal_rope_cache_begin(void);
void iris_metal_reset_transient(void);

/* Debug / Cache clearing */
void iris_metal_clear_weight_cache_only(void);
void iris_metal_clear_bf16_cache_only(void);
void iris_metal_clear_f16_cache_only(void);
void iris_metal_clear_activation_pool_only(void);

/*
 * SGEMM Operations
 */
void iris_cuda_sgemm(int transpose_a, int transpose_b,
                     int M, int N, int K,
                     float alpha,
                     const float *A, int lda,
                     const float *B, int ldb,
                     float beta,
                     float *C, int ldc);
void iris_metal_sgemm(int transpose_a, int transpose_b,
                      int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc);

void iris_cuda_sgemm_cached(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda,
                            const float *B, int ldb,
                            float beta,
                            float *C, int ldc);
void iris_metal_sgemm_cached(int transpose_a, int transpose_b,
                             int M, int N, int K,
                             float alpha,
                             const float *A, int lda,
                             const float *B, int ldb,
                             float beta,
                             float *C, int ldc);

void iris_cuda_sgemm_bf16(int transpose_a, int transpose_b,
                          int M, int N, int K,
                          float alpha,
                          const float *A, int lda,
                          const uint16_t *B_bf16, int ldb,
                          float beta,
                          float *C, int ldc);
void iris_metal_sgemm_bf16(int transpose_a, int transpose_b,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda,
                           const uint16_t *B_bf16, int ldb,
                           float beta,
                           float *C, int ldc);

int iris_cuda_conv2d(float *out, const float *in,
                     const float *weight, const float *bias,
                     int batch, int in_ch, int out_ch,
                     int H, int W, int kH, int kW,
                     int stride, int padding);
int iris_metal_conv2d(float *out, const float *in,
                      const float *weight, const float *bias,
                      int batch, int in_ch, int out_ch,
                      int H, int W, int kH, int kW,
                      int stride, int padding);

void iris_cuda_sgemm_batch(int transpose_a, int transpose_b,
                           int M, int N, int K,
                           float alpha,
                           const float *A, int lda, int stride_a,
                           const float *B, int ldb, int stride_b,
                           float beta,
                           float *C, int ldc, int stride_c,
                           int batch_count);
void iris_metal_sgemm_batch(int transpose_a, int transpose_b,
                            int M, int N, int K,
                            float alpha,
                            const float *A, int lda, int stride_a,
                            const float *B, int ldb, int stride_b,
                            float beta,
                            float *C, int ldc, int stride_c,
                            int batch_count);

void iris_cuda_sync(void);
void iris_metal_sync(void);
void iris_cuda_wait_idle(void);
void iris_metal_wait_idle(void);

void iris_cuda_begin_batch(void);
void iris_metal_begin_batch(void);
void iris_cuda_end_batch(void);
void iris_metal_end_batch(void);
int iris_cuda_in_batch(void);
int iris_metal_in_batch(void);

size_t iris_cuda_memory_used(void);
size_t iris_metal_memory_used(void);

/* ========================================================================
 * GPU Tensor API
 * ======================================================================== */

typedef struct iris_gpu_tensor *iris_gpu_tensor_t;

iris_gpu_tensor_t iris_gpu_tensor_create(const float *data, size_t num_elements);
iris_gpu_tensor_t iris_gpu_tensor_alloc(size_t num_elements);
iris_gpu_tensor_t iris_gpu_tensor_alloc_persistent(size_t num_elements);
void iris_gpu_tensor_set_persistent(iris_gpu_tensor_t tensor, int persistent);
void iris_gpu_tensor_read(iris_gpu_tensor_t tensor, float *out);
void iris_gpu_tensor_write(iris_gpu_tensor_t tensor, const float *data);
float *iris_gpu_tensor_data(iris_gpu_tensor_t tensor);
void iris_gpu_tensor_free(iris_gpu_tensor_t tensor);
size_t iris_gpu_tensor_size(iris_gpu_tensor_t tensor);
int iris_gpu_tensor_is_f16(iris_gpu_tensor_t tensor);

/* ========================================================================
 * GPU Operations on Tensors
 * ======================================================================== */

iris_gpu_tensor_t iris_gpu_linear(iris_gpu_tensor_t x,
                                   const float *W, const float *b,
                                   int seq_len, int in_dim, int out_dim);

iris_gpu_tensor_t iris_gpu_linear_bf16(iris_gpu_tensor_t x,
                                        const uint16_t *W_bf16,
                                        int seq_len, int in_dim, int out_dim);

int iris_gpu_linear_bf16_into(iris_gpu_tensor_t out,
                              iris_gpu_tensor_t x,
                              const uint16_t *W_bf16,
                              int seq_len, int in_dim, int out_dim);

iris_gpu_tensor_t iris_gpu_linear_bf16_bf16out(iris_gpu_tensor_t x,
                                               const uint16_t *W_bf16,
                                               int seq_len, int in_dim, int out_dim);

iris_gpu_tensor_t iris_gpu_tensor_alloc_f16(size_t num_elements);

int iris_gpu_attention_mps_bf16(iris_gpu_tensor_t out,
                                iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V,
                                int seq_q, int seq_k, int num_heads, int head_dim, float scale);

int iris_gpu_attention_bf16(iris_gpu_tensor_t out,
                            iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V,
                            int seq_q, int seq_k, int num_heads, int head_dim, float scale);

void iris_gpu_sync(void);
void iris_gpu_batch_begin(void);
void iris_gpu_batch_end(void);
void iris_gpu_chain_begin(void);
void iris_gpu_chain_end(void);
int iris_gpu_in_chain(void);

/* Normalization / Modulation */
void iris_gpu_adaln_norm(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                         const float *shift, const float *scale,
                         int seq, int hidden, float eps);

void iris_gpu_rms_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                            const float *weight, int seq, int hidden, float eps);

void iris_gpu_qk_rms_norm(iris_gpu_tensor_t q, iris_gpu_tensor_t k,
                          const float *q_weight, const float *k_weight,
                          int seq, int heads, int head_dim, float eps);

/* RoPE */
void iris_gpu_rope_2d(iris_gpu_tensor_t x, const float *cos_freq, const float *sin_freq,
                      int seq, int heads, int head_dim, int axis_dim);

void iris_gpu_rope_single_f32(iris_gpu_tensor_t x,
                               const float *cos_freq, const float *sin_freq,
                               int seq, int heads, int head_dim);

void iris_gpu_rope_single_pair_f32(iris_gpu_tensor_t q, iris_gpu_tensor_t k,
                                   const float *cos_freq, const float *sin_freq,
                                   int seq, int heads, int head_dim);

void iris_gpu_rope_unified(iris_gpu_tensor_t q, iris_gpu_tensor_t k,
                           const float *txt_cos, const float *txt_sin,
                           const float *img_cos, const float *img_sin,
                           int seq, int img_offset, int heads, int head_dim, int axis_dim);

/* Elementwise */
void iris_gpu_silu_mul(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n);

void iris_gpu_gated_add(iris_gpu_tensor_t out, const float *gate,
                        iris_gpu_tensor_t proj, int seq, int hidden);

void iris_gpu_split_qkv_mlp(iris_gpu_tensor_t fused,
                            iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v,
                            iris_gpu_tensor_t gate, iris_gpu_tensor_t up,
                            int seq, int hidden, int mlp_hidden);

void iris_gpu_concat_attn_mlp(iris_gpu_tensor_t attn, iris_gpu_tensor_t mlp,
                              iris_gpu_tensor_t out, int seq, int hidden, int mlp_hidden);

int iris_gpu_attention_fused(iris_gpu_tensor_t out,
                             iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V,
                             int seq_q, int seq_k, int num_heads, int head_dim, float scale);

int iris_gpu_attention_bf16_native(iris_gpu_tensor_t out,
                                    iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V,
                                    int seq_q, int seq_k, int num_heads, int head_dim, float scale);

int iris_gpu_attention_fused_bf16(iris_gpu_tensor_t out,
                                   iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V,
                                   int seq_q, int seq_k, int num_heads, int head_dim, float scale);

/* ========================================================================
 * BF16 GPU Tensor Operations
 * ======================================================================== */

void iris_gpu_adaln_norm_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                               iris_gpu_tensor_t shift_bf16, iris_gpu_tensor_t scale_bf16,
                               int seq, int hidden, float eps);

void iris_gpu_qk_rms_norm_bf16(iris_gpu_tensor_t q, iris_gpu_tensor_t k,
                                iris_gpu_tensor_t q_weight_bf16, iris_gpu_tensor_t k_weight_bf16,
                                int seq, int heads, int head_dim, float eps);

int iris_gpu_head_rms_norm_bf16(iris_gpu_tensor_t x, iris_gpu_tensor_t weight_bf16,
                                 int seq, int heads, int head_dim, float eps);

void iris_gpu_rms_norm_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                             iris_gpu_tensor_t weight, int seq, int hidden, float eps);

void iris_gpu_add_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int n);

void iris_gpu_copy_bf16(iris_gpu_tensor_t dst, iris_gpu_tensor_t src, size_t n);

void iris_gpu_silu_mul_bf16(iris_gpu_tensor_t gate, iris_gpu_tensor_t up, int n);

void iris_gpu_gated_add_bf16(iris_gpu_tensor_t out, iris_gpu_tensor_t gate_bf16,
                              iris_gpu_tensor_t proj, int seq, int hidden);

void iris_gpu_rope_unified_bf16(iris_gpu_tensor_t q, iris_gpu_tensor_t k,
                                 const float *txt_cos, const float *txt_sin,
                                 const float *img_cos, const float *img_sin,
                                 int seq, int img_offset, int heads, int head_dim, int axis_dim);

void iris_gpu_rope_2d_bf16(iris_gpu_tensor_t x,
                            const float *cos_freq, const float *sin_freq,
                            int seq, int heads, int head_dim, int axis_dim);

int iris_gpu_causal_attention_bf16(iris_gpu_tensor_t out,
                                    iris_gpu_tensor_t Q, iris_gpu_tensor_t K, iris_gpu_tensor_t V,
                                    const int *attention_mask,
                                    int seq, int num_q_heads, int num_kv_heads,
                                    int head_dim, float scale);

void iris_gpu_rope_text_bf16(iris_gpu_tensor_t q, iris_gpu_tensor_t k,
                              const float *cos_cache, const float *sin_cache,
                              int seq, int num_q_heads, int num_kv_heads, int head_dim);

void iris_gpu_concat_seq_bf16(iris_gpu_tensor_t out,
                               iris_gpu_tensor_t a, iris_gpu_tensor_t b,
                               int seq_a, int seq_b, int hidden);

void iris_gpu_slice_seq_bf16(iris_gpu_tensor_t out,
                              iris_gpu_tensor_t in,
                              int seq_out, int hidden, int start);

void iris_gpu_split_qkv_mlp_bf16(iris_gpu_tensor_t fused,
                                  iris_gpu_tensor_t q, iris_gpu_tensor_t k, iris_gpu_tensor_t v,
                                  iris_gpu_tensor_t gate, iris_gpu_tensor_t up,
                                  int seq, int hidden, int mlp_hidden);

void iris_gpu_concat_attn_mlp_bf16(iris_gpu_tensor_t attn, iris_gpu_tensor_t mlp,
                                    iris_gpu_tensor_t out, int seq, int hidden, int mlp_hidden);

/* ========================================================================
 * F32 VAE Tensor Operations
 * ======================================================================== */

void iris_gpu_group_norm_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x,
                              const float *gamma, const float *beta,
                              int batch, int channels, int spatial, int num_groups, float eps);

void iris_gpu_swish_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t x, int n);

void iris_gpu_add_f32(iris_gpu_tensor_t out, iris_gpu_tensor_t a, iris_gpu_tensor_t b, int n);

iris_gpu_tensor_t iris_gpu_upsample_nearest_2x_f32(iris_gpu_tensor_t x,
                                                     int channels, int H, int W);

iris_gpu_tensor_t iris_gpu_conv2d_f32(iris_gpu_tensor_t x,
                                       const float *weight, const float *bias,
                                       int batch, int in_ch, int out_ch,
                                       int H, int W, int kH, int kW,
                                       int stride, int padding);

void iris_gpu_copy_f32(iris_gpu_tensor_t dst, iris_gpu_tensor_t src, size_t n);

void iris_gpu_copy_region_f32(iris_gpu_tensor_t dst, size_t dst_offset,
                               iris_gpu_tensor_t src, size_t src_offset,
                               size_t n);

iris_gpu_tensor_t iris_gpu_tensor_f32_to_bf16(iris_gpu_tensor_t f32_tensor);
iris_gpu_tensor_t iris_gpu_tensor_bf16_to_f32(iris_gpu_tensor_t bf16_tensor);

int iris_gpu_convert_f32_to_bf16_into(iris_gpu_tensor_t bf16_out, iris_gpu_tensor_t f32_in);
int iris_gpu_convert_bf16_to_f32_into(iris_gpu_tensor_t f32_out, iris_gpu_tensor_t bf16_in);

iris_gpu_tensor_t iris_gpu_linear_bf16_native(iris_gpu_tensor_t x,
                                               const uint16_t *W_bf16,
                                               int seq_len, int in_dim, int out_dim);

int iris_gpu_linear_bf16_native_into(iris_gpu_tensor_t out,
                                     iris_gpu_tensor_t x,
                                     const uint16_t *W_bf16,
                                     int seq_len, int in_dim, int out_dim);

void iris_gpu_transpose_to_heads_bf16(iris_gpu_tensor_t in, iris_gpu_tensor_t out,
                                       int seq, int heads, int head_dim);

void iris_gpu_transpose_from_heads_bf16(iris_gpu_tensor_t in, iris_gpu_tensor_t out,
                                         int seq, int heads, int head_dim);

void iris_metal_attention(float *out,
                          const float *Q, const float *K, const float *V,
                          float *scores_scratch,
                          int heads, int seq_q, int seq_k, int head_dim,
                          float scale);

void iris_metal_attention_bf16(float *out,
                               const float *Q, const float *K, const float *V,
                               float *scores_scratch,
                               int heads, int seq_q, int seq_k, int head_dim,
                               float scale);

int iris_metal_causal_attention(float *out,
                                 const float *Q, const float *K, const float *V,
                                 const int *attention_mask,
                                 int seq, int num_q_heads, int num_kv_heads,
                                 int head_dim, float scale);

int iris_metal_attention_fused(float *out,
                               const float *Q, const float *K, const float *V,
                               int seq_q, int seq_k, int num_heads, int head_dim,
                               float scale);

int iris_cuda_init_shaders(void);
int iris_metal_init_shaders(void);

void iris_metal_rms_norm(float *out, const float *x, const float *weight,
                         int seq_len, int hidden, float eps);

void iris_metal_qk_rms_norm(float *q, float *k,
                            const float *q_weight, const float *k_weight,
                            int seq, int heads, int head_dim, float eps);

void iris_metal_adaln_norm(float *out, const float *x,
                           const float *shift, const float *scale,
                           int seq_len, int hidden, float eps);

void iris_metal_silu(float *x, int n);

void iris_metal_silu_mul(float *gate, const float *up, int n);

void iris_metal_softmax(float *x, int rows, int cols);

void iris_metal_rope_2d(float *x, const float *cos_freq, const float *sin_freq,
                        int seq, int heads, int head_dim, int axis_dim);

int iris_cuda_shaders_available(void);
int iris_metal_shaders_available(void);

void iris_cuda_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);
void iris_metal_warmup_bf16(const uint16_t *bf16_weights, size_t num_elements);

void iris_cuda_warmup_bf16_buffer(const uint16_t *bf16_weights, size_t num_elements);
void iris_metal_warmup_bf16_buffer(const uint16_t *bf16_weights, size_t num_elements);

int iris_bf16_pipeline_available(void);

#ifdef __cplusplus
}
#endif

#endif /* IRIS_CUDA_H */
