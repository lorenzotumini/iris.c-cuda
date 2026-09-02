/*
 * Iris CUDA Compute Kernels
 *
 * Implements GPU-accelerated kernels matching iris_shaders.metal:
 * - RMSNorm / QK RMSNorm / Head RMSNorm
 * - LayerNorm + AdaLN modulation
 * - RoPE (Flux 4-axis, Z-Image 3-axis, Qwen3 text RoPE)
 * - Fused Attention / Causal Attention with GQA
 * - SiLU / SwiGLU / Activations / Reductions
 * - VAE operations (GroupNorm, Swish, Upsampling, Conv2D)
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>

#define WARP_SIZE 32
#define MAX_HEAD_DIM 128

/* Type conversion helpers */
__device__ __forceinline__ float bf16_to_f32(const uint16_t val) {
    __nv_bfloat16 bf = *reinterpret_cast<const __nv_bfloat16*>(&val);
    return __bfloat162float(bf);
}

__device__ __forceinline__ uint16_t f32_to_bf16(float val) {
    __nv_bfloat16 bf = __float2bfloat16(val);
    return *reinterpret_cast<uint16_t*>(&bf);
}

/* ========================================================================
 * RMSNorm & QK RMSNorm
 * ======================================================================== */

__global__ void rms_norm_f32_kernel(
    const float *__restrict__ x,
    const float *__restrict__ weight,
    float *__restrict__ out,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float sdata[];

    const float *x_row = x + row * hidden;
    float *out_row = out + row * hidden;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(sdata[0] / (float)hidden + eps);

    for (int i = tid; i < hidden; i += threads) {
        out_row[i] = x_row[i] * rms_inv * weight[i];
    }
}

__global__ void rms_norm_bf16_kernel(
    const uint16_t *__restrict__ x,
    const uint16_t *__restrict__ weight,
    uint16_t *__restrict__ out,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float sdata[];

    const uint16_t *x_row = x + row * hidden;
    uint16_t *out_row = out + row * hidden;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        local_sum += val * val;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    float rms_inv = rsqrtf(sdata[0] / (float)hidden + eps);

    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        float w = bf16_to_f32(weight[i]);
        out_row[i] = f32_to_bf16(val * rms_inv * w);
    }
}

__global__ void qk_rms_norm_f32_kernel(
    float *__restrict__ q,
    float *__restrict__ k,
    const float *__restrict__ q_weight,
    const float *__restrict__ k_weight,
    int heads,
    int head_dim,
    float eps
) {
    int seq_idx = blockIdx.x;
    int head_idx = threadIdx.x;

    if (head_idx >= heads) return;

    int hidden = heads * head_dim;
    int offset = seq_idx * hidden + head_idx * head_dim;

    float sum_sq_q = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float v = q[offset + d];
        sum_sq_q += v * v;
    }
    float rms_inv_q = rsqrtf(sum_sq_q / (float)head_dim + eps);
    for (int d = 0; d < head_dim; d++) {
        q[offset + d] = q[offset + d] * rms_inv_q * q_weight[d];
    }

    float sum_sq_k = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float v = k[offset + d];
        sum_sq_k += v * v;
    }
    float rms_inv_k = rsqrtf(sum_sq_k / (float)head_dim + eps);
    for (int d = 0; d < head_dim; d++) {
        k[offset + d] = k[offset + d] * rms_inv_k * k_weight[d];
    }
}

__global__ void qk_rms_norm_bf16_kernel(
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    const uint16_t *__restrict__ q_weight,
    const uint16_t *__restrict__ k_weight,
    int heads,
    int head_dim,
    float eps
) {
    int seq_idx = blockIdx.x;
    int head_idx = threadIdx.x;

    if (head_idx >= heads) return;

    int hidden = heads * head_dim;
    int offset = seq_idx * hidden + head_idx * head_dim;

    float sum_sq_q = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float v = bf16_to_f32(q[offset + d]);
        sum_sq_q += v * v;
    }
    float rms_inv_q = rsqrtf(sum_sq_q / (float)head_dim + eps);
    for (int d = 0; d < head_dim; d++) {
        float v = bf16_to_f32(q[offset + d]);
        float w = bf16_to_f32(q_weight[d]);
        q[offset + d] = f32_to_bf16(v * rms_inv_q * w);
    }

    float sum_sq_k = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float v = bf16_to_f32(k[offset + d]);
        sum_sq_k += v * v;
    }
    float rms_inv_k = rsqrtf(sum_sq_k / (float)head_dim + eps);
    for (int d = 0; d < head_dim; d++) {
        float v = bf16_to_f32(k[offset + d]);
        float w = bf16_to_f32(k_weight[d]);
        k[offset + d] = f32_to_bf16(v * rms_inv_k * w);
    }
}

__global__ void head_rms_norm_bf16_kernel(
    uint16_t *__restrict__ x,
    const uint16_t *__restrict__ weight,
    int heads,
    int head_dim,
    float eps
) {
    int seq_idx = blockIdx.x;
    int head_idx = threadIdx.x;

    if (head_idx >= heads) return;

    int hidden = heads * head_dim;
    int offset = seq_idx * hidden + head_idx * head_dim;

    float sum_sq = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        float v = bf16_to_f32(x[offset + d]);
        sum_sq += v * v;
    }
    float rms_inv = rsqrtf(sum_sq / (float)head_dim + eps);
    for (int d = 0; d < head_dim; d++) {
        float v = bf16_to_f32(x[offset + d]);
        float w = bf16_to_f32(weight[d]);
        x[offset + d] = f32_to_bf16(v * rms_inv * w);
    }
}

/* ========================================================================
 * AdaLN Normalization (LayerNorm + Modulation)
 * ======================================================================== */

__global__ void adaln_norm_f32_kernel(
    const float *__restrict__ x,
    const float *__restrict__ shift,
    const float *__restrict__ scale,
    float *__restrict__ out,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float s_mem[];
    float *s_sum = s_mem;
    float *s_sum_sq = s_mem + threads;

    const float *x_row = x + row * hidden;
    float *out_row = out + row * hidden;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = x_row[i];
        local_sum += val;
        local_sum_sq += val * val;
    }
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)hidden;
    float var = s_sum_sq[0] / (float)hidden - mean * mean;
    float std_inv = rsqrtf(var + eps);

    for (int i = tid; i < hidden; i += threads) {
        float norm = (x_row[i] - mean) * std_inv;
        out_row[i] = (1.0f + scale[i]) * norm + shift[i];
    }
}

__global__ void adaln_norm_bf16_kernel(
    const uint16_t *__restrict__ x,
    const uint16_t *__restrict__ shift,
    const uint16_t *__restrict__ scale,
    uint16_t *__restrict__ out,
    int hidden,
    float eps
) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float s_mem[];
    float *s_sum = s_mem;
    float *s_sum_sq = s_mem + threads;

    const uint16_t *x_row = x + row * hidden;
    uint16_t *out_row = out + row * hidden;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        local_sum += val;
        local_sum_sq += val * val;
    }
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)hidden;
    float var = s_sum_sq[0] / (float)hidden - mean * mean;
    float std_inv = rsqrtf(var + eps);

    for (int i = tid; i < hidden; i += threads) {
        float val = bf16_to_f32(x_row[i]);
        float sh = bf16_to_f32(shift[i]);
        float sc = bf16_to_f32(scale[i]);
        float norm = (val - mean) * std_inv;
        out_row[i] = f32_to_bf16((1.0f + sc) * norm + sh);
    }
}

/* ========================================================================
 * Elementwise & Activations
 * ======================================================================== */

__global__ void silu_f32_kernel(float *x, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float val = x[idx];
        x[idx] = val / (1.0f + expf(-val));
    }
}

__global__ void silu_bf16_kernel(uint16_t *x, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float val = bf16_to_f32(x[idx]);
        x[idx] = f32_to_bf16(val / (1.0f + expf(-val)));
    }
}

__global__ void silu_mul_f32_kernel(float *gate, const float *up, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float g = gate[idx];
        float silu_g = g / (1.0f + expf(-g));
        gate[idx] = silu_g * up[idx];
    }
}

__global__ void silu_mul_bf16_kernel(uint16_t *gate, const uint16_t *up, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float g = bf16_to_f32(gate[idx]);
        float u = bf16_to_f32(up[idx]);
        float silu_g = g / (1.0f + expf(-g));
        gate[idx] = f32_to_bf16(silu_g * u);
    }
}

__global__ void gated_add_f32_kernel(float *out, const float *gate, const float *proj, int seq, int hidden) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = seq * hidden;
    if (idx < total) {
        int h = idx % hidden;
        out[idx] += gate[h] * proj[idx];
    }
}

__global__ void gated_add_bf16_kernel(uint16_t *out, const uint16_t *gate, const uint16_t *proj, int seq, int hidden) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = seq * hidden;
    if (idx < total) {
        int h = idx % hidden;
        float o = bf16_to_f32(out[idx]);
        float g = bf16_to_f32(gate[h]);
        float p = bf16_to_f32(proj[idx]);
        out[idx] = f32_to_bf16(o + g * p);
    }
}

__global__ void add_f32_kernel(float *out, const float *a, const float *b, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

__global__ void add_bf16_kernel(uint16_t *out, const uint16_t *a, const uint16_t *b, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float av = bf16_to_f32(a[idx]);
        float bv = bf16_to_f32(b[idx]);
        out[idx] = f32_to_bf16(av + bv);
    }
}

__global__ void add_bias_f32_kernel(float *out, const float *bias, int rows, int cols) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = rows * cols;
    if (idx < total) out[idx] += bias[idx % cols];
}

__global__ void softmax_f32_kernel(float *x, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float sdata[];

    float *row_ptr = x + row * cols;

    float local_max = -1e30f;
    for (int i = tid; i < cols; i += threads) {
        local_max = fmaxf(local_max, row_ptr[i]);
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        float e = expf(row_ptr[i] - max_val);
        row_ptr[i] = e;
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int i = tid; i < cols; i += threads) {
        row_ptr[i] *= inv_sum;
    }
}

__global__ void softmax_bf16_kernel(uint16_t *x, int rows, int cols) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float sdata[];

    uint16_t *row_ptr = x + row * cols;

    float local_max = -1e30f;
    for (int i = tid; i < cols; i += threads) {
        local_max = fmaxf(local_max, bf16_to_f32(row_ptr[i]));
    }
    sdata[tid] = local_max;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < cols; i += threads) {
        float e = expf(bf16_to_f32(row_ptr[i]) - max_val);
        row_ptr[i] = f32_to_bf16(e);
        local_sum += e;
    }
    sdata[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / sdata[0];

    for (int i = tid; i < cols; i += threads) {
        float e = bf16_to_f32(row_ptr[i]);
        row_ptr[i] = f32_to_bf16(e * inv_sum);
    }
}

__global__ void f32_to_bf16_kernel(const float *in, uint16_t *out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = f32_to_bf16(in[idx]);
    }
}

__global__ void bf16_to_f32_kernel(const uint16_t *in, float *out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = bf16_to_f32(in[idx]);
    }
}

/* ========================================================================
 * Tensor Slicing & Reshaping
 * ======================================================================== */

__global__ void split_qkv_mlp_f32_kernel(
    const float *__restrict__ fused,
    float *__restrict__ q,
    float *__restrict__ k,
    float *__restrict__ v,
    float *__restrict__ gate,
    float *__restrict__ up,
    int seq, int hidden, int mlp_hidden
) {
    int seq_idx = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;

    int fused_stride = hidden * 3 + mlp_hidden * 2;
    const float *fused_row = fused + seq_idx * fused_stride;

    if (d < hidden) {
        q[seq_idx * hidden + d] = fused_row[d];
        k[seq_idx * hidden + d] = fused_row[hidden + d];
        v[seq_idx * hidden + d] = fused_row[hidden * 2 + d];
    } else if (d < hidden + mlp_hidden) {
        int mlp_d = d - hidden;
        gate[seq_idx * mlp_hidden + mlp_d] = fused_row[hidden * 3 + mlp_d];
        up[seq_idx * mlp_hidden + mlp_d] = fused_row[hidden * 3 + mlp_hidden + mlp_d];
    }
}

__global__ void split_qkv_mlp_bf16_kernel(
    const uint16_t *__restrict__ fused,
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    uint16_t *__restrict__ v,
    uint16_t *__restrict__ gate,
    uint16_t *__restrict__ up,
    int seq, int hidden, int mlp_hidden
) {
    int seq_idx = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;

    int fused_stride = hidden * 3 + mlp_hidden * 2;
    const uint16_t *fused_row = fused + seq_idx * fused_stride;

    if (d < hidden) {
        q[seq_idx * hidden + d] = fused_row[d];
        k[seq_idx * hidden + d] = fused_row[hidden + d];
        v[seq_idx * hidden + d] = fused_row[hidden * 2 + d];
    } else if (d < hidden + mlp_hidden) {
        int mlp_d = d - hidden;
        gate[seq_idx * mlp_hidden + mlp_d] = fused_row[hidden * 3 + mlp_d];
        up[seq_idx * mlp_hidden + mlp_d] = fused_row[hidden * 3 + mlp_hidden + mlp_d];
    }
}

__global__ void concat_attn_mlp_f32_kernel(
    const float *__restrict__ attn,
    const float *__restrict__ mlp,
    float *__restrict__ out,
    int seq, int hidden, int mlp_hidden
) {
    int seq_idx = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    int total_dim = hidden + mlp_hidden;

    if (d < hidden) {
        out[seq_idx * total_dim + d] = attn[seq_idx * hidden + d];
    } else if (d < total_dim) {
        out[seq_idx * total_dim + d] = mlp[seq_idx * mlp_hidden + (d - hidden)];
    }
}

__global__ void concat_attn_mlp_bf16_kernel(
    const uint16_t *__restrict__ attn,
    const uint16_t *__restrict__ mlp,
    uint16_t *__restrict__ out,
    int seq, int hidden, int mlp_hidden
) {
    int seq_idx = blockIdx.x;
    int d = threadIdx.x + blockIdx.y * blockDim.x;
    int total_dim = hidden + mlp_hidden;

    if (d < hidden) {
        out[seq_idx * total_dim + d] = attn[seq_idx * hidden + d];
    } else if (d < total_dim) {
        out[seq_idx * total_dim + d] = mlp[seq_idx * mlp_hidden + (d - hidden)];
    }
}

__global__ void concat_seq_bf16_kernel(
    uint16_t *__restrict__ out,
    const uint16_t *__restrict__ a,
    const uint16_t *__restrict__ b,
    int seq_a, int seq_b, int hidden
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_a = seq_a * hidden;
    int total = (seq_a + seq_b) * hidden;
    if (idx < total) {
        if (idx < total_a) {
            out[idx] = a[idx];
        } else {
            out[idx] = b[idx - total_a];
        }
    }
}

__global__ void slice_seq_bf16_kernel(
    uint16_t *__restrict__ out,
    const uint16_t *__restrict__ in,
    int seq_out, int hidden, int start
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = seq_out * hidden;
    if (idx < total) {
        out[idx] = in[(start * hidden) + idx];
    }
}

__global__ void transpose_to_heads_bf16_kernel(
    const uint16_t *__restrict__ in,
    uint16_t *__restrict__ out,
    int seq, int heads, int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int d = threadIdx.x;

    if (seq_idx < seq && head_idx < heads && d < head_dim) {
        int in_idx = seq_idx * (heads * head_dim) + head_idx * head_dim + d;
        int out_idx = head_idx * (seq * head_dim) + seq_idx * head_dim + d;
        out[out_idx] = in[in_idx];
    }
}

__global__ void transpose_from_heads_bf16_kernel(
    const uint16_t *__restrict__ in,
    uint16_t *__restrict__ out,
    int seq, int heads, int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int d = threadIdx.x;

    if (seq_idx < seq && head_idx < heads && d < head_dim) {
        int in_idx = head_idx * (seq * head_dim) + seq_idx * head_dim + d;
        int out_idx = seq_idx * (heads * head_dim) + head_idx * head_dim + d;
        out[out_idx] = in[in_idx];
    }
}

/* ========================================================================
 * RoPE (Rotary Position Embeddings)
 * ======================================================================== */

__global__ void apply_rope_2d_f32_kernel(
    float *__restrict__ x,
    const float *__restrict__ cos_freq,
    const float *__restrict__ sin_freq,
    int seq, int heads, int head_dim, int axis_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq || head_idx >= heads) return;

    int hidden = heads * head_dim;
    float *vec = x + seq_idx * hidden + head_idx * head_dim;
    const float *cos_row = cos_freq + seq_idx * head_dim;
    const float *sin_row = sin_freq + seq_idx * head_dim;

    int half_axis = axis_dim / 2;
    for (int axis = 0; axis < 4; axis++) {
        int axis_offset = axis * axis_dim;
        for (int d = 0; d < half_axis; d++) {
            int i0 = axis_offset + d;
            int i1 = axis_offset + half_axis + d;

            float c = cos_row[i0];
            float s = sin_row[i0];

            float x0 = vec[i0];
            float x1 = vec[i1];

            vec[i0] = x0 * c - x1 * s;
            vec[i1] = x0 * s + x1 * c;
        }
    }
}

__global__ void apply_rope_2d_bf16_kernel(
    uint16_t *__restrict__ x,
    const float *__restrict__ cos_freq,
    const float *__restrict__ sin_freq,
    int seq, int heads, int head_dim, int axis_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq || head_idx >= heads) return;

    int hidden = heads * head_dim;
    uint16_t *vec = x + seq_idx * hidden + head_idx * head_dim;
    const float *cos_row = cos_freq + seq_idx * head_dim;
    const float *sin_row = sin_freq + seq_idx * head_dim;

    (void)axis_dim;
    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = bf16_to_f32(vec[d]);
        float x1 = bf16_to_f32(vec[d + 1]);

        vec[d] = f32_to_bf16(x0 * c - x1 * s);
        vec[d + 1] = f32_to_bf16(x1 * c + x0 * s);
    }
}

__global__ void apply_rope_unified_f32_kernel(
    float *__restrict__ x,
    const float *__restrict__ txt_cos,
    const float *__restrict__ txt_sin,
    const float *__restrict__ img_cos,
    const float *__restrict__ img_sin,
    int seq, int img_offset, int heads, int head_dim, int axis_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq || head_idx >= heads) return;

    int hidden = heads * head_dim;
    float *vec = x + seq_idx * hidden + head_idx * head_dim;

    const float *cos_row;
    const float *sin_row;

    if (seq_idx < img_offset) {
        cos_row = txt_cos + seq_idx * head_dim;
        sin_row = txt_sin + seq_idx * head_dim;
    } else {
        int img_idx = seq_idx - img_offset;
        cos_row = img_cos + img_idx * head_dim;
        sin_row = img_sin + img_idx * head_dim;
    }

    (void)axis_dim;
    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = vec[d];
        float x1 = vec[d + 1];

        vec[d] = x0 * c - x1 * s;
        vec[d + 1] = x1 * c + x0 * s;
    }
}

__global__ void apply_rope_unified_bf16_kernel(
    uint16_t *__restrict__ x,
    const float *__restrict__ txt_cos,
    const float *__restrict__ txt_sin,
    const float *__restrict__ img_cos,
    const float *__restrict__ img_sin,
    int seq, int img_offset, int heads, int head_dim, int axis_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq || head_idx >= heads) return;

    int hidden = heads * head_dim;
    uint16_t *vec = x + seq_idx * hidden + head_idx * head_dim;

    const float *cos_row;
    const float *sin_row;

    if (seq_idx < img_offset) {
        cos_row = txt_cos + seq_idx * head_dim;
        sin_row = txt_sin + seq_idx * head_dim;
    } else {
        int img_idx = seq_idx - img_offset;
        cos_row = img_cos + img_idx * head_dim;
        sin_row = img_sin + img_idx * head_dim;
    }

    (void)axis_dim;
    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = bf16_to_f32(vec[d]);
        float x1 = bf16_to_f32(vec[d + 1]);

        vec[d] = f32_to_bf16(x0 * c - x1 * s);
        vec[d + 1] = f32_to_bf16(x1 * c + x0 * s);
    }
}

__global__ void apply_rope_single_f32_kernel(
    float *__restrict__ x,
    const float *__restrict__ cos_freq,
    const float *__restrict__ sin_freq,
    int seq, int heads, int head_dim
) {
    int seq_idx = blockIdx.x;
    int head_idx = blockIdx.y;

    if (seq_idx >= seq || head_idx >= heads) return;

    int hidden = heads * head_dim;
    float *vec = x + seq_idx * hidden + head_idx * head_dim;
    const float *cos_row = cos_freq + seq_idx * head_dim;
    const float *sin_row = sin_freq + seq_idx * head_dim;

    for (int d = 0; d < head_dim; d += 2) {
        float c = cos_row[d];
        float s = sin_row[d];

        float x0 = vec[d];
        float x1 = vec[d + 1];

        vec[d] = x0 * c - x1 * s;
        vec[d + 1] = x1 * c + x0 * s;
    }
}

__global__ void rope_text_bf16_kernel(
    uint16_t *__restrict__ q,
    uint16_t *__restrict__ k,
    const float *__restrict__ cos_cache,
    const float *__restrict__ sin_cache,
    int seq, int num_q_heads, int num_kv_heads, int head_dim
) {
    int pos = blockIdx.x;
    int half_dim = head_dim / 2;

    if (pos >= seq) return;

    for (int h = threadIdx.x; h < num_q_heads; h += blockDim.x) {
        uint16_t *q_head = q + pos * (num_q_heads * head_dim) + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float x0 = bf16_to_f32(q_head[i]);
            float x1 = bf16_to_f32(q_head[i + half_dim]);
            float c = cos_cache[pos * half_dim + i];
            float s = sin_cache[pos * half_dim + i];

            q_head[i] = f32_to_bf16(x0 * c - x1 * s);
            q_head[i + half_dim] = f32_to_bf16(x0 * s + x1 * c);
        }
    }

    for (int h = threadIdx.x; h < num_kv_heads; h += blockDim.x) {
        uint16_t *k_head = k + pos * (num_kv_heads * head_dim) + h * head_dim;
        for (int i = 0; i < half_dim; i++) {
            float x0 = bf16_to_f32(k_head[i]);
            float x1 = bf16_to_f32(k_head[i + half_dim]);
            float c = cos_cache[pos * half_dim + i];
            float s = sin_cache[pos * half_dim + i];

            k_head[i] = f32_to_bf16(x0 * c - x1 * s);
            k_head[i + half_dim] = f32_to_bf16(x0 * s + x1 * c);
        }
    }
}

/* ========================================================================
 * Attention Kernels
 * ======================================================================== */

__global__ void attention_fused_bf16_kernel(
    const uint16_t *__restrict__ Q,
    const uint16_t *__restrict__ K,
    const uint16_t *__restrict__ V,
    uint16_t *__restrict__ out,
    int seq_q, int seq_k, int num_heads, int head_dim, float scale
) {
    extern __shared__ float shared_scores[];

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (query_idx >= seq_q || head_idx >= num_heads) return;

    int hidden = num_heads * head_dim;

    const uint16_t *q_row = Q + query_idx * hidden + head_idx * head_dim;
    uint16_t *out_row = out + query_idx * hidden + head_idx * head_dim;

    const uint16_t *K_head = K + head_idx * head_dim;
    const uint16_t *V_head = V + head_idx * head_dim;

    __shared__ float shared_q[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += threads) {
        shared_q[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float dot = 0.0f;
        const uint16_t *k_row = K_head + key_idx * hidden;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_q[d] * bf16_to_f32(k_row[d]);
        }
        float score = dot * scale;
        shared_scores[key_idx] = score;
        local_max = fmaxf(local_max, score);
    }

    __shared__ float s_max[256];
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float e = expf(shared_scores[key_idx] - max_val);
        shared_scores[key_idx] = e;
        local_sum += e;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / s_sum[0];
    __syncthreads();

    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq_k; key_idx++) {
            float v_val = bf16_to_f32(V_head[key_idx * hidden + d]);
            acc += shared_scores[key_idx] * v_val;
        }
        out_row[d] = f32_to_bf16(acc);
    }
}

__global__ void attention_fused_f32_kernel(
    const float *__restrict__ Q,
    const float *__restrict__ K,
    const float *__restrict__ V,
    float *__restrict__ out,
    int seq_q, int seq_k, int num_heads, int head_dim, float scale
) {
    extern __shared__ float shared_scores[];

    int query_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (query_idx >= seq_q || head_idx >= num_heads) return;

    int hidden = num_heads * head_dim;

    const float *q_row = Q + query_idx * hidden + head_idx * head_dim;
    float *out_row = out + query_idx * hidden + head_idx * head_dim;

    const float *K_head = K + head_idx * head_dim;
    const float *V_head = V + head_idx * head_dim;

    __shared__ float shared_q[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += threads) {
        shared_q[d] = q_row[d];
    }
    __syncthreads();

    float local_max = -1e30f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float dot = 0.0f;
        const float *k_row = K_head + key_idx * hidden;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_q[d] * k_row[d];
        }
        float score = dot * scale;
        shared_scores[key_idx] = score;
        local_max = fmaxf(local_max, score);
    }

    __shared__ float s_max[256];
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        float e = expf(shared_scores[key_idx] - max_val);
        shared_scores[key_idx] = e;
        local_sum += e;
    }

    __shared__ float s_sum[256];
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / s_sum[0];
    __syncthreads();

    for (int key_idx = tid; key_idx < seq_k; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < seq_k; key_idx++) {
            acc += shared_scores[key_idx] * V_head[key_idx * hidden + d];
        }
        out_row[d] = acc;
    }
}

__global__ void causal_attention_fused_bf16_kernel(
    const uint16_t *__restrict__ Q,
    const uint16_t *__restrict__ K,
    const uint16_t *__restrict__ V,
    uint16_t *__restrict__ out,
    const int *__restrict__ attention_mask,
    int seq, int num_q_heads, int num_kv_heads, int head_dim, float scale
) {
    extern __shared__ float shared_scores[];

    int query_idx = blockIdx.x;
    int q_head_idx = blockIdx.y;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    if (query_idx >= seq || q_head_idx >= num_q_heads) return;

    int q_per_kv = num_q_heads / num_kv_heads;
    int kv_head_idx = q_head_idx / q_per_kv;

    int q_hidden = num_q_heads * head_dim;
    int kv_hidden = num_kv_heads * head_dim;

    const uint16_t *q_row = Q + query_idx * q_hidden + q_head_idx * head_dim;
    uint16_t *out_row = out + query_idx * q_hidden + q_head_idx * head_dim;

    const uint16_t *K_head = K + kv_head_idx * head_dim;
    const uint16_t *V_head = V + kv_head_idx * head_dim;

    __shared__ float shared_q[MAX_HEAD_DIM];
    for (int d = tid; d < head_dim; d += threads) {
        shared_q[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    float local_max = -1e30f;
    int valid_keys = query_idx + 1;

    for (int key_idx = tid; key_idx < valid_keys; key_idx += threads) {
        if (attention_mask && !attention_mask[key_idx]) {
            shared_scores[key_idx] = -1e30f;
            continue;
        }
        float dot = 0.0f;
        const uint16_t *k_row = K_head + key_idx * kv_hidden;
        for (int d = 0; d < head_dim; d++) {
            dot += shared_q[d] * bf16_to_f32(k_row[d]);
        }
        float score = dot * scale;
        shared_scores[key_idx] = score;
        local_max = fmaxf(local_max, score);
    }

    __shared__ float s_max[256];
    s_max[tid] = local_max;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
        __syncthreads();
    }
    float max_val = s_max[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int key_idx = tid; key_idx < valid_keys; key_idx += threads) {
        if (attention_mask && !attention_mask[key_idx]) {
            shared_scores[key_idx] = 0.0f;
        } else {
            float e = expf(shared_scores[key_idx] - max_val);
            shared_scores[key_idx] = e;
            local_sum += e;
        }
    }

    __shared__ float s_sum[256];
    s_sum[tid] = local_sum;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) s_sum[tid] += s_sum[tid + s];
        __syncthreads();
    }
    float inv_sum = (s_sum[0] > 0.0f) ? (1.0f / s_sum[0]) : 0.0f;
    __syncthreads();

    for (int key_idx = tid; key_idx < valid_keys; key_idx += threads) {
        shared_scores[key_idx] *= inv_sum;
    }
    __syncthreads();

    for (int d = tid; d < head_dim; d += threads) {
        float acc = 0.0f;
        for (int key_idx = 0; key_idx < valid_keys; key_idx++) {
            float v_val = bf16_to_f32(V_head[key_idx * kv_hidden + d]);
            acc += shared_scores[key_idx] * v_val;
        }
        out_row[d] = f32_to_bf16(acc);
    }
}

/* ========================================================================
 * VAE Shaders (GroupNorm, Swish, Add, Upsample, Conv2D)
 * ======================================================================== */

__global__ void group_norm_f32_kernel(
    const float *__restrict__ x,
    const float *__restrict__ gamma,
    const float *__restrict__ beta,
    float *__restrict__ out,
    int channels,
    int spatial,
    int channels_per_group,
    float eps
) {
    int group_id = blockIdx.x;
    int tid = threadIdx.x;
    int threads = blockDim.x;

    extern __shared__ float s_mem[];
    float *s_sum = s_mem;
    float *s_sum_sq = s_mem + threads;

    int num_groups = channels / channels_per_group;
    int batch_idx = group_id / num_groups;
    int group_idx = group_id % num_groups;

    int c_start = group_idx * channels_per_group;
    int group_size = channels_per_group * spatial;

    const float *x_batch = x + batch_idx * channels * spatial;
    float *out_batch = out + batch_idx * channels * spatial;

    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = tid; i < group_size; i += threads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        float val = x_batch[c * spatial + s];
        local_sum += val;
        local_sum_sq += val * val;
    }
    s_sum[tid] = local_sum;
    s_sum_sq[tid] = local_sum_sq;
    __syncthreads();

    for (int s = threads / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
            s_sum_sq[tid] += s_sum_sq[tid + s];
        }
        __syncthreads();
    }

    float mean = s_sum[0] / (float)group_size;
    float var = s_sum_sq[0] / (float)group_size - mean * mean;
    float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < group_size; i += threads) {
        int c = c_start + i / spatial;
        int s = i % spatial;
        int idx = c * spatial + s;
        float val = (x_batch[idx] - mean) * inv_std;
        out_batch[idx] = gamma[c] * val + beta[c];
    }
}

__global__ void swish_f32_kernel(const float *x, float *out, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        float v = x[idx];
        out[idx] = v / (1.0f + expf(-v));
    }
}

/* VAE convolutions use NCHW while the attention GEMMs consume token-major
 * [batch, spatial, channels] tensors.  Keep the conversion on-device. */
__global__ void nchw_to_nhwc_f32_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    int channels, int spatial, int total
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;
    int s = idx % spatial;
    int c = (idx / spatial) % channels;
    int b = idx / (channels * spatial);
    out[((size_t)b * spatial + s) * channels + c] = in[idx];
}

__global__ void nhwc_to_nchw_f32_kernel(
    const float *__restrict__ in, float *__restrict__ out,
    int channels, int spatial, int total
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= total) return;
    int c = idx % channels;
    int s = (idx / channels) % spatial;
    int b = idx / (channels * spatial);
    out[((size_t)b * channels + c) * spatial + s] = in[idx];
}

__global__ void upsample_nearest_2x_f32_kernel(
    const float *__restrict__ x,
    float *__restrict__ out,
    int channels, int in_h, int in_w
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int out_h = in_h * 2;
    int out_w = in_w * 2;
    int out_spatial = out_h * out_w;
    int total = channels * out_spatial;

    if (idx < total) {
        int c = idx / out_spatial;
        int rem = idx % out_spatial;
        int oy = rem / out_w;
        int ox = rem % out_w;

        int iy = oy / 2;
        int ix = ox / 2;

        out[c * out_spatial + oy * out_w + ox] = x[c * in_h * in_w + iy * in_w + ix];
    }
}

__global__ void conv2d_f32_direct_kernel(
    const float *__restrict__ in,
    const float *__restrict__ weight,
    const float *__restrict__ bias,
    float *__restrict__ out,
    int batch, int in_ch, int out_ch,
    int in_h, int in_w, int out_h, int out_w,
    int kH, int kW, int stride, int padding
) {
    int out_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total_out = batch * out_ch * out_h * out_w;

    if (out_idx >= total_out) return;

    int ox = out_idx % out_w;
    int temp = out_idx / out_w;
    int oy = temp % out_h;
    temp /= out_h;
    int oc = temp % out_ch;
    int b = temp / out_ch;

    float acc = (bias != NULL) ? bias[oc] : 0.0f;

    const float *w_oc = weight + oc * in_ch * kH * kW;
    const float *in_b = in + b * in_ch * in_h * in_w;

    for (int ic = 0; ic < in_ch; ic++) {
        const float *in_ic = in_b + ic * in_h * in_w;
        const float *w_ic = w_oc + ic * kH * kW;

        for (int ky = 0; ky < kH; ky++) {
            int iy = oy * stride - padding + ky;
            if (iy >= 0 && iy < in_h) {
                for (int kx = 0; kx < kW; kx++) {
                    int ix = ox * stride - padding + kx;
                    if (ix >= 0 && ix < in_w) {
                        acc += in_ic[iy * in_w + ix] * w_ic[ky * kW + kx];
                    }
                }
            }
        }
    }

    out[out_idx] = acc;
}

/* Build a row-major [in_ch*kH*kW, count] matrix for a spatial tile. */
__global__ void im2col_f32_kernel(
    const float *__restrict__ in,
    float *__restrict__ col,
    int in_ch, int in_h, int in_w,
    int out_h, int out_w,
    int kH, int kW, int stride, int padding,
    int start, int count
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int kernel_elems = in_ch * kH * kW;
    int total = kernel_elems * count;
    if (idx >= total) return;

    int p_local = idx % count;
    int k = idx / count;
    int p = start + p_local;
    int ox = p % out_w;
    int oy = p / out_w;
    int kx = k % kW;
    int t = k / kW;
    int ky = t % kH;
    int ic = t / kH;
    int iy = oy * stride - padding + ky;
    int ix = ox * stride - padding + kx;

    float value = 0.0f;
    if (iy >= 0 && iy < in_h && ix >= 0 && ix < in_w)
        value = in[(ic * in_h + iy) * in_w + ix];
    col[(size_t)k * count + p_local] = value;
}

__global__ void add_bias_nchw_f32_kernel(float *out, const float *bias,
                                          int batch, int channels, int spatial) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int total = batch * channels * spatial;
    if (idx < total) {
        int channel = (idx / spatial) % channels;
        out[idx] += bias[channel];
    }
}

/* ========================================================================
 * Launcher Functions for Host Invocation
 * ======================================================================== */

extern "C" {

void launch_rms_norm_f32(const float *x, const float *weight, float *out, int seq, int hidden, float eps, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = threads * sizeof(float);
    rms_norm_f32_kernel<<<seq, threads, shared_size, stream>>>(x, weight, out, hidden, eps);
}

void launch_rms_norm_bf16(const uint16_t *x, const uint16_t *weight, uint16_t *out, int seq, int hidden, float eps, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = threads * sizeof(float);
    rms_norm_bf16_kernel<<<seq, threads, shared_size, stream>>>(x, weight, out, hidden, eps);
}

void launch_qk_rms_norm_f32(float *q, float *k, const float *q_weight, const float *k_weight, int seq, int heads, int head_dim, float eps, cudaStream_t stream) {
    qk_rms_norm_f32_kernel<<<seq, heads, 0, stream>>>(q, k, q_weight, k_weight, heads, head_dim, eps);
}

void launch_qk_rms_norm_bf16(uint16_t *q, uint16_t *k, const uint16_t *q_weight, const uint16_t *k_weight, int seq, int heads, int head_dim, float eps, cudaStream_t stream) {
    qk_rms_norm_bf16_kernel<<<seq, heads, 0, stream>>>(q, k, q_weight, k_weight, heads, head_dim, eps);
}

void launch_head_rms_norm_bf16(uint16_t *x, const uint16_t *weight, int seq, int heads, int head_dim, float eps, cudaStream_t stream) {
    head_rms_norm_bf16_kernel<<<seq, heads, 0, stream>>>(x, weight, heads, head_dim, eps);
}

void launch_adaln_norm_f32(const float *x, const float *shift, const float *scale, float *out, int seq, int hidden, float eps, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = 2 * threads * sizeof(float);
    adaln_norm_f32_kernel<<<seq, threads, shared_size, stream>>>(x, shift, scale, out, hidden, eps);
}

void launch_adaln_norm_bf16(const uint16_t *x, const uint16_t *shift, const uint16_t *scale, uint16_t *out, int seq, int hidden, float eps, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = 2 * threads * sizeof(float);
    adaln_norm_bf16_kernel<<<seq, threads, shared_size, stream>>>(x, shift, scale, out, hidden, eps);
}

void launch_silu_f32(float *x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_f32_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

void launch_silu_bf16(uint16_t *x, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_bf16_kernel<<<blocks, threads, 0, stream>>>(x, n);
}

void launch_silu_mul_f32(float *gate, const float *up, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_f32_kernel<<<blocks, threads, 0, stream>>>(gate, up, n);
}

void launch_silu_mul_bf16(uint16_t *gate, const uint16_t *up, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    silu_mul_bf16_kernel<<<blocks, threads, 0, stream>>>(gate, up, n);
}

void launch_gated_add_f32(float *out, const float *gate, const float *proj, int seq, int hidden, cudaStream_t stream) {
    int n = seq * hidden;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gated_add_f32_kernel<<<blocks, threads, 0, stream>>>(out, gate, proj, seq, hidden);
}

void launch_gated_add_bf16(uint16_t *out, const uint16_t *gate, const uint16_t *proj, int seq, int hidden, cudaStream_t stream) {
    int n = seq * hidden;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    gated_add_bf16_kernel<<<blocks, threads, 0, stream>>>(out, gate, proj, seq, hidden);
}

void launch_add_f32(float *out, const float *a, const float *b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_f32_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

void launch_add_bf16(uint16_t *out, const uint16_t *a, const uint16_t *b, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bf16_kernel<<<blocks, threads, 0, stream>>>(out, a, b, n);
}

void launch_add_bias_f32(float *out, const float *bias, int rows, int cols, cudaStream_t stream) {
    int n = rows * cols;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_bias_f32_kernel<<<blocks, threads, 0, stream>>>(out, bias, rows, cols);
}

void launch_softmax_f32(float *x, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = threads * sizeof(float);
    softmax_f32_kernel<<<rows, threads, shared_size, stream>>>(x, rows, cols);
}

void launch_softmax_bf16(uint16_t *x, int rows, int cols, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = threads * sizeof(float);
    softmax_bf16_kernel<<<rows, threads, shared_size, stream>>>(x, rows, cols);
}

void launch_f32_to_bf16(const float *in, uint16_t *out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_bf16_to_f32(const uint16_t *in, float *out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, n);
}

void launch_split_qkv_mlp_f32(const float *fused, float *q, float *k, float *v, float *gate, float *up, int seq, int hidden, int mlp_hidden, cudaStream_t stream) {
    int total_dim = hidden + mlp_hidden;
    int threads = 256;
    dim3 grid(seq, (total_dim + threads - 1) / threads);
    split_qkv_mlp_f32_kernel<<<grid, threads, 0, stream>>>(fused, q, k, v, gate, up, seq, hidden, mlp_hidden);
}

void launch_split_qkv_mlp_bf16(const uint16_t *fused, uint16_t *q, uint16_t *k, uint16_t *v, uint16_t *gate, uint16_t *up, int seq, int hidden, int mlp_hidden, cudaStream_t stream) {
    int total_dim = hidden + mlp_hidden;
    int threads = 256;
    dim3 grid(seq, (total_dim + threads - 1) / threads);
    split_qkv_mlp_bf16_kernel<<<grid, threads, 0, stream>>>(fused, q, k, v, gate, up, seq, hidden, mlp_hidden);
}

void launch_concat_attn_mlp_f32(const float *attn, const float *mlp, float *out, int seq, int hidden, int mlp_hidden, cudaStream_t stream) {
    int total_dim = hidden + mlp_hidden;
    int threads = 256;
    dim3 grid(seq, (total_dim + threads - 1) / threads);
    concat_attn_mlp_f32_kernel<<<grid, threads, 0, stream>>>(attn, mlp, out, seq, hidden, mlp_hidden);
}

void launch_concat_attn_mlp_bf16(const uint16_t *attn, const uint16_t *mlp, uint16_t *out, int seq, int hidden, int mlp_hidden, cudaStream_t stream) {
    int total_dim = hidden + mlp_hidden;
    int threads = 256;
    dim3 grid(seq, (total_dim + threads - 1) / threads);
    concat_attn_mlp_bf16_kernel<<<grid, threads, 0, stream>>>(attn, mlp, out, seq, hidden, mlp_hidden);
}

void launch_concat_seq_bf16(uint16_t *out, const uint16_t *a, const uint16_t *b, int seq_a, int seq_b, int hidden, cudaStream_t stream) {
    int n = (seq_a + seq_b) * hidden;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    concat_seq_bf16_kernel<<<blocks, threads, 0, stream>>>(out, a, b, seq_a, seq_b, hidden);
}

void launch_slice_seq_bf16(uint16_t *out, const uint16_t *in, int seq_out, int hidden, int start, cudaStream_t stream) {
    int n = seq_out * hidden;
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    slice_seq_bf16_kernel<<<blocks, threads, 0, stream>>>(out, in, seq_out, hidden, start);
}

void launch_transpose_to_heads_bf16(const uint16_t *in, uint16_t *out, int seq, int heads, int head_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    transpose_to_heads_bf16_kernel<<<grid, head_dim, 0, stream>>>(in, out, seq, heads, head_dim);
}

void launch_transpose_from_heads_bf16(const uint16_t *in, uint16_t *out, int seq, int heads, int head_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    transpose_from_heads_bf16_kernel<<<grid, head_dim, 0, stream>>>(in, out, seq, heads, head_dim);
}

void launch_apply_rope_2d_f32(float *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    apply_rope_2d_f32_kernel<<<grid, 1, 0, stream>>>(x, cos_freq, sin_freq, seq, heads, head_dim, axis_dim);
}

void launch_apply_rope_2d_bf16(uint16_t *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, int axis_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    apply_rope_2d_bf16_kernel<<<grid, 1, 0, stream>>>(x, cos_freq, sin_freq, seq, heads, head_dim, axis_dim);
}

void launch_apply_rope_unified_f32(float *x, const float *txt_cos, const float *txt_sin, const float *img_cos, const float *img_sin, int seq, int img_offset, int heads, int head_dim, int axis_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    apply_rope_unified_f32_kernel<<<grid, 1, 0, stream>>>(x, txt_cos, txt_sin, img_cos, img_sin, seq, img_offset, heads, head_dim, axis_dim);
}

void launch_apply_rope_unified_bf16(uint16_t *x, const float *txt_cos, const float *txt_sin, const float *img_cos, const float *img_sin, int seq, int img_offset, int heads, int head_dim, int axis_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    apply_rope_unified_bf16_kernel<<<grid, 1, 0, stream>>>(x, txt_cos, txt_sin, img_cos, img_sin, seq, img_offset, heads, head_dim, axis_dim);
}

void launch_apply_rope_single_f32(float *x, const float *cos_freq, const float *sin_freq, int seq, int heads, int head_dim, cudaStream_t stream) {
    dim3 grid(seq, heads);
    apply_rope_single_f32_kernel<<<grid, 1, 0, stream>>>(x, cos_freq, sin_freq, seq, heads, head_dim);
}

void launch_rope_text_bf16(uint16_t *q, uint16_t *k, const float *cos_cache, const float *sin_cache, int seq, int num_q_heads, int num_kv_heads, int head_dim, cudaStream_t stream) {
    int max_heads = (num_q_heads > num_kv_heads) ? num_q_heads : num_kv_heads;
    int threads = (max_heads < 256) ? max_heads : 256;
    rope_text_bf16_kernel<<<seq, threads, 0, stream>>>(q, k, cos_cache, sin_cache, seq, num_q_heads, num_kv_heads, head_dim);
}

int launch_attention_fused_f32(const float *Q, const float *K, const float *V, float *out, int seq_q, int seq_k, int num_heads, int head_dim, float scale, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = seq_k * sizeof(float);
    if (shared_size > 48 * 1024) {
        cudaError_t config_err = cudaFuncSetAttribute(attention_fused_f32_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_size);
        if (config_err != cudaSuccess) return 0;
    }
    dim3 grid(seq_q, num_heads);
    attention_fused_f32_kernel<<<grid, threads, shared_size, stream>>>(Q, K, V, out, seq_q, seq_k, num_heads, head_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

int launch_attention_fused_bf16(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *out, int seq_q, int seq_k, int num_heads, int head_dim, float scale, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = seq_k * sizeof(float);
    if (shared_size > 48 * 1024) {
        cudaError_t config_err = cudaFuncSetAttribute(attention_fused_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_size);
        if (config_err != cudaSuccess) return 0;
    }
    dim3 grid(seq_q, num_heads);
    attention_fused_bf16_kernel<<<grid, threads, shared_size, stream>>>(Q, K, V, out, seq_q, seq_k, num_heads, head_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

int launch_causal_attention_fused_bf16(const uint16_t *Q, const uint16_t *K, const uint16_t *V, uint16_t *out, const int *attention_mask, int seq, int num_q_heads, int num_kv_heads, int head_dim, float scale, cudaStream_t stream) {
    int threads = 256;
    size_t shared_size = seq * sizeof(float);
    if (shared_size > 48 * 1024) {
        cudaError_t config_err = cudaFuncSetAttribute(causal_attention_fused_bf16_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared_size);
        if (config_err != cudaSuccess) return 0;
    }
    dim3 grid(seq, num_q_heads);
    causal_attention_fused_bf16_kernel<<<grid, threads, shared_size, stream>>>(Q, K, V, out, attention_mask, seq, num_q_heads, num_kv_heads, head_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

void launch_group_norm_f32(const float *x, const float *gamma, const float *beta, float *out, int batch, int channels, int spatial, int channels_per_group, float eps, cudaStream_t stream) {
    int num_groups = channels / channels_per_group;
    int threads = 256;
    size_t shared_size = 2 * threads * sizeof(float);
    group_norm_f32_kernel<<<batch * num_groups, threads, shared_size, stream>>>(x, gamma, beta, out, channels, spatial, channels_per_group, eps);
}

void launch_swish_f32(const float *x, float *out, int n, cudaStream_t stream) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    swish_f32_kernel<<<blocks, threads, 0, stream>>>(x, out, n);
}

void launch_nchw_to_nhwc_f32(const float *in, float *out, int batch, int channels, int spatial, cudaStream_t stream) {
    int total = batch * channels * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nchw_to_nhwc_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, channels, spatial, total);
}

void launch_nhwc_to_nchw_f32(const float *in, float *out, int batch, int channels, int spatial, cudaStream_t stream) {
    int total = batch * channels * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    nhwc_to_nchw_f32_kernel<<<blocks, threads, 0, stream>>>(in, out, channels, spatial, total);
}

void launch_upsample_nearest_2x_f32(const float *x, float *out, int channels, int in_h, int in_w, cudaStream_t stream) {
    int total = channels * (in_h * 2) * (in_w * 2);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    upsample_nearest_2x_f32_kernel<<<blocks, threads, 0, stream>>>(x, out, channels, in_h, in_w);
}

void launch_conv2d_f32(const float *in, const float *weight, const float *bias, float *out, int batch, int in_ch, int out_ch, int in_h, int in_w, int out_h, int out_w, int kH, int kW, int stride, int padding, cudaStream_t stream) {
    int total_out = batch * out_ch * out_h * out_w;
    int threads = 256;
    int blocks = (total_out + threads - 1) / threads;
    conv2d_f32_direct_kernel<<<blocks, threads, 0, stream>>>(in, weight, bias, out, batch, in_ch, out_ch, in_h, in_w, out_h, out_w, kH, kW, stride, padding);
}

int launch_im2col_f32(const float *in, float *col, int in_ch, int in_h, int in_w, int out_h, int out_w, int kH, int kW, int stride, int padding, int start, int count, cudaStream_t stream) {
    int64_t total = (int64_t)in_ch * kH * kW * count;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    im2col_f32_kernel<<<blocks, threads, 0, stream>>>(in, col, in_ch, in_h, in_w, out_h, out_w, kH, kW, stride, padding, start, count);
    return cudaPeekAtLastError() == cudaSuccess;
}

void launch_add_bias_nchw_f32(float *out, const float *bias, int batch, int channels, int spatial, cudaStream_t stream) {
    int total = batch * channels * spatial;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    add_bias_nchw_f32_kernel<<<blocks, threads, 0, stream>>>(out, bias, batch, channels, spatial);
}

} /* extern "C" */
