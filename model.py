"""
Flash Attention in CUDA from Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - vector_add
__global__ void vector_add(const float* a, const float* b, float* c, int n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        c[idx] = a[idx] + b[idx];
    }
}

# Step 2 - scale_array
__global__ void scale_array(float* a, float scalar, int n) {
    
    int idx = threadIdx.x + blockDim.x * blockIdx.x;

    if (idx < n){
        a[idx] = a[idx] * scalar;
    }
}

# Step 3 - elementwise_exp
__global__ void elementwise_exp(float* a, int n) {
    // TODO: replace each a[i] with expf(a[i])
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n){
        a[idx] = expf(a[idx]);
    }
}

# Step 4 - row_max
__global__ void row_max(const float* matrix, float* out, int rows, int cols) {
    // TODO: compute the max of each row and write it to out[r].
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows){

        float max_val = matrix[idx * cols];

        for( int i = 1; i < cols ; i++){
            float val = matrix[idx * cols + i];

            if (val > max_val){
                max_val = val;
            }
        }
        out[idx] = max_val;
    }

}

# Step 5 - row_sum
__global__ void row_sum(const float* matrix, float* out, int rows, int cols) {
    // TODO: write out[r] = sum of matrix row r
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < rows){
        float sum = 0.0f;

        for(int i =0; i<cols ;i++){

            sum += matrix[idx * cols + i];
        
        }
            
            out[idx] = sum; 
    }
}

# Step 6 - dot_product
__device__ float dot_product(const float* a, const float* b, int n) {
    // TODO: return the dot product of a and b

    float sum = 0.0f;
    
    for(int i =0; i<n ;i++){
        sum += a[i]*b[i];
    }

    return sum;
}

# Step 7 - matmul
__global__ void matmul(const float* a, const float* b, float* c, int m, int k, int n) {
    // TODO: compute C = A * B for row-major matrices

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if(row < m && col <n){
        float sum =0.0f;

        for(int i=0 ; i<k ;i++){
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

# Step 8 - transpose
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}

# Step 9 - qk_scores
__global__ void qk_scores(const float* q,
                          const float* k,
                          float* scores,
                          int seq_len,
                          int head_dim)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < seq_len && j < seq_len)
    {
        const float* q_row = q + i * head_dim;
        const float* k_row = k + j * head_dim;

        float dot = dot_product(q_row, k_row, head_dim);

        scores[i * seq_len + j] = dot / sqrtf((float)head_dim);
    }
}

# Step 10 - softmax_rows
__global__ void softmax_rows(float* matrix, int rows, int cols)
{
    // One thread handles one row
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= rows)
        return;

    int offset = row * cols;

    // --------------------------------------------------
    // 1. Find maximum value in this row
    // --------------------------------------------------

    float max_val = matrix[offset];

    for (int col = 1; col < cols; col++)
    {
        float value = matrix[offset + col];

        if (value > max_val)
            max_val = value;
    }

    // --------------------------------------------------
    // 2. Compute exp(x - max) and sum
    // --------------------------------------------------

    float sum = 0.0f;

    for (int col = 0; col < cols; col++)
    {
        float value = expf(matrix[offset + col] - max_val);

        matrix[offset + col] = value;

        sum += value;
    }

    // --------------------------------------------------
    // 3. Normalize
    // --------------------------------------------------

    for (int col = 0; col < cols; col++)
    {
        matrix[offset + col] /= sum;
    }
}

# Step 11 - pv_matmul (not yet solved)
# TODO: implement

# Step 12 - naive_attention (not yet solved)
# TODO: implement

# Step 13 - online_max (not yet solved)
# TODO: implement

# Step 14 - correction_factor (not yet solved)
# TODO: implement

# Step 15 - update_running_sum (not yet solved)
# TODO: implement

# Step 16 - rescale_output (not yet solved)
# TODO: implement

# Step 17 - load_tile (not yet solved)
# TODO: implement

# Step 18 - tile_scores (not yet solved)
# TODO: implement

# Step 19 - tile_rowmax (not yet solved)
# TODO: implement

# Step 20 - tile_exp (not yet solved)
# TODO: implement

# Step 21 - tile_rowsum (not yet solved)
# TODO: implement

# Step 22 - accumulate_pv (not yet solved)
# TODO: implement

# Step 23 - flash_attention_kernel (not yet solved)
# TODO: implement

# Step 24 - flash_attention_launcher (not yet solved)
# TODO: implement

# Step 25 - causal_mask (not yet solved)
# TODO: implement

# Step 26 - flash_attention_causal_kernel (not yet solved)
# TODO: implement

