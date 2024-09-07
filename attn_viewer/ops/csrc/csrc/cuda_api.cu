#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <torch/python.h>

#include "cuda_api.h"
#include <stdio.h>
#include <stdint.h>

template<typename T>
struct SumOp {
    __device__ inline T operator()(T const & x, T const & y) { return x + y;}
};

template<typename T>
struct MaxOp {
    __device__ inline T operator()(T const & x, T const & y) { return x > y ? x : y; }
};

template <int THREADS>
struct AllReduce {
    static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
    template <typename T, typename Operator>
    static __device__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
        return AllReduce<OFFSET>::run(x, op);
    }
};

// Specialization for 2 threads, which stops the recursion
template <>
struct AllReduce<2> {
    template <typename T, typename Operator>
    static __device__ T run(T x, Operator &op) {
        x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
        return x;
    }
};


template<typename tensor_t, int num_threads>
__device__ int count(const tensor_t* tensor1, const tensor_t* tensor2, int topk) {
    // tensor.shape = (topk)
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    int cnt = 0;
    for (int i = 0; i < topk; i++) {
        auto t1_reg = tensor1[i];
        for (int j = 0; j < topk; j += bdim) {
            auto idx = j + tid;
            if (idx >= topk) {
                break;
            }
            auto t2_reg = tensor2[idx];
            if (t1_reg == t2_reg) {
                cnt += 1;
                break;
            }
        }
    }

    SumOp<int> op;
    cnt = AllReduce<num_threads>::run(cnt, op);

    return cnt;
}

template<typename tensor_t, int num_threads>
__global__ void interset_count_nton_kernel(const tensor_t* tensor1_list, const tensor_t* tensor2_list, int* out_list, int seqlen, int topk) {
    /*
       expected return (seqlen, topk) n to n
       blockx, blocky for (seq1, seq2)
    */

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int gdimx = gridDim.x;
    int gdimy = gridDim.y;

    for (int xi = 0; xi < seqlen; xi += gdimx) {
        auto t1_idx = xi + bidx;
        auto tensor1 = tensor1_list + t1_idx * topk;
        auto out = out_list + t1_idx * seqlen;
        if (t1_idx >= seqlen) {
            break;
        }

        for (int yi = 0; yi < seqlen; yi += gdimy) {
            auto t2_idx = yi + bidy;
            auto offset2 = t2_idx * topk;
            auto tensor2 = tensor2_list + offset2;
            if (t2_idx >= seqlen) {
                break;
            }
            auto cnt = count<tensor_t, num_threads>(tensor1, tensor2, topk);
            if (tid == 0) {
                out[t2_idx] = cnt;
            }
        }
    }
}

template<typename tensor_t, int num_threads>
__device__ int union_count(const tensor_t* tensor1, const tensor_t* tensor2, bool* counter, int topk, int max) {
    // tensor.shape = (topk)
    int tid = threadIdx.x;
    int bdim = blockDim.x;

    // union set
    for (int j = 0; j < topk; j += bdim) {
        auto idx = j + tid;
        if (idx >= topk) {
            break;
        }
        auto t1_reg = tensor1[idx];
        auto t2_reg = tensor2[idx];
        counter[t1_reg] = true;
        counter[t2_reg] = true;
    }

    // count
    int cnt = 0;
    for (int j = 0; j < max; j += bdim) {
        auto idx = j + tid;
        if (idx >= max) {
            break;
        }
        if (counter[idx]) {
            cnt += 1;
        }
    }

    SumOp<int> op;
    cnt = AllReduce<num_threads>::run(cnt, op);

    return cnt;
}


template<typename tensor_t, int num_threads>
__global__ void unionset_count_nton_kernel(const tensor_t* tensor1_list, const tensor_t* tensor2_list, int* out_list, bool* counter_list, int seqlen, int topk, int max) {
    /*
       expected return (seqlen, topk) n to n
       blockx, blocky for (seq1, seq2)
    */

    int tid = threadIdx.x;
    int bidx = blockIdx.x;
    int bidy = blockIdx.y;

    int gdimx = gridDim.x;
    int gdimy = gridDim.y;

    for (int xi = 0; xi < seqlen; xi += gdimx) {
        auto t1_idx = xi + bidx;
        auto tensor1 = tensor1_list + t1_idx * topk;
        auto out = out_list + t1_idx * seqlen;
        auto counters = counter_list + t1_idx * max * seqlen;
        if (t1_idx >= seqlen) {
            break;
        }

        for (int yi = 0; yi < seqlen; yi += gdimy) {
            auto t2_idx = yi + bidy;
            auto offset2 = t2_idx * topk;
            auto tensor2 = tensor2_list + offset2;
            auto counter = counters + t2_idx * max;
            if (t2_idx >= seqlen) {
                break;
            }
            auto cnt = union_count<tensor_t, num_threads>(tensor1, tensor2, counter, topk, max);
            if (tid == 0) {
                out[t2_idx] = cnt;
            }
        }
    }
}



// intsertsetction count
torch::Tensor interset_count_nton(torch::Tensor &tensor1_list, torch::Tensor &tensor2_list) {
    CHECK_INPUT(tensor1_list);
    CHECK_INPUT(tensor2_list);
    // tensor.shape = (seqlen, seqlen)
    // out.shape = (seqlen, seqlen)

    int seqlen = tensor1_list.sizes()[0];
    int topk = tensor1_list.sizes()[1];
    auto opt = tensor1_list.options();
    torch::Tensor out_list = torch::empty({seqlen, seqlen}, opt.dtype(torch::kInt32));


    const int num_threads_block_x = 512;
    const int num_threads_block_y = 512;
    // max warp size is 32
    const int num_threads = 32;

    dim3 grid(num_threads_block_x, num_threads_block_y);
    dim3 block(num_threads);
    
    AT_DISPATCH_INTEGRAL_TYPES(tensor1_list.scalar_type(), "interset_count_nton", [&] {
        using elem_type = scalar_t;

        auto kernel = interset_count_nton_kernel<elem_type, num_threads>;
        kernel<<<grid, block, 0>>>((elem_type*)tensor1_list.data_ptr(), (elem_type*)tensor2_list.data_ptr(), (int*)out_list.data_ptr(), seqlen, topk);
    });

    CUDA_ERROR_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    return out_list;
}

// union count
torch::Tensor unionset_count_nton(torch::Tensor &tensor1_list, torch::Tensor &tensor2_list, int max) {
    CHECK_INPUT(tensor1_list);
    CHECK_INPUT(tensor2_list);
    // tensor.shape = (seqlen, seqlen)
    // out.shape = (seqlen, seqlen)

    int seqlen = tensor1_list.sizes()[0];
    int topk = tensor1_list.sizes()[1];
    auto opt = tensor1_list.options();
    torch::Tensor out_list = torch::empty({seqlen, seqlen}, opt.dtype(torch::kInt32));
    torch::Tensor counter_list = torch::zeros({seqlen, seqlen, max}, opt.dtype(torch::kBool));


    const int num_threads_block_x = 512;
    const int num_threads_block_y = 512;
    // max warp size is 32
    const int num_threads = 32;

    dim3 grid(num_threads_block_x, num_threads_block_y);
    dim3 block(num_threads);
    
    AT_DISPATCH_INTEGRAL_TYPES(tensor1_list.scalar_type(), "interset_count_nton", [&] {
        using elem_type = scalar_t;

        auto kernel = unionset_count_nton_kernel<elem_type, num_threads>;
        kernel<<<grid, block, 0>>>((elem_type*)tensor1_list.data_ptr(), (elem_type*)tensor2_list.data_ptr(), (int*)out_list.data_ptr(), (bool*)counter_list.data_ptr(), seqlen, topk, max);
    });

    CUDA_ERROR_CHECK(cudaGetLastError());
    cudaDeviceSynchronize();

    return out_list;
}




PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("package_name", &function_name, "function_docstring"")

  m.def("interset_count_nton", &interset_count_nton, "all to all intersection count");
  m.def("unionset_count_nton", &unionset_count_nton, "all to all unionset count");
}
