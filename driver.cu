#include <iostream>
#include <functional>
#include <vector>
#include <unordered_map>
#include <string>
#include <random> 
#include <cuda_runtime.h>

constexpr int NUM_TUPLES = 1024;

#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)            \
                      << " (" << __FILE__ << ":" << __LINE__ << ")" << "\n";  \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// ====== GraphletManager ======
class GraphletManager {
public:
    int* get_column(int col_id) {
        if (columns_.find(col_id) != columns_.end())
            return columns_[col_id];

        int* d_data;
        CUDA_CHECK(cudaMalloc(&d_data, NUM_TUPLES * sizeof(int)));

        int h_data[NUM_TUPLES];
        for (int i = 0; i < NUM_TUPLES; ++i) 
            h_data[i] = i;  // Fill with 0 to NUM_TUPLES-1

        CUDA_CHECK(cudaMemcpy(d_data, h_data, NUM_TUPLES * sizeof(int), cudaMemcpyHostToDevice));
        columns_[col_id] = d_data;
        return d_data;
    }

    ~GraphletManager() {
        for (auto& pair : columns_)
            cudaFree(pair.second);
    }

private:
    std::unordered_map<int, int*> columns_;
};


// ====== GPU kernels ======
__global__ void pipeline1_kernel(int* col0, int* col1, int* out_vals, int* out_count) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= NUM_TUPLES) return;

    int val0 = col0[tid];
    if (val0 > 10) {
        int idx = atomicAdd(out_count, 1);
        out_vals[idx] = col1[tid];
    }
}

// ====== Host-side pipeline functions ======
void pipeline1(GraphletManager& gm, std::vector<int>& output) {
    int* col0 = gm.get_column(0);
    int* col1 = gm.get_column(1);

    int h_col1[NUM_TUPLES];
    CUDA_CHECK(cudaMemcpy(h_col1, col1, NUM_TUPLES * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 0; i < NUM_TUPLES; ++i) std::cout << h_col1[i] << " ";
    std::cout << std::endl;

    int* d_out_vals;
    int* d_out_count;
    CUDA_CHECK(cudaMalloc(&d_out_vals, NUM_TUPLES * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_count, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_out_count, 0, sizeof(int)));

    pipeline1_kernel<<<(NUM_TUPLES + 255)/256, 256>>>(col0, col1, d_out_vals, d_out_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int h_count;
    CUDA_CHECK(cudaMemcpy(&h_count, d_out_count, sizeof(int), cudaMemcpyDeviceToHost));

    output.resize(h_count);
    if (h_count > 0)
        CUDA_CHECK(cudaMemcpy(output.data(), d_out_vals, h_count * sizeof(int), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_out_vals));
    CUDA_CHECK(cudaFree(d_out_count));
}

void pipeline2(GraphletManager& gm, std::vector<int>& output) {
    std::cout << "pipeline2 not implemented yet.\n";
}

// ====== Pipeline Launcher ======
using PipelineFn = std::function<void(GraphletManager&, std::vector<int>&)>;

std::unordered_map<std::string, PipelineFn> pipeline_registry = {
    {"pipeline1", pipeline1},
    {"pipeline2", pipeline2}
};

void launch_pipeline_by_name(const std::string& name, GraphletManager& gm) {
    auto it = pipeline_registry.find(name);
    if (it == pipeline_registry.end()) {
        std::cerr << "Unknown pipeline name: " << name << std::endl;
        return;
    }

    std::vector<int> results;
    it->second(gm, results);

    std::cout << name << " output:\n";
    for (int v : results) std::cout << v << " ";
    std::cout << "\nTotal matched: " << results.size() << std::endl;
}

// ====== Main Driver ======
int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./pipeline_driver <pipeline_name>" << std::endl;
        return 1;
    }

    std::string pipeline_name = argv[1];
    GraphletManager gm;
    launch_pipeline_by_name(pipeline_name, gm);
    return 0;
}
