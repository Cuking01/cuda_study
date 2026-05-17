#include <cuda_fp16.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <cstddef>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fe = cudnn_frontend;

#define CHECK_CUDNN(x)                                      \
    do {                                                    \
        cudnnStatus_t status = (x);                         \
        if (status != CUDNN_STATUS_SUCCESS) {               \
            throw std::runtime_error(cudnnGetErrorString(status)); \
        }                                                   \
    } while (0)

struct SdpaPlan {
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t workspace_size = 0;
    void* workspace = nullptr;

    std::shared_ptr<fe::graph::Tensor_attributes> Q;
    std::shared_ptr<fe::graph::Tensor_attributes> K;
    std::shared_ptr<fe::graph::Tensor_attributes> V;
    std::shared_ptr<fe::graph::Tensor_attributes> O;

    explicit SdpaPlan(int n, int heads) {
        CHECK_CUDNN(cudnnCreate(&handle));

        constexpr int64_t d = 128;
        const int64_t h = heads;
        const int64_t s = n;
        float scale = 1.0f / std::sqrt(static_cast<float>(d));

        graph = std::make_shared<fe::graph::Graph>();

        graph->set_io_data_type(fe::DataType_t::HALF)
             .set_intermediate_data_type(fe::DataType_t::FLOAT)
             .set_compute_data_type(fe::DataType_t::FLOAT);

        std::vector<int64_t> dim = {1, h, s, d};
        std::vector<int64_t> stride = {h * s * d, s * d, d, 1};

        Q = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("Q")
                .set_dim(dim)
                .set_stride(stride)
                .set_data_type(fe::DataType_t::HALF));

        K = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("K")
                .set_dim(dim)
                .set_stride(stride)
                .set_data_type(fe::DataType_t::HALF));

        V = graph->tensor(
            fe::graph::Tensor_attributes()
                .set_name("V")
                .set_dim(dim)
                .set_stride(stride)
                .set_data_type(fe::DataType_t::HALF));

        auto sdpa_options =
            fe::graph::SDPA_attributes()
                .set_name("self_attention")
                .set_is_inference(true)
                .set_attn_scale(scale)
                .set_causal_mask(true);

        auto result = graph->sdpa(Q, K, V, sdpa_options);

        O = std::get<0>(result);
        O->set_output(true)
          .set_dim(dim)
          .set_stride(stride)
          .set_data_type(fe::DataType_t::HALF);

        graph->validate();
        graph->build_operation_graph(handle);
        graph->create_execution_plans({fe::HeurMode_t::A});
        graph->check_support(handle);
        graph->build_plans(handle);

        workspace_size = graph->get_workspace_size();

        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
        }
    }

    ~SdpaPlan() {
        if (workspace) {
            cudaFree(workspace);
        }
        if (handle) {
            cudnnDestroy(handle);
        }
    }

    void run(cudaStream_t stream,
             const half* q,
             const half* k,
             const half* v,
             half* o) {
        CHECK_CUDNN(cudnnSetStream(handle, stream));

        std::unordered_map<std::shared_ptr<fe::graph::Tensor_attributes>, void*> variant_pack = {
            {Q, const_cast<half*>(q)},
            {K, const_cast<half*>(k)},
            {V, const_cast<half*>(v)},
            {O, o},
        };

        graph->execute(handle, variant_pack, workspace);
    }
};


void fa_cudnn(cudaStream_t stream,const half* q,const half* k,const half* v,half* o,int n,int heads)
{
    using PlanKey = std::pair<int, int>;

    struct PlanKeyHash {
        std::size_t operator()(const PlanKey& key) const {
            return (static_cast<std::size_t>(key.first) << 32) ^ static_cast<std::size_t>(key.second);
        }
    };

    static std::unordered_map<PlanKey, std::unique_ptr<SdpaPlan>, PlanKeyHash> plan_cache;
    PlanKey key{n, heads};
    auto it = plan_cache.find(key);
    if (it == plan_cache.end()) {
        it = plan_cache.emplace(key, std::make_unique<SdpaPlan>(n, heads)).first;
    }
    it->second->run(stream, q, k, v, o);
}
