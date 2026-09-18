// Builds an IVF-PQ index through the public API, for the benchmark scripts:
//
//   ivf_pq_build_index --data_type float|int8|uint8 --data_file <base.bin>
//                      --index_prefix <prefix> --dim N --ivf_nlist N
//                      --ivf_pq_chunks N [--page_size N]
//
// Writes <prefix>_ivf_pq_index.bin, <prefix>_ivf_raw_vectors.bin and the
// upstream-format <prefix>_pq_pivots.bin / _pq_compressed.bin.

#include "bufann/bufann_api.h"

#include <chrono>
#include <cstring>
#include <iostream>
#include <string>

using namespace diskann::inplace;

namespace {

std::string get_arg(int argc, char** argv, const char* name, const std::string& def) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    }
    return def;
}

template<typename T>
int run(const std::string& data_file, const std::string& prefix, const BufANNConfig& cfg) {
    auto t0 = std::chrono::steady_clock::now();
    BufANNIndex<T>* idx = bufann_build<T>(data_file, prefix, cfg);
    double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::cout << "built IVF-PQ index at " << prefix << ": " << idx->ivf->index.assignments.cluster_id.size()
              << " vectors, nlist " << cfg.ivf_nlist << ", " << cfg.ivf_pq_chunks << " PQ chunks, " << s << " s"
              << std::endl;
    bufann_free<T>(idx);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string data_type = get_arg(argc, argv, "--data_type", "float");
    const std::string data_file = get_arg(argc, argv, "--data_file", "");
    const std::string prefix = get_arg(argc, argv, "--index_prefix", "");
    BufANNConfig cfg;
    cfg.index_type = IndexType::IvfPq;
    cfg.dim = uint32_t(std::stoul(get_arg(argc, argv, "--dim", "0")));
    cfg.ivf_nlist = uint32_t(std::stoul(get_arg(argc, argv, "--ivf_nlist", "0")));
    cfg.ivf_pq_chunks = uint32_t(std::stoul(get_arg(argc, argv, "--ivf_pq_chunks", "0")));
    cfg.page_size = uint32_t(std::stoul(get_arg(argc, argv, "--page_size", "4096")));
    if (data_file.empty() || prefix.empty() || cfg.dim == 0 || cfg.ivf_nlist == 0 || cfg.ivf_pq_chunks == 0) {
        std::cerr << "Usage: ivf_pq_build_index --data_type float|int8|uint8 --data_file <base.bin> "
                     "--index_prefix <prefix> --dim N --ivf_nlist N --ivf_pq_chunks N [--page_size N]"
                  << std::endl;
        return 2;
    }
    try {
        if (data_type == "float") return run<float>(data_file, prefix, cfg);
        if (data_type == "uint8") return run<uint8_t>(data_file, prefix, cfg);
        if (data_type == "int8") return run<int8_t>(data_file, prefix, cfg);
        std::cerr << "ERROR: --data_type must be float, int8 or uint8" << std::endl;
        return 2;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
