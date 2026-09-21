// bufann_api.cpp — implementation of the BufANN public interface.

#include "bufann/bufann_api.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/stat.h>
#include <unistd.h>

#include "ann_exception.h"
#include "aux_utils.h"
#include "distance.h"
#include "index.h"
#include "neighbor.h"
#include "parameters.h"
#include "partition_and_pq.h"
#include "utils.h"
#include "bufann/inplace_backend.h"
#include "bufann/inplace_graph_ops.h"

using namespace diskann::inplace;

// ---------------------------------------------------------------------------
// Internal helpers (file-local)
// ---------------------------------------------------------------------------

namespace {

template<typename T>
struct QueryThreadCache {
    std::unique_ptr<InPlaceSearchScratch> scratch;
    uint32_t aligned_dim = 0;
    uint32_t n_chunks = 0;
    uint32_t elem_size = 0;
    uint32_t search_list_size = 0;
    uint32_t max_degree = 0;
    uint32_t beamwidth = 0;
};

template<typename T>
static InPlaceSearchScratch& query_scratch_for(uint32_t aligned_dim,
                                               uint32_t n_chunks,
                                               uint32_t elem_size,
                                               uint32_t search_list_size,
                                               uint32_t max_degree,
                                               uint32_t beamwidth) {
    thread_local QueryThreadCache<T> cache;
    if (!cache.scratch ||
        cache.aligned_dim != aligned_dim ||
        cache.n_chunks != n_chunks ||
        cache.elem_size != elem_size ||
        cache.search_list_size != search_list_size ||
        cache.max_degree != max_degree ||
        cache.beamwidth != beamwidth) {
        cache.scratch = std::make_unique<InPlaceSearchScratch>();
        cache.scratch->init(aligned_dim, n_chunks, elem_size,
                            search_list_size, max_degree, beamwidth);
        cache.aligned_dim = aligned_dim;
        cache.n_chunks = n_chunks;
        cache.elem_size = elem_size;
        cache.search_list_size = search_list_size;
        cache.max_degree = max_degree;
        cache.beamwidth = beamwidth;
    }
    return *cache.scratch;
}

static bool api_path_exists(const std::string& path) {
    return access(path.c_str(), F_OK) == 0;
}

static uint32_t read_bin_cols_i32(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return 0;
    int32_t rows = 0, cols = 0;
    in.read(reinterpret_cast<char*>(&rows), sizeof(rows));
    in.read(reinterpret_cast<char*>(&cols), sizeof(cols));
    if (!in || rows < 0 || cols <= 0)
        return 0;
    return static_cast<uint32_t>(cols);
}

static uint32_t resolve_existing_pq_chunks(const std::string& pq_prefix,
                                           uint32_t configured_chunks) {
    if (configured_chunks == 0 || pq_prefix.empty())
        return configured_chunks;
    const std::string codes_path = pq_prefix + "_pq_compressed.bin";
    const uint32_t file_chunks = read_bin_cols_i32(codes_path);
    if (file_chunks == 0)
        return configured_chunks;
    if (file_chunks != configured_chunks) {
        diskann::cerr << "WARNING: configured pq_chunks=" << configured_chunks
                      << " but " << codes_path << " stores " << file_chunks
                      << " chunks; using file metadata." << std::endl;
    }
    return file_chunks;
}

// Create a Distance<T> object on the heap for the given metric.
// Ownership is transferred to the caller.
template<typename T>
diskann::Distance<T>* make_distance(diskann::Metric metric);

template<>
diskann::Distance<float>* make_distance<float>(diskann::Metric metric) {
    if (metric == diskann::Metric::COSINE)
        return new diskann::DistanceCosineFloat();
    return new diskann::DistanceL2();
}

template<>
diskann::Distance<uint8_t>* make_distance<uint8_t>(diskann::Metric) {
    return new diskann::DistanceL2UInt8();
}

template<>
diskann::Distance<int8_t>* make_distance<int8_t>(diskann::Metric metric) {
    if (metric == diskann::Metric::COSINE)
        return new diskann::DistanceCosineInt8();
    return new diskann::DistanceL2Int8();
}

// Build PQ pivots and codes for `data_bin` if they don't already exist.
template<typename T>
static void ensure_pq(const std::string& data_bin,
                      const std::string& pq_prefix,
                      uint32_t pq_chunks) {
    if (pq_chunks == 0 || pq_prefix.empty()) return;
    const std::string pivots_path = pq_prefix + "_pq_pivots.bin";
    const std::string codes_path  = pq_prefix + "_pq_compressed.bin";
    if (api_path_exists(pivots_path) && api_path_exists(codes_path)) return;

    size_t npts = 0, dim = 0;
    diskann::get_bin_metadata(data_bin, npts, dim);
    double sample_rate = 0.05;
    if (npts <= 250000)      sample_rate = 1.0;
    else if (npts <= 5000000) sample_rate = 0.1;

    float* train_data = nullptr;
    size_t train_size = 0, train_dim = 0;
    gen_random_slice<T>(data_bin, sample_rate, train_data, train_size, train_dim);
    if (!train_data || train_size == 0)
        throw std::runtime_error("failed to prepare PQ training sample: " + data_bin);

    generate_pq_pivots(train_data, train_size, static_cast<unsigned>(train_dim),
                       NUM_PQ_CENTERS, pq_chunks, NUM_K_MEANS_ITERS, pivots_path);
    generate_pq_data_from_pivots<T>(data_bin, NUM_PQ_CENTERS, pq_chunks,
                                    pivots_path, codes_path);
    delete[] train_data;
}

// Persist medoid IDs and their coordinate vectors alongside the heap.
static void save_medoids(const std::string& path,
                         const std::vector<unsigned>& medoids) {
    if (medoids.empty()) return;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("cannot write medoids: " + path);
    int32_t rows = static_cast<int32_t>(medoids.size()), cols = 1;
    out.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    out.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    out.write(reinterpret_cast<const char*>(medoids.data()),
              medoids.size() * sizeof(uint32_t));
}

static std::vector<unsigned> load_medoids(const std::string& path) {
    std::vector<unsigned> medoids;
    if (!api_path_exists(path)) return medoids;
    uint32_t* raw = nullptr;
    size_t nr = 0, nc = 0;
    diskann::load_bin<uint32_t>(path, raw, nr, nc);
    medoids.assign(raw, raw + nr * nc);
    delete[] raw;
    return medoids;
}

// Save medoid centroids as float (DiskANN `_centroids.bin` format).
static void save_medoid_coords_float(const std::string& path,
                                     const std::vector<float>& coords,
                                     size_t n_medoids,
                                     uint32_t aligned_dim) {
    if (coords.empty()) return;
    diskann::save_bin<float>(path, const_cast<float*>(coords.data()), n_medoids, aligned_dim);
}

// Load medoid centroids stored as float (matches DiskANN `_centroids.bin`).
// Strict: refuses int8/uint8 legacy files — those must be regenerated.
static std::vector<float> load_medoid_coords_float(const std::string& path,
                                                   uint32_t aligned_dim,
                                                   size_t expected_rows) {
    std::vector<float> coords;
    if (!api_path_exists(path)) return coords;
    float* raw = nullptr;
    size_t rows = 0, dim = 0, rounded = 0;
    diskann::load_aligned_bin<float>(path, raw, rows, dim, rounded);
    // DiskANN `_centroids.bin` stores raw data_dim (e.g. 100), and
    // load_aligned_bin zero-pads each row to rounded = ROUND_UP(dim, 8).
    // Accept when the padded width matches our aligned_dim; the file's
    // raw dim may be smaller but never larger.
    if (rounded != aligned_dim || dim > aligned_dim) {
        diskann::aligned_free(raw);
        throw std::runtime_error("medoid coord dim mismatch: " + path);
    }
    if (expected_rows > 0 && rows != expected_rows) {
        diskann::aligned_free(raw);
        throw std::runtime_error("medoid coord row mismatch: " + path);
    }
    coords.assign(raw, raw + rows * rounded);
    diskann::aligned_free(raw);
    return coords;
}

// Read the coordinates of medoid nodes directly from the graph store, then
// upcast to float for centroid storage. Used by the in-memory build path
// when no precomputed DiskANN-style centroids file is available.
template<typename T>
static std::vector<float> collect_medoid_coords_float(InPlaceGraphStore& store,
                                                      const std::vector<unsigned>& medoids,
                                                      uint32_t aligned_dim) {
    std::vector<float> coords_float;
    if (medoids.empty()) return coords_float;
    std::vector<T> coords_t(medoids.size() * static_cast<size_t>(aligned_dim),
                            static_cast<T>(0));
    std::vector<uint8_t> found;
    store.batch_fetch_coords(medoids, reinterpret_cast<char*>(coords_t.data()),
                             found);
    for (size_t i = 0; i < medoids.size(); ++i) {
        if (!found[i])
            throw std::runtime_error("failed to read medoid coords from store");
    }
    coords_float.assign(coords_t.size(), 0.0f);
    for (size_t i = 0; i < coords_t.size(); ++i) {
        coords_float[i] = static_cast<float>(coords_t[i]);
    }
    return coords_float;
}

// Choose the medoid (entry point) by sampling: find the vector closest to the
// dataset centroid over a random stride-based sample.
template<typename T>
static uint32_t choose_entry_point(const T* base_data,
                                   uint32_t npts,
                                   uint32_t aligned_dim) {
    if (npts == 0) return 0;
    const uint32_t sample_count = std::min<uint32_t>(npts, 4096);
    const uint32_t stride       = std::max<uint32_t>(1, npts / sample_count);

    std::vector<double> centroid(aligned_dim, 0.0);
    uint32_t actual = 0;
    for (uint32_t i = 0; i < npts && actual < sample_count; i += stride, ++actual) {
        const T* v = base_data + static_cast<size_t>(i) * aligned_dim;
        for (uint32_t d = 0; d < aligned_dim; ++d)
            centroid[d] += static_cast<double>(v[d]);
    }
    if (actual == 0) return 0;
    for (double& x : centroid) x /= static_cast<double>(actual);

    double best_dist = std::numeric_limits<double>::max();
    uint32_t best_id = 0;
    actual = 0;
    for (uint32_t i = 0; i < npts && actual < sample_count; i += stride, ++actual) {
        const T* v = base_data + static_cast<size_t>(i) * aligned_dim;
        double dist = 0.0;
        for (uint32_t d = 0; d < aligned_dim; ++d) {
            double diff = static_cast<double>(v[d]) - centroid[d];
            dist += diff * diff;
        }
        if (dist < best_dist) { best_dist = dist; best_id = i; }
    }
    return best_id;
}

// Build the in-memory Vamana graph and bulk-load it into the store.
template<typename T>
static std::vector<unsigned> build_inmemory(
        const std::string& base_bin,
        uint32_t npts,
        uint32_t dim,
        uint32_t aligned_dim,
        const BufANNConfig& config,
        InPlaceGraphStore& store) {
    T* base_data = nullptr;
    size_t load_npts = 0, load_dim = 0, base_aligned_dim = 0;
    diskann::load_aligned_bin<T>(base_bin, base_data, load_npts, load_dim, base_aligned_dim);

    diskann::Index<T, uint32_t> mem_index(diskann::Metric::L2, dim,
                                          npts + 1024, false, false, false);
    diskann::Parameters params;
    params.Set<unsigned>("L",            config.L);
    params.Set<unsigned>("R",            config.R);
    params.Set<float>   ("alpha",        config.alpha);
    params.Set<unsigned>("C",            config.C);
    params.Set<unsigned>("num_threads",  config.build_threads);
    params.Set<bool>    ("saturate_graph", config.saturate_graph);
    mem_index.build(base_bin.c_str(), npts, params);

    store.bulk_load_from_index(mem_index, npts);
    const uint32_t entry = choose_entry_point(base_data, npts, aligned_dim);
    store.set_entry_point(entry);
    store.seed_entry_pool_from_reservoir();
    diskann::aligned_free(base_data);

    return {entry};
}

// Build an out-of-core merged Vamana index and stream it into the store page
// by page, respecting the given memory budget.
template<typename T>
static std::vector<unsigned> build_disk_stream(
        const std::string& base_bin,
        const std::string& temp_dir,
        uint32_t npts,
        uint32_t dim,
        uint32_t aligned_dim,
        const BufANNConfig& config,
        InPlaceGraphStore& store) {
    if (std::system(("mkdir -p " + temp_dir).c_str()) != 0)
        throw std::runtime_error("cannot create temp dir: " + temp_dir);

    const std::string prefix      = temp_dir + "/bufann_vamana";
    const std::string index_path  = prefix + "_mem.index";
    const std::string medoids_path= prefix + "_medoids.bin";
    const std::string centroids_path = prefix + "_centroids.bin";

    const size_t training_set_size = std::max<size_t>(1,
        diskann::PQ_TRAINING_SET_FRACTION * npts > diskann::MAX_PQ_TRAINING_SET_SIZE
            ? diskann::MAX_PQ_TRAINING_SET_SIZE
            : static_cast<size_t>(std::round(diskann::PQ_TRAINING_SET_FRACTION * npts)));
    const double sampling_rate =
        npts > 0 ? static_cast<double>(training_set_size) / static_cast<double>(npts) : 1.0;

    if (diskann::build_merged_vamana_index<T>(
            base_bin, diskann::Metric::L2, false,
            config.L, config.R, sampling_rate,
            config.memory_budget_gb, index_path,
            medoids_path, centroids_path, nullptr) != 0) {
        throw std::runtime_error("build_merged_vamana_index failed");
    }

    std::vector<unsigned> medoids = load_medoids(medoids_path);

    // Stream the built graph + data into the InPlaceGraphStore.
    std::ifstream graph_in(index_path, std::ios::binary);
    if (!graph_in) throw std::runtime_error("cannot open stream graph: " + index_path);
    uint64_t hdr_size = 0; uint32_t width = 0, entry = 0; uint64_t frozen_pts = 0;
    graph_in.read(reinterpret_cast<char*>(&hdr_size),   sizeof(uint64_t));
    graph_in.read(reinterpret_cast<char*>(&width),      sizeof(uint32_t));
    graph_in.read(reinterpret_cast<char*>(&entry),      sizeof(uint32_t));
    graph_in.read(reinterpret_cast<char*>(&frozen_pts), sizeof(uint64_t));
    (void)hdr_size; (void)width; (void)frozen_pts;

    std::ifstream data_in(base_bin, std::ios::binary);
    if (!data_in) throw std::runtime_error("cannot open base data: " + base_bin);
    int32_t file_npts = 0, file_dim = 0;
    data_in.read(reinterpret_cast<char*>(&file_npts), sizeof(int32_t));
    data_in.read(reinterpret_cast<char*>(&file_dim),  sizeof(int32_t));
    if (static_cast<uint32_t>(file_dim) != dim)
        throw std::runtime_error("stream conversion: dim mismatch");

    std::vector<T> packed(dim), aligned(aligned_dim, static_cast<T>(0));
    for (uint32_t i = 0; i < npts; ++i) {
        uint32_t degree = 0;
        graph_in.read(reinterpret_cast<char*>(&degree), sizeof(uint32_t));
        std::vector<uint32_t> nbrs(degree);
        if (degree > 0)
            graph_in.read(reinterpret_cast<char*>(nbrs.data()), degree * sizeof(uint32_t));
        data_in.read(reinterpret_cast<char*>(packed.data()), dim * sizeof(T));
        if (!graph_in || !data_in)
            throw std::runtime_error("unexpected EOF in stream conversion");
        std::fill(aligned.begin(), aligned.end(), static_cast<T>(0));
        std::memcpy(aligned.data(), packed.data(), dim * sizeof(T));

        store.allocate_node(i);
        NodeRID rid;
        PinnedFrame frame = store.pin_page_for_node(i, WRITE, &rid);
        if (!frame.valid())
            throw std::runtime_error("pin failed during stream conversion");
        auto guard = frame.write_guard();
        MutableNodeRef node = guard.write_node(rid);
        if (!node.valid()) {
            throw std::runtime_error("write guard failed during stream conversion");
        }
        node.set_coords<T>(aligned.data());
        node.set_neighbors(nbrs.data(), nbrs.size());
        store.publish_node(i);
    }
    store.set_entry_point(entry);
    store.seed_entry_pool_from_reservoir();
    if (medoids.empty() && store.is_active(entry)) medoids.push_back(entry);
    return medoids;
}

// Return every unique live medoid as a search start seed. Deleted medoids
// remain in the persisted list, so skip them and fall back to the store's live
// entry pool only when no persisted medoid is usable.
template<typename T>
static std::vector<unsigned> all_live_medoids(BufANNIndex<T>& idx) {
    std::vector<unsigned> init_ids;
    init_ids.reserve(idx.medoids.size());
    for (const uint32_t medoid : idx.medoids) {
        if (!idx.store.is_active(medoid) ||
            idx.store.is_tag_deleted(idx.store.node_tag(medoid))) {
            continue;
        }
        if (std::find(init_ids.begin(), init_ids.end(), medoid) == init_ids.end()) {
            init_ids.push_back(medoid);
        }
    }

    if (init_ids.empty()) {
        std::vector<uint32_t> pool;
        idx.store.get_entry_points(pool);
        for (uint32_t cand : pool) {
            if (!idx.store.is_tag_deleted(idx.store.node_tag(cand)) &&
                std::find(init_ids.begin(), init_ids.end(), cand) == init_ids.end()) {
                init_ids.push_back(cand);
            }
        }
    }
    return init_ids;
}

template<typename T>
static inline void encode_pq_if_enabled(BufANNIndex<T>& idx,
                                        uint32_t node_id,
                                        const T* coords) {
    if (idx.store.n_chunks() > 0) {
        idx.store.encode_pq(node_id, coords, idx.pq_table);
    }
}

// Allocate and zero-pad a single vector from the caller's raw `coords`
// (config.dim elements) into a buffer with `aligned_dim` elements.
template<typename T>
static std::vector<T> pad_vector(const T* coords, uint32_t dim, uint32_t aligned_dim) {
    std::vector<T> buf(aligned_dim, static_cast<T>(0));
    std::memcpy(buf.data(), coords, dim * sizeof(T));
    return buf;
}

// Common index initialization: sets up the store and loads PQ.
template<typename T>
static void init_store(BufANNIndex<T>& idx,
                       const std::string& heap_path,
                       bool truncate_heap) {
    BufANNConfig& cfg = idx.config;
    const uint32_t degree_cap = cfg.R;
    idx.aligned_dim = diskann::align_dim(cfg.dim);
    idx.dist_cmp = make_distance<T>(cfg.metric);
    idx.dist_cmp_float = make_distance<float>(cfg.metric);
    cfg.pq_chunks = resolve_existing_pq_chunks(idx.pq_prefix, cfg.pq_chunks);

    idx.store.init(cfg.dim, degree_cap, sizeof(T),
                   cfg.page_size, cfg.buffer_pool_frames,
                   heap_path,
                   cfg.flush_budget_pages_per_cycle, cfg.flush_wakeup_ms,
                   truncate_heap, cfg.max_dataset_size);

    if (cfg.pq_chunks > 0 && !idx.pq_prefix.empty()) {
        const std::string pivots_path = idx.pq_prefix + "_pq_pivots.bin";
        const std::string codes_path = idx.pq_prefix + "_pq_compressed.bin";
        if (api_path_exists(pivots_path) && api_path_exists(codes_path)) {
            idx.pq_table.load_pq_centroid_bin(pivots_path.c_str(), cfg.pq_chunks);
            idx.store.load_pq_codes_from_disk_index(idx.pq_prefix, cfg.pq_chunks);
        } else {
            diskann::cout << "InPlace: PQ files not found at prefix " << idx.pq_prefix
                          << ", skipping PQ load." << std::endl;
        }
    }
}

} // anonymous namespace

uint32_t diskann::inplace::bufann_snapshot_active_cap(
        const std::string& index_prefix) {
    struct SnapshotHeaderPrefix {
        char magic[8];
        uint32_t version;
        uint32_t dim;
        uint32_t aligned_dim;
        uint32_t Mmax;
        uint32_t elem_size;
        uint32_t page_size;
        uint32_t slot_size;
        uint32_t slots_per_page;
        uint32_t _reserved;
        uint32_t max_nodes;
        uint32_t n_chunks;
        uint32_t num_active;
        uint32_t total_pages;
        uint64_t rid_size;
        uint64_t active_cap;
    } hdr{};
    const std::string meta_path = index_prefix + ".meta";
    std::ifstream in(meta_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read snapshot metadata: " + meta_path);
    in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
    const bool ok_magic =
        memcmp(hdr.magic, "IPGSNP4", 8) == 0 ||
        memcmp(hdr.magic, "IPGSNP3", 8) == 0 ||
        memcmp(hdr.magic, "IPGSNP2", 8) == 0;
    if (!ok_magic || hdr.active_cap > std::numeric_limits<uint32_t>::max()) {
        throw std::runtime_error("invalid snapshot metadata: " + meta_path);
    }
    return static_cast<uint32_t>(hdr.active_cap);
}

// ---------------------------------------------------------------------------
// bufann_build
// ---------------------------------------------------------------------------

template<typename T>
BufANNIndex<T>* diskann::inplace::bufann_build(
        const std::string& data_bin,
        const std::string& index_prefix,
        const BufANNConfig& config,
        const std::string& build_temp_dir) {

    if (config.dim == 0)
        throw std::invalid_argument("bufann_build: config.dim must be non-zero");
    if (config.index_type == IndexType::IvfPq) {
        auto* idx = new BufANNIndex<T>();
        idx->config = config;
        try {
            idx->ivf = ivf_pq_backend_build<T>(data_bin, index_prefix, config);
        } catch (...) {
            delete idx;
            throw;
        }
        return idx;
    }

    size_t npts = 0, raw_dim = 0;
    diskann::get_bin_metadata(data_bin, npts, raw_dim);
    if (raw_dim != config.dim)
        throw std::runtime_error("bufann_build: data dim " +
                                 std::to_string(raw_dim) +
                                 " != config.dim " +
                                 std::to_string(config.dim));
    if (npts > std::numeric_limits<uint32_t>::max())
        throw std::runtime_error("bufann_build: data size exceeds uint32_t node ID range");

    auto* idx = new BufANNIndex<T>();
    idx->config     = config;
    if (idx->config.max_dataset_size == 0) {
        idx->config.max_dataset_size = static_cast<uint32_t>(npts);
    }
    if (idx->config.max_dataset_size < npts) {
        const uint32_t configured_max = idx->config.max_dataset_size;
        delete idx;
        throw std::runtime_error("bufann_build: max_dataset_size " +
                                 std::to_string(configured_max) +
                                 " < data size " + std::to_string(npts));
    }
    idx->pq_prefix  = index_prefix;

    const std::string heap_path        = index_prefix + ".heap";
    const std::string heap_meta_path   = index_prefix + ".meta";
    const std::string medoids_path     = index_prefix + ".medoids";
    const std::string medoid_coords_path = index_prefix + ".centroids";

    try {
        init_store<T>(*idx, heap_path, /*truncate_heap=*/true);

        // Build PQ tables if requested (must happen before building the graph
        // so codes can be encoded as nodes are streamed in).
        if (config.pq_chunks > 0)
            ensure_pq<T>(data_bin, index_prefix, config.pq_chunks);

        if (config.memory_budget_gb <= 0.0f) {
            // In-memory build: load full dataset, build Vamana, bulk-load.
            idx->medoids = build_inmemory<T>(
                data_bin, static_cast<uint32_t>(npts),
                static_cast<uint32_t>(raw_dim), idx->aligned_dim,
                config, idx->store);
        } else {
            // Out-of-core build: stream the Vamana graph into the store page by page.
            const std::string temp_dir =
                build_temp_dir.empty() ? (index_prefix + "_build_tmp") : build_temp_dir;
            idx->medoids = build_disk_stream<T>(
                data_bin, temp_dir,
                static_cast<uint32_t>(npts), static_cast<uint32_t>(raw_dim),
                idx->aligned_dim, config, idx->store);
        }

        idx->medoid_coords = collect_medoid_coords_float<T>(
            idx->store, idx->medoids, idx->aligned_dim);

        idx->store.flush();
        idx->store.save_snapshot(heap_meta_path);
        save_medoids(medoids_path, idx->medoids);
        save_medoid_coords_float(medoid_coords_path, idx->medoid_coords,
                                 idx->medoids.size(), idx->aligned_dim);

    } catch (...) {
        delete idx;
        throw;
    }

    return idx;
}

// ---------------------------------------------------------------------------
// bufann_load
// ---------------------------------------------------------------------------

template<typename T>
BufANNIndex<T>* diskann::inplace::bufann_load(
        const std::string& index_prefix,
        const BufANNConfig& config) {

    if (config.dim == 0)
        throw std::invalid_argument("bufann_load: config.dim must be non-zero");
    if (config.index_type == IndexType::IvfPq) {
        auto* idx = new BufANNIndex<T>();
        idx->config = config;
        try {
            idx->ivf = ivf_pq_backend_load<T>(index_prefix, config);
        } catch (...) {
            delete idx;
            throw;
        }
        return idx;
    }

    const std::string heap_path          = index_prefix + ".heap";
    const std::string heap_meta_path     = index_prefix + ".meta";
    const std::string medoids_path       = index_prefix + ".medoids";
    const std::string medoid_coords_path = index_prefix + ".centroids";

    if (!api_path_exists(heap_meta_path))
        throw std::runtime_error("bufann_load: index not found at " + heap_meta_path);

    auto* idx = new BufANNIndex<T>();
    idx->config    = config;
    if (idx->config.max_dataset_size == 0) {
        idx->config.max_dataset_size = bufann_snapshot_active_cap(index_prefix);
    } else {
        const uint32_t active_cap = bufann_snapshot_active_cap(index_prefix);
        if (idx->config.max_dataset_size < active_cap) {
            const uint32_t configured_max = idx->config.max_dataset_size;
            delete idx;
            throw std::runtime_error("bufann_load: max_dataset_size " +
                                     std::to_string(configured_max) +
                                     " < snapshot active_cap " +
                                     std::to_string(active_cap));
        }
    }
    idx->pq_prefix = index_prefix;

    try {
        init_store<T>(*idx, heap_path, /*truncate_heap=*/false);

        idx->store.load_snapshot(heap_meta_path);

        idx->medoids = load_medoids(medoids_path);
        if (idx->medoids.empty())
            throw std::runtime_error("bufann_load: medoids file missing or empty");

        idx->medoid_coords = load_medoid_coords_float(
            medoid_coords_path, idx->aligned_dim, idx->medoids.size());
        // DiskANN only emits `_disk.index_centroids.bin` for multi-shard
        // builds, so small single-shard indexes legitimately have no
        // .centroids file. Retain the existing format validation for
        // multi-medoid indexes even though search now seeds all live medoids.
        if (idx->medoid_coords.empty() && idx->medoids.size() > 1)
            throw std::runtime_error("bufann_load: medoid coords file missing or empty");

    } catch (...) {
        delete idx;
        throw;
    }

    return idx;
}

// ---------------------------------------------------------------------------
// bufann_insert
// ---------------------------------------------------------------------------

template<typename T>
void diskann::inplace::bufann_insert(
        BufANNIndex<T>& idx,
        TagType tag,
        const T* coords,
        uint32_t search_L) {
    if (idx.ivf) {
        ivf_pq_backend_insert<T>(*idx.ivf, idx.config, tag, coords);  // search_L is a graph parameter
        return;
    }
    if (tag == INVALID_TAG) {
        throw std::invalid_argument("bufann_insert: INVALID_TAG is reserved");
    }
    if (idx.store.is_tag_deleted(tag)) {
        throw std::runtime_error("bufann_insert: tag is deleted " +
                                 std::to_string(tag));
    }
    uint32_t node_id;  // reserved + materialized inside the BEAM_SEARCH scope below

    const uint32_t L = (search_L == 0) ? idx.config.L : search_L;
    const uint32_t aligned_dim = idx.aligned_dim;
    const BufANNConfig& cfg = idx.config;

    const T* vec = coords;
    std::vector<T> aligned_coords;
    if (cfg.dim != aligned_dim) {
        aligned_coords = pad_vector(coords, cfg.dim, aligned_dim);
        vec = aligned_coords.data();
    }

    InPlaceSearchScratch& scratch =
        query_scratch_for<T>(aligned_dim, idx.store.n_chunks(), sizeof(T),
                             L, idx.store.max_degree(), cfg.beamwidth);
    scratch.reset();

    // 1. Reserve a page slot for the new node, then beam-search for candidates.
    auto& candidates = scratch.traversal_results;
    {
        InsertIOPhaseScope sc(InsertIOPhaseScope::BEAM_SEARCH);
        node_id = idx.store.allocate_node();
        if (!idx.store.set_node_tag(node_id, tag)) {
            throw std::runtime_error("bufann_insert: failed to register tag " +
                                     std::to_string(tag));
        }
        std::vector<unsigned> init_ids = all_live_medoids<T>(idx);
        graph_iterate_to_fixed_point<T>(
            vec, L, init_ids, cfg.beamwidth, &idx.store, aligned_dim,
            idx.dist_cmp, &scratch, candidates,
            /*use_deferred_overlay=*/false,
            TraversalScope::INSERT, &idx.pq_table);
    }

    // 2. Prune candidates to at most R neighbors using RNG occlusion.
    std::vector<unsigned> pruned;
    {
        InsertIOPhaseScope sc(InsertIOPhaseScope::PRUNE);
        graph_prune_neighbors_pq<T>(
            node_id, candidates, cfg.R, cfg.C, cfg.alpha,
            pruned, &idx.store, &scratch,
            /*aligned_dim=*/0, /*distance=*/nullptr,
            DistanceScope::INSERT, &idx.pq_table);
    }

    // 3. Write the new node's coordinates, forward edges, PQ code, then publish.
    {
        InsertIOPhaseScope sc(InsertIOPhaseScope::WRITE_NEW_NODE);
        NodeRID rid;
        PinnedFrame frame = idx.store.pin_page_for_node(node_id, WRITE, &rid);
        if (!frame.valid())
            throw std::runtime_error("bufann_insert: failed to pin node " +
                                     std::to_string(node_id));
        {
            auto guard = frame.write_guard();
            MutableNodeRef node = guard.write_node(rid);
            if (!node.valid()) {
                throw std::runtime_error("bufann_insert: failed to access pinned node " +
                                         std::to_string(node_id));
            }
            node.set_coords<T>(vec);
            node.set_neighbors(pruned.data(), std::min<size_t>(cfg.R, pruned.size()));
        }
        encode_pq_if_enabled<T>(idx, node_id, vec);
        idx.store.publish_node(node_id);
    }

    // 4. Immediate reverse-edge repair without the deferred-edge queue.
    {
        InsertIOPhaseScope sc(InsertIOPhaseScope::BIDIRECTIONAL);
        graph_inter_insert_immediate<T>(
            node_id, pruned, &idx.store, cfg.R, cfg.C, cfg.alpha,
            idx.aligned_dim, idx.dist_cmp, &idx.pq_table);
    }

    insert_stats_increment_n();
}

// ---------------------------------------------------------------------------
// bufann_delete
// ---------------------------------------------------------------------------

template<typename T>
void diskann::inplace::bufann_delete(
        BufANNIndex<T>& idx,
        TagType tag) {
    if (idx.ivf) {
        ivf_pq_backend_delete(*idx.ivf, tag);
        return;
    }
    idx.store.mark_tag_deleted(tag);
}

// ---------------------------------------------------------------------------
// bufann_cleanup_deleted_edges
// ---------------------------------------------------------------------------

template<typename T>
void diskann::inplace::bufann_delete_batch(
        BufANNIndex<T>& idx,
        const TagType* tags,
        size_t count) {
    if (idx.ivf) {
        for (size_t i = 0; i < count; ++i) ivf_pq_backend_delete(*idx.ivf, tags[i]);
        return;
    }
    if (count == 0) return;
    for (size_t i = 0; i < count; ++i) {
        idx.store.mark_tag_deleted(tags[i]);
    }
}

template<typename T>
void diskann::inplace::bufann_cleanup_deleted_edges(BufANNIndex<T>& idx,
                                                    uint32_t num_threads,
                                                    uint32_t delete_micro_batch) {
    if (idx.ivf)
        throw std::runtime_error("bufann_cleanup_deleted_edges: not supported for an IVF-PQ index yet");
    const auto maintenance_begin = std::chrono::steady_clock::now();
    auto elapsed_s = [](const std::chrono::steady_clock::time_point& begin) {
        return std::chrono::duration<double>(
                   std::chrono::steady_clock::now() - begin).count();
    };

    std::cout << "[BufANN] cleanup_deleted_edges(): started maintenance"
              << " threads=" << num_threads
              << " pending_deleted_tags=" << idx.store.deleted_tag_count()
              << std::endl;

    std::vector<uint32_t> batch = idx.store.drain_deleted_tags_to_delete_pending();

    const uint32_t repair_threads = std::max<uint32_t>(1u, num_threads);
    const uint32_t micro = std::max<uint32_t>(1u, delete_micro_batch);
    const size_t repair_chunk_count =
        (batch.size() + static_cast<size_t>(micro) - 1) /
        static_cast<size_t>(micro);
    const auto repair_begin = std::chrono::steady_clock::now();
    std::cout << "[BufANN] cleanup_deleted_edges(): started repair"
              << " batch_size=" << batch.size()
              << " micro_batch=" << micro
              << " repair_threads=" << repair_threads
              << " chunks=" << repair_chunk_count << std::endl;
    uint32_t repaired_edges = 0;
    if (!batch.empty()) {
        const BufANNConfig& cfg = idx.config;
        const uint64_t repair_total = static_cast<uint64_t>(batch.size());
        const uint64_t progress_step = std::max<uint64_t>(1, repair_total / 20);
        std::mutex progress_mu;
        auto log_repair_progress = [&](uint64_t begin, uint64_t end) {
            auto print_progress = [&](uint64_t target) {
                const double pct =
                    (100.0 * static_cast<double>(target)) /
                    static_cast<double>(repair_total);
                std::lock_guard<std::mutex> lk(progress_mu);
                std::cout << "Repair progress: " << target << "/"
                          << repair_total << " (" << std::fixed
                          << std::setprecision(1) << pct << "%, elapsed "
                          << std::setprecision(2) << elapsed_s(repair_begin)
                          << "s)" << std::endl;
            };
            for (uint64_t target = progress_step; target < repair_total;
                 target += progress_step) {
                if (begin >= target || target > end) continue;
                print_progress(target);
            }
            if (begin < repair_total && repair_total <= end) {
                print_progress(repair_total);
            }
        };
#pragma omp parallel for num_threads(static_cast<int>(repair_threads)) \
    schedule(dynamic, 1) reduction(+:repaired_edges)
        for (int64_t chunk = 0;
             chunk < static_cast<int64_t>(repair_chunk_count); ++chunk) {
            const size_t begin =
                static_cast<size_t>(chunk) * static_cast<size_t>(micro);
            const size_t end =
                std::min(batch.size(), begin + static_cast<size_t>(micro));
            std::vector<uint32_t> repair_batch(batch.begin() + begin,
                                               batch.begin() + end);
            repaired_edges += idx.store.template repair_deleted_explicit<T>(
                repair_batch, cfg.R, cfg.C, cfg.alpha, idx.aligned_dim,
                idx.dist_cmp, cfg.c_replace, cfg.delete_repair_L,
                cfg.beamwidth, &idx.pq_table);
            log_repair_progress(static_cast<uint64_t>(begin),
                                static_cast<uint64_t>(end));
        }
    }
    std::cout << "[BufANN] cleanup_deleted_edges(): finished repair"
              << " elapsed_s=" << elapsed_s(repair_begin)
              << " batch_size=" << batch.size()
              << " micro_batch=" << micro
              << " chunks=" << repair_chunk_count
              << " repaired_edges=" << repaired_edges << std::endl;

    const auto edge_removal_begin = std::chrono::steady_clock::now();
    std::cout << "[BufANN] cleanup_deleted_edges(): started dead edge removal"
              << " threads=" << num_threads << std::endl;
    const uint32_t removed_edges =
        idx.store.remove_deleted_edges_full_scan(num_threads);
    std::cout << "[BufANN] cleanup_deleted_edges(): finished dead edge removal"
              << " elapsed_s=" << elapsed_s(edge_removal_begin)
              << " removed_edges=" << removed_edges << std::endl;

    std::cout << "[BufANN] cleanup_deleted_edges(): finished maintenance"
              << " total_elapsed_s=" << elapsed_s(maintenance_begin)
              << std::endl;
}

// ---------------------------------------------------------------------------
// bufann_flush_dirty
// ---------------------------------------------------------------------------

template<typename T>
void diskann::inplace::bufann_flush_dirty(BufANNIndex<T>& idx) {
    if (idx.ivf) return;
    idx.store.flush();
}

template<typename T>
uint32_t diskann::inplace::bufann_flush_dirty_budget(BufANNIndex<T>& idx,
                                                      uint32_t max_pages) {
    if (idx.ivf) return 0;
    return idx.store.flush_dirty_budget(max_pages);
}

// ---------------------------------------------------------------------------
// bufann_query
// ---------------------------------------------------------------------------

template<typename T>
uint32_t diskann::inplace::bufann_query_into(
        BufANNIndex<T>& idx,
        const T* query_vec,
        uint32_t topK,
        TagType* out_tags,
        uint32_t search_L) {

    if (idx.ivf) {
        thread_local IVFPQSearchScratch ivf_scratch;
        return ivf_pq_backend_query<T>(*idx.ivf, idx.config, query_vec, topK, out_tags, search_L, 0, ivf_scratch);
    }

    const uint32_t L = (search_L == 0) ? idx.config.L : search_L;
    const uint32_t effective_L = std::max(L, topK);
    const uint32_t aligned_dim = idx.aligned_dim;
    const BufANNConfig& cfg = idx.config;

    const T* q = query_vec;
    std::vector<T> aligned_query;
    if (cfg.dim != aligned_dim) {
        // Only allocate a padded query buffer when the SIMD-aligned dim
        // exceeds the logical dim.
        aligned_query = pad_vector(query_vec, cfg.dim, aligned_dim);
        q = aligned_query.data();
    }

    InPlaceSearchScratch& scratch =
        query_scratch_for<T>(aligned_dim, idx.store.n_chunks(), sizeof(T),
                             effective_L, idx.store.max_degree(), cfg.beamwidth);
    scratch.reset();

    auto& results = scratch.traversal_results;
    std::vector<unsigned> init_ids = all_live_medoids<T>(idx);

    graph_iterate_to_fixed_point<T>(
        q, effective_L, init_ids, cfg.beamwidth, &idx.store, aligned_dim,
        idx.dist_cmp, &scratch, results,
        /*use_deferred_overlay=*/false,
        TraversalScope::SEARCH, &idx.pq_table);
    scratch.flush_distance_stats(idx.store.stats());

    // graph_iterate_to_fixed_point returns results sorted by distance.
    const uint32_t n = std::min<uint32_t>(topK, static_cast<uint32_t>(results.size()));
    if (out_tags != nullptr) {
        for (uint32_t i = 0; i < n; ++i) {
            out_tags[i] = idx.store.node_tag(results[i].id);
        }
    }
    return n;
}

template<typename T>
std::vector<TagType> diskann::inplace::bufann_query(
        BufANNIndex<T>& idx,
        const T* query_vec,
        uint32_t topK,
        uint32_t search_L) {

    std::vector<TagType> out(topK);
    const uint32_t n = bufann_query_into(
        idx, query_vec, topK, out.data(), search_L);
    out.resize(n);
    return out;
}

// ---------------------------------------------------------------------------
// bufann_cleanup
// ---------------------------------------------------------------------------

template<typename T>
void diskann::inplace::bufann_cleanup(BufANNIndex<T>& idx) {
    bufann_cleanup_deleted_edges<T>(idx);
    bufann_flush_dirty<T>(idx);
}

// ---------------------------------------------------------------------------
// bufann_free
// ---------------------------------------------------------------------------

template<typename T>
void diskann::inplace::bufann_free(BufANNIndex<T>*& idx) {
    delete idx;
    idx = nullptr;
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

template BufANNIndex<float>*
diskann::inplace::bufann_build<float>(
    const std::string&, const std::string&, const BufANNConfig&, const std::string&);
template BufANNIndex<uint8_t>*
diskann::inplace::bufann_build<uint8_t>(
    const std::string&, const std::string&, const BufANNConfig&, const std::string&);
template BufANNIndex<int8_t>*
diskann::inplace::bufann_build<int8_t>(
    const std::string&, const std::string&, const BufANNConfig&, const std::string&);

template BufANNIndex<float>*
diskann::inplace::bufann_load<float>(const std::string&, const BufANNConfig&);
template BufANNIndex<uint8_t>*
diskann::inplace::bufann_load<uint8_t>(const std::string&, const BufANNConfig&);
template BufANNIndex<int8_t>*
diskann::inplace::bufann_load<int8_t>(const std::string&, const BufANNConfig&);

template void diskann::inplace::bufann_insert<float>(
    BufANNIndex<float>&, uint32_t, const float*, uint32_t);
template void diskann::inplace::bufann_insert<uint8_t>(
    BufANNIndex<uint8_t>&, uint32_t, const uint8_t*, uint32_t);
template void diskann::inplace::bufann_insert<int8_t>(
    BufANNIndex<int8_t>&, uint32_t, const int8_t*, uint32_t);

template void diskann::inplace::bufann_delete<float>(
    BufANNIndex<float>&, uint32_t);
template void diskann::inplace::bufann_delete<uint8_t>(
    BufANNIndex<uint8_t>&, uint32_t);
template void diskann::inplace::bufann_delete<int8_t>(
    BufANNIndex<int8_t>&, uint32_t);

template void diskann::inplace::bufann_delete_batch<float>(
    BufANNIndex<float>&, const uint32_t*, size_t);
template void diskann::inplace::bufann_delete_batch<uint8_t>(
    BufANNIndex<uint8_t>&, const uint32_t*, size_t);
template void diskann::inplace::bufann_delete_batch<int8_t>(
    BufANNIndex<int8_t>&, const uint32_t*, size_t);

template void diskann::inplace::bufann_cleanup_deleted_edges<float>(
    BufANNIndex<float>&, uint32_t, uint32_t);
template void diskann::inplace::bufann_cleanup_deleted_edges<uint8_t>(
    BufANNIndex<uint8_t>&, uint32_t, uint32_t);
template void diskann::inplace::bufann_cleanup_deleted_edges<int8_t>(
    BufANNIndex<int8_t>&, uint32_t, uint32_t);

template void diskann::inplace::bufann_flush_dirty<float>(
    BufANNIndex<float>&);
template void diskann::inplace::bufann_flush_dirty<uint8_t>(
    BufANNIndex<uint8_t>&);
template void diskann::inplace::bufann_flush_dirty<int8_t>(
    BufANNIndex<int8_t>&);

template uint32_t diskann::inplace::bufann_flush_dirty_budget<float>(
    BufANNIndex<float>&, uint32_t);
template uint32_t diskann::inplace::bufann_flush_dirty_budget<uint8_t>(
    BufANNIndex<uint8_t>&, uint32_t);
template uint32_t diskann::inplace::bufann_flush_dirty_budget<int8_t>(
    BufANNIndex<int8_t>&, uint32_t);

template uint32_t diskann::inplace::bufann_query_into<float>(
    BufANNIndex<float>&, const float*, uint32_t, uint32_t*, uint32_t);
template uint32_t diskann::inplace::bufann_query_into<uint8_t>(
    BufANNIndex<uint8_t>&, const uint8_t*, uint32_t, uint32_t*, uint32_t);
template uint32_t diskann::inplace::bufann_query_into<int8_t>(
    BufANNIndex<int8_t>&, const int8_t*, uint32_t, uint32_t*, uint32_t);

template std::vector<uint32_t> diskann::inplace::bufann_query<float>(
    BufANNIndex<float>&, const float*, uint32_t, uint32_t);
template std::vector<uint32_t> diskann::inplace::bufann_query<uint8_t>(
    BufANNIndex<uint8_t>&, const uint8_t*, uint32_t, uint32_t);
template std::vector<uint32_t> diskann::inplace::bufann_query<int8_t>(
    BufANNIndex<int8_t>&, const int8_t*, uint32_t, uint32_t);

template void diskann::inplace::bufann_cleanup<float>(BufANNIndex<float>&);
template void diskann::inplace::bufann_cleanup<uint8_t>(BufANNIndex<uint8_t>&);
template void diskann::inplace::bufann_cleanup<int8_t>(BufANNIndex<int8_t>&);

template void diskann::inplace::bufann_free<float>(BufANNIndex<float>*&);
template void diskann::inplace::bufann_free<uint8_t>(BufANNIndex<uint8_t>*&);
template void diskann::inplace::bufann_free<int8_t>(BufANNIndex<int8_t>*&);
