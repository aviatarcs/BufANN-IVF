// Proximity graph over the IVF centroids, so a query finds its nearest
// partitions by visiting a few hundred centroids rather than computing its
// distance to all nlist of them (a GEMV that reads nlist x aligned_dim
// floats: 32 MB per query at nlist 65536 and dim 128). Vamana: two passes of
// greedy search and robust prune over every centroid, the second with
// alpha > 1 to keep long edges. Built from the centroids after load, in a
// few seconds at nlist 65536; not persisted.

#pragma once

#include <cstdint>
#include <vector>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

constexpr uint32_t IVF_CENTROID_GRAPH_NONE = 0xFFFFFFFFu;

struct IVFCentroidGraphParams {
    uint32_t degree  = 32;    // R, the maximum out-degree
    uint32_t build_L = 100;   // beam width of the build's searches
    float    alpha   = 1.2f;  // prune slack of the second pass (squared distances)
};

struct IVFCentroidGraph {
    uint32_t nlist  = 0;
    uint32_t degree = 0;
    uint32_t entry  = 0;              // the centroid nearest the centroids' mean
    std::vector<uint32_t> neighbors;  // [nlist x degree]; a list ends at its first IVF_CENTROID_GRAPH_NONE
};

// Buffers one thread reuses across searches of any graph; not shared
// between threads.
struct IVFCentroidGraphScratch {
    struct Candidate {
        float dist;
        uint32_t id;
        bool expanded;
    };
    std::vector<uint32_t> visited_epoch;  // [nlist]; == epoch when visited by the current search
    uint32_t epoch = 0;
    std::vector<Candidate> frontier;      // min-heap by dist: visited, not yet expanded
    std::vector<Candidate> best;          // max-heap by dist: the L nearest visited so far
    std::vector<uint32_t> nbrs;           // one centroid's out-edges
};

// The build is parallel (OpenMP), so which of two equally good edges a node
// keeps may differ between runs; the graph's search quality does not.
IVFCentroidGraph build_ivf_centroid_graph(const IVFMetadata& meta, const IVFCentroidGraphParams& params = {});

// Leaves in `out` the `n` centroids nearest `query` (meta.dim floats) among
// those a best-first search with a beam of `L` >= n visits, ascending by
// exact squared distance. With L >= nlist it visits every centroid reachable
// from the entry, which for a built graph is all of them. `start` other than
// IVF_CENTROID_GRAPH_NONE begins the search there instead of at graph.entry:
// from a point's previous centroid, a narrow beam suffices.
void search_ivf_centroid_graph(const IVFMetadata& meta, const IVFCentroidGraph& graph, const float* query,
                               uint32_t L, uint32_t n, IVFCentroidGraphScratch& scratch,
                               std::vector<uint32_t>& out, uint32_t start = IVF_CENTROID_GRAPH_NONE);

}  // namespace inplace
}  // namespace diskann
