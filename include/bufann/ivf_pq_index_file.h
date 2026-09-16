// The combined IVF-PQ index file (build step 6): one header followed by
// every persisted section, so a load is a single consistent snapshot.
//
//   [IVFPQIndexFileHeader]
//   [centroids       f32  nlist x aligned_dim]
//   [cluster ids     u32  N]
//   [posting offsets u32  nlist + 1]
//   [posting ids     u32  N]
//   [pq pivots       f32  chunks x k x chunk_dim]
//   [pq codes        u8   N x chunks]
//   [rid table       u32  N]
//
// The raw vectors stay in the heap file next to it; the header records the
// heap's geometry and size. The file is written to a temporary name and
// renamed into place, so a crash leaves the previous file or none.

#pragma once

#include <string>

#include "bufann/ivf_pq.h"

namespace diskann {
namespace inplace {

std::string ivf_pq_index_path(const std::string& index_prefix);

void write_ivf_pq_index(const std::string& index_prefix, const IVFPQIndex& index);

// Validates the header, every section's size against the header's shape,
// and the cross-section invariants (posting lists are the exact inverse of
// the cluster ids, RIDs address allocated heap slots, the heap file has the
// recorded size). Throws on any inconsistency.
IVFPQIndex load_ivf_pq_index(const std::string& index_prefix);

}  // namespace inplace
}  // namespace diskann
