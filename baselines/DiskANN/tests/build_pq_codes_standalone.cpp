// Standalone PQ code builder: consumes an existing <prefix>_pq_pivots.bin
// and writes <prefix>_pq_compressed.bin for the first npts_override vectors.
// This permits training a shared codebook on a full dataset while encoding
// only the prefix represented by a canonical graph.
//
// Usage:
//   build_pq_codes_standalone <type=float|int8|uint8>
//                             <base.bin> <index_prefix> <num_pq_chunks>
//                             <npts_override>

// npts_override=0 encodes the full input for consistency with the underlying
// DiskANN API. A positive value must not exceed the input point count.

// Outputs:
//   <index_prefix>_pq_compressed.bin

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

#include "partition_and_pq.h"
#include "pq_flash_index.h"  // MAX_PQ_CHUNKS
#include "utils.h"

template<typename T>
int run(const std::string &base_file, const std::string &prefix,
        uint32_t num_pq_chunks, size_t npts_override) {
  size_t npts, dim;
  diskann::get_bin_metadata(base_file, npts, dim);
  if (npts_override > npts) {
    diskann::cerr << "npts_override=" << npts_override
                  << " exceeds input point count " << npts << std::endl;
    return 1;
  }

  num_pq_chunks =
      std::min(num_pq_chunks, static_cast<uint32_t>(MAX_PQ_CHUNKS));
  const size_t effective_npts = npts_override == 0 ? npts : npts_override;
  const std::string pq_pivots_path = prefix + "_pq_pivots.bin";
  const std::string pq_compressed_path = prefix + "_pq_compressed.bin";

  diskann::cout << "build_pq_codes_standalone: base=" << base_file
                << " prefix=" << prefix << " file_npts=" << npts
                << " effective_npts=" << effective_npts << " dim=" << dim
                << " num_pq_chunks=" << num_pq_chunks << std::endl;

  int rc = ::generate_pq_data_from_pivots<T>(
      base_file, 256, num_pq_chunks, pq_pivots_path, pq_compressed_path,
      /*offset=*/0, npts_override);
  if (rc == 0)
    diskann::cout << "Wrote " << pq_compressed_path << std::endl;
  return rc;
}

int main(int argc, char **argv) {
  if (argc != 6) {
    diskann::cout << "Usage: " << argv[0]
                  << " <type=float|int8|uint8>"
                     " <base.bin> <index_prefix> <num_pq_chunks>"
                     " <npts_override>"
                  << std::endl;
    return 1;
  }

  std::string type = argv[1];
  std::string base_file = argv[2];
  std::string prefix = argv[3];
  uint32_t num_pq_chunks = static_cast<uint32_t>(std::stoul(argv[4]));
  size_t npts_override = static_cast<size_t>(std::stoull(argv[5]));

  if (type == "float")
    return run<float>(base_file, prefix, num_pq_chunks, npts_override);
  if (type == "int8")
    return run<int8_t>(base_file, prefix, num_pq_chunks, npts_override);
  if (type == "uint8")
    return run<uint8_t>(base_file, prefix, num_pq_chunks, npts_override);
  diskann::cerr << "Unknown type: " << type << std::endl;
  return 1;
}
