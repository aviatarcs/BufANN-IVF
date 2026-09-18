#include "bufann/ivf_pq_index_file.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <vector>

#include "bufann/ivf_pq_build.h"
#include "bufann/ivf_pq_require.h"
#include "utils.h"

namespace diskann {
namespace inplace {

std::string ivf_pq_index_path(const std::string& index_prefix) {
    return index_prefix + "_ivf_pq_index.bin";
}

namespace {

template<typename T>
uint64_t bytes_of(const std::vector<T>& v) {
    return uint64_t(v.size()) * sizeof(T);
}

// Lays the sections out back to back after the header.
IVFPQIndexFileHeader make_header(const IVFPQIndex& index) {
    IVFPQIndexFileHeader h;
    h.magic = IVF_PQ_INDEX_MAGIC;
    h.version = IVF_PQ_INDEX_VERSION;
    h.nlist = index.meta.nlist;
    h.dim = index.meta.dim;
    h.aligned_dim = index.meta.aligned_dim;
    h.pq_chunks = index.pq.chunks;
    h.pq_chunk_dim = index.pq.chunk_dim;
    h.pq_k = index.pq.k;
    h.num_vectors = index.assignments.cluster_id.size();
    h.raw_vector_elem_size = index.heap_layout.elem_size;
    h.raw_vector_page_size = index.heap_layout.page_size;
    h.raw_vectors_bytes = uint64_t(index.heap_pages) * index.heap_layout.page_size;
    h.raw_vector_next_slot = index.heap_next_slot;

    uint64_t next = sizeof(IVFPQIndexFileHeader);
    auto place = [&](uint64_t& offset, uint64_t& bytes, uint64_t n) {
        offset = next;
        bytes = n;
        next += n;
    };
    place(h.centroids_offset, h.centroids_bytes, bytes_of(index.meta.centroids));
    place(h.cluster_assignments_offset, h.cluster_assignments_bytes, bytes_of(index.assignments.cluster_id));
    place(h.posting_offsets_offset, h.posting_offsets_bytes, bytes_of(index.lists.offsets));
    place(h.posting_ids_offset, h.posting_ids_bytes, bytes_of(index.lists.ids));
    place(h.pq_pivots_offset, h.pq_pivots_bytes, bytes_of(index.pq.pivots));
    place(h.pq_codes_offset, h.pq_codes_bytes, bytes_of(index.pq.codes));
    place(h.rid_table_offset, h.rid_table_bytes, bytes_of(index.rid_table.rid));
    return h;
}

class FileWriter {
public:
    explicit FileWriter(const std::string& path) : _path(path) {
        _fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        IVF_PQ_REQUIRE(_fd >= 0, "Failed to create " + path);
    }
    ~FileWriter() {
        if (_fd >= 0) ::close(_fd);
    }
    template<typename T>
    void write(const std::vector<T>& v) { write(v.data(), bytes_of(v)); }
    void write(const void* data, uint64_t bytes) {
        const char* p = static_cast<const char*>(data);
        while (bytes > 0) {
            ssize_t n = ::write(_fd, p, bytes);
            IVF_PQ_REQUIRE(n > 0, "Failed to write " + _path);
            p += n, bytes -= n;
        }
    }
    void sync_and_close() {
        IVF_PQ_REQUIRE(::fsync(_fd) == 0 && ::close(_fd) == 0, "Failed to sync " + _path);
        _fd = -1;
    }

private:
    std::string _path;
    int _fd = -1;
};

class FileReader {
public:
    explicit FileReader(const std::string& path) : _path(path) {
        _fd = ::open(path.c_str(), O_RDONLY);
        IVF_PQ_REQUIRE(_fd >= 0, "Failed to open " + path);
        struct stat st;
        IVF_PQ_REQUIRE(::fstat(_fd, &st) == 0, "Failed to stat " + path);
        _size = uint64_t(st.st_size);
    }
    ~FileReader() { ::close(_fd); }
    uint64_t size() const { return _size; }

    template<typename T>
    std::vector<T> read(uint64_t offset, uint64_t bytes, const char* what) {
        IVF_PQ_REQUIRE(bytes % sizeof(T) == 0 && offset <= _size && bytes <= _size - offset,
                       std::string(what) + " section does not lie within " + _path);
        std::vector<T> v(bytes / sizeof(T));
        char* p = reinterpret_cast<char*>(v.data());
        while (bytes > 0) {
            ssize_t n = ::pread(_fd, p, bytes, off_t(offset));
            IVF_PQ_REQUIRE(n > 0, "Failed to read " + _path);
            p += n, offset += n, bytes -= n;
        }
        return v;
    }

private:
    std::string _path;
    int _fd = -1;
    uint64_t _size = 0;
};

// Every section's byte count must be what the header's shape implies.
void require_section_sizes(const IVFPQIndexFileHeader& h) {
    const uint64_t n = h.num_vectors;
    IVF_PQ_REQUIRE(h.centroids_bytes == uint64_t(h.nlist) * h.aligned_dim * sizeof(float),
                   "centroid section size does not match nlist x aligned_dim");
    IVF_PQ_REQUIRE(h.cluster_assignments_bytes == n * sizeof(uint32_t),
                   "cluster-id section size does not match num_vectors");
    IVF_PQ_REQUIRE(h.posting_offsets_bytes == (uint64_t(h.nlist) + 1) * sizeof(uint32_t),
                   "posting-offsets section size does not match nlist + 1");
    IVF_PQ_REQUIRE(h.posting_ids_bytes == n * sizeof(uint32_t),
                   "posting-ids section size does not match num_vectors");
    IVF_PQ_REQUIRE(h.pq_pivots_bytes == uint64_t(h.pq_chunks) * h.pq_k * h.pq_chunk_dim * sizeof(float),
                   "PQ pivot section size does not match chunks x k x chunk_dim");
    IVF_PQ_REQUIRE(h.pq_codes_bytes == n * h.pq_chunks, "PQ code section size does not match N x chunks");
    IVF_PQ_REQUIRE(h.rid_table_bytes == n * sizeof(uint32_t),
                   "RID-table section size does not match num_vectors");
}

void require_header(const IVFPQIndexFileHeader& h, const std::string& path) {
    IVF_PQ_REQUIRE(h.magic == IVF_PQ_INDEX_MAGIC, "not an IVF-PQ index file: " + path);
    IVF_PQ_REQUIRE(h.version == IVF_PQ_INDEX_VERSION,
                   "IVF-PQ index file version " + std::to_string(h.version) + " is not supported");
    IVF_PQ_REQUIRE(h.nlist > 0 && h.dim > 0 && h.aligned_dim == align_dim(h.dim),
                   "IVF-PQ index header has an invalid nlist/dim");
    IVF_PQ_REQUIRE(h.pq_chunks > 0 && h.pq_k > 0 && h.pq_k <= 256 &&
                       uint64_t(h.pq_chunks) * h.pq_chunk_dim == h.dim,
                   "IVF-PQ index header has an invalid PQ shape");
    IVF_PQ_REQUIRE(h.num_vectors <= uint64_t(RAW_VECTOR_RID_SLOT_MASK) + 1,
                   "IVF-PQ index header num_vectors exceeds the RID slot space");
    IVF_PQ_REQUIRE(h.raw_vector_elem_size > 0 && h.raw_vector_elem_size % h.dim == 0 &&
                       h.raw_vector_page_size > 0 && h.raw_vectors_bytes % h.raw_vector_page_size == 0 &&
                       h.raw_vectors_bytes / h.raw_vector_page_size <= UINT32_MAX &&
                       h.raw_vector_next_slot <= uint64_t(RAW_VECTOR_RID_SLOT_MASK) + 1,
                   "IVF-PQ index header has an invalid raw-vector heap descriptor");
    require_section_sizes(h);
}

// Posting lists are the exact inverse of the cluster ids; RIDs point below
// the heap's allocation cursor, which lies within its allocated pages.
void require_consistent(const IVFPQIndex& index) {
    const size_t n = index.assignments.cluster_id.size();
    const uint32_t nlist = index.meta.nlist;
    for (size_t i = 0; i < n; ++i) {
        IVF_PQ_REQUIRE(index.assignments.cluster_id[i] < nlist,
                       "vector " + std::to_string(i) + " is assigned to a cluster >= nlist");
    }

    const std::vector<uint32_t>& offsets = index.lists.offsets;
    const std::vector<uint32_t>& ids = index.lists.ids;
    IVF_PQ_REQUIRE(offsets.front() == 0 && offsets.back() == n && std::is_sorted(offsets.begin(), offsets.end()),
                   "posting offsets are not a valid row index over num_vectors");
    std::vector<uint8_t> listed(n, 0);
    for (uint32_t c = 0; c < nlist; ++c) {
        for (uint32_t k = offsets[c]; k < offsets[c + 1]; ++k) {
            uint32_t id = ids[k];
            IVF_PQ_REQUIRE(id < n && index.assignments.cluster_id[id] == c && !listed[id],
                           "posting list " + std::to_string(c) + " holds vector " + std::to_string(id) +
                               " which is out of range, assigned elsewhere, or listed twice");
            listed[id] = 1;
        }
    }

    const uint64_t slots = uint64_t(index.heap_pages) * index.heap_layout.slots_per_page;
    IVF_PQ_REQUIRE(index.heap_next_slot <= slots, "heap slot cursor lies past the heap's allocated pages");
    for (size_t i = 0; i < n; ++i) {
        IVF_PQ_REQUIRE(rid_flat_slot(index.rid_table.rid[i]) < index.heap_next_slot,
                       "RID of vector " + std::to_string(i) + " lies at or past the heap's slot cursor");
    }
}

}  // namespace

void write_ivf_pq_index(const std::string& index_prefix, const IVFPQIndex& index) {
    IVFPQIndexFileHeader h = make_header(index);
    require_header(h, index_prefix);
    require_consistent(index);

    const std::string path = ivf_pq_index_path(index_prefix);
    const std::string tmp = path + ".tmp";
    {
        FileWriter out(tmp);
        out.write(&h, sizeof(h));
        out.write(index.meta.centroids);
        out.write(index.assignments.cluster_id);
        out.write(index.lists.offsets);
        out.write(index.lists.ids);
        out.write(index.pq.pivots);
        out.write(index.pq.codes);
        static_assert(sizeof(RawVectorRID) == sizeof(uint32_t), "RID table is stored as uint32");
        out.write(index.rid_table.rid.data(), bytes_of(index.rid_table.rid));
        out.sync_and_close();
    }
    IVF_PQ_REQUIRE(::rename(tmp.c_str(), path.c_str()) == 0, "Failed to move " + tmp + " into place");
}

IVFPQIndex load_ivf_pq_index(const std::string& index_prefix) {
    const std::string path = ivf_pq_index_path(index_prefix);
    IVF_PQ_REQUIRE(file_exists(path), "file not found: " + path);
    FileReader in(path);

    IVFPQIndexFileHeader h;
    std::vector<char> raw = in.read<char>(0, sizeof(h), "header");
    std::memcpy(&h, raw.data(), sizeof(h));
    require_header(h, path);

    IVFPQIndex index;
    index.meta.nlist = h.nlist;
    index.meta.dim = h.dim;
    index.meta.aligned_dim = h.aligned_dim;
    index.meta.centroids = in.read<float>(h.centroids_offset, h.centroids_bytes, "centroids");
    index.assignments.cluster_id = in.read<uint32_t>(h.cluster_assignments_offset, h.cluster_assignments_bytes, "cluster ids");
    index.lists.offsets = in.read<uint32_t>(h.posting_offsets_offset, h.posting_offsets_bytes, "posting offsets");
    index.lists.ids = in.read<uint32_t>(h.posting_ids_offset, h.posting_ids_bytes, "posting ids");
    index.pq.chunks = h.pq_chunks;
    index.pq.chunk_dim = h.pq_chunk_dim;
    index.pq.k = h.pq_k;
    index.pq.pivots = in.read<float>(h.pq_pivots_offset, h.pq_pivots_bytes, "PQ pivots");
    index.pq.codes = in.read<uint8_t>(h.pq_codes_offset, h.pq_codes_bytes, "PQ codes");
    std::vector<uint32_t> packed = in.read<uint32_t>(h.rid_table_offset, h.rid_table_bytes, "RID table");
    index.rid_table.rid.reserve(packed.size());
    for (uint32_t p : packed) index.rid_table.rid.push_back(RawVectorRID{p});

    index.heap_layout = compute_raw_vector_heap_layout(h.raw_vector_page_size, h.raw_vector_elem_size);
    index.heap_pages = uint32_t(h.raw_vectors_bytes / h.raw_vector_page_size);
    index.heap_next_slot = uint32_t(h.raw_vector_next_slot);
    require_consistent(index);

    const std::string heap_path = ivf_raw_vectors_path(index_prefix);
    IVF_PQ_REQUIRE(file_exists(heap_path) && get_file_size(heap_path) == h.raw_vectors_bytes,
                   "raw-vector heap " + heap_path + " is missing or not the " +
                       std::to_string(h.raw_vectors_bytes) + " bytes the index records");
    return index;
}

}  // namespace inplace
}  // namespace diskann
