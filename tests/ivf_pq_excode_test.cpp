// Tests for the 9-bit ex-code store (ivf_pq_excode.h): the rotation is
// orthogonal; the estimate is within its deterministic bound of the brute
// force distance, with the bound's angle recomputed here from the record's
// documented layout, and ranks neighbours nearly as brute force does; the
// store's file equals one record per base vector in posting order encoded
// directly, byte for byte, however many build windows it took; the index
// file round-trips it and rejects corrupt descriptors; the search with an
// exact re-rank of the whole shortlist equals the raw re-rank, finds inserts,
// and counts its pages.
//
//   ivf_pq_excode_test                                 the self-checking tests
//   ivf_pq_excode_test <base> <query> <gt> [nlist] [chunks]
//                                                      recall on a real (float or
//                                                      uint8, by extension) dataset

#include "bufann/ivf_pq_excode.h"
#include "bufann/ivf_pq_mutate.h"
#include "ivf_pq_test_util.h"

#include <fcntl.h>
#include <unistd.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <set>

using namespace diskann::inplace;

namespace {

const uint32_t N = 6000, NQ = 40, NLIST = 16, CHUNKS = 8, K = 10, SEED = 7;

std::vector<float> rotate(const std::vector<float>& P, const float* v, uint32_t dim) {
    std::vector<float> out(dim, 0.0f);
    for (uint32_t i = 0; i < dim; ++i) {
        double s = 0.0;
        for (uint32_t j = 0; j < dim; ++j) s += double(P[size_t(i) * dim + j]) * v[j];
        out[i] = float(s);
    }
    return out;
}

// x = u - 255.5 from the record's sign plane and low bytes.
std::vector<double> decode_x(const uint8_t* rec, uint32_t dim) {
    const uint8_t* signs = rec + 8;
    const uint8_t* low = signs + (dim + 7) / 8;
    std::vector<double> x(dim);
    for (uint32_t i = 0; i < dim; ++i) x[i] = double(((signs[i / 8] >> (i % 8)) & 1) * 256 + low[i]) - 255.5;
    return x;
}

bool test_rotation_orthogonal(uint32_t dim) {
    TestCase t("random rotation is orthogonal, dim " + std::to_string(dim));
    const std::vector<float> P = make_ivf_random_rotation(dim, 3);
    double worst = 0.0;
    for (uint32_t i = 0; i < dim; ++i) {
        for (uint32_t j = 0; j < dim; ++j) {
            double s = 0.0;
            for (uint32_t d = 0; d < dim; ++d) s += double(P[size_t(i) * dim + d]) * P[size_t(j) * dim + d];
            worst = std::max(worst, std::fabs(s - (i == j ? 1.0 : 0.0)));
        }
    }
    t.check(worst < 1e-5, "P P^T differs from I by " + std::to_string(worst));
    t.check(make_ivf_random_rotation(dim, 3) == P, "rotation is not deterministic in its seed");
    t.check(make_ivf_random_rotation(dim, 4) != P, "rotation ignores its seed");
    return t.done();
}

// Vectors around one centre, encoded directly; the estimate against brute
// force for every (query, vector) pair.
bool test_estimate_bound(uint32_t dim) {
    TestCase t("estimate within its bound of brute force, dim " + std::to_string(dim));
    const uint32_t n = 2000, nq = 20;
    std::vector<float> base = draw_blobs<float>(n, 11, dim, 1), queries = draw_blobs<float>(nq, 12, dim, 1);
    std::vector<float> c(dim, 0.0f);
    for (uint32_t i = 0; i < n; ++i)
        for (uint32_t d = 0; d < dim; ++d) c[d] += base[size_t(i) * dim + d] / n;
    IVFExCodes ex;
    ex.bits = IVF_EXCODE_BITS;
    ex.rotation = make_ivf_random_rotation(dim, 5);
    const std::vector<float> rc = rotate(ex.rotation, c.data(), dim);
    const uint32_t rec = ivf_excode_record_bytes(dim);
    std::vector<uint8_t> records(size_t(n) * rec);
    std::vector<double> tan_theta(n), rnorm(n);
    double tan_sum = 0.0;
    for (uint32_t i = 0; i < n; ++i) {
        std::vector<float> res(dim);
        for (uint32_t d = 0; d < dim; ++d) res[d] = base[size_t(i) * dim + d] - c[d];
        const std::vector<float> r = rotate(ex.rotation, res.data(), dim);
        ivf_excode_encode(r.data(), rc.data(), dim, records.data() + size_t(i) * rec);
        const std::vector<double> x = decode_x(records.data() + size_t(i) * rec, dim);
        double xr = 0, xx = 0, rr = 0;
        for (uint32_t d = 0; d < dim; ++d) xr += x[d] * r[d], xx += x[d] * x[d], rr += double(r[d]) * r[d];
        const double cos = xr / std::sqrt(xx * rr);
        tan_theta[i] = std::sqrt(std::max(0.0, 1.0 - cos * cos)) / cos;
        rnorm[i] = std::sqrt(rr);
        tan_sum += tan_theta[i];
    }
    // 8 magnitude bits put x within a few thousandths of a radian of r.
    t.check(tan_sum / n < 0.004, "mean tan(theta) " + std::to_string(tan_sum / n) + " is too large for 9 bits");
    size_t outside = 0, hits = 0;
    double rel_err = 0.0;
    IVFExQuery eq;
    for (uint32_t q = 0; q < nq; ++q) {
        const float* query = queries.data() + size_t(q) * dim;
        ivf_excode_prepare_query(ex, dim, query, eq);
        const float qc = sq_dist(query, c.data(), dim);
        std::vector<float> exact(n), est(n);
        for (uint32_t i = 0; i < n; ++i) {
            exact[i] = sq_dist(query, base.data() + size_t(i) * dim, dim);
            est[i] = ivf_excode_estimate(records.data() + size_t(i) * rec, dim, eq, qc);
            // The bound, plus float rounding of the terms that cancel.
            const double bound = 2.0 * rnorm[i] * std::sqrt(qc) * tan_theta[i] + 1e-4 * (exact[i] + qc + rnorm[i] * rnorm[i]);
            outside += std::fabs(double(est[i]) - exact[i]) > bound;
            rel_err += std::fabs(double(est[i]) - exact[i]) / exact[i];
        }
        const std::vector<uint32_t> a = top_k(exact, K), b = top_k(est, K);
        for (uint32_t id : b) hits += std::find(a.begin(), a.end(), id) != a.end();
    }
    t.check(outside == 0, std::to_string(outside) + " estimates lie outside their error bound");
    t.check(rel_err / (double(n) * nq) < 2e-3, "mean relative error " + std::to_string(rel_err / (double(n) * nq)));
    t.check(hits >= size_t(0.95 * nq * K), "top-10 by estimate recalls " + std::to_string(hits) + " of " +
                                               std::to_string(nq * K) + " brute-force neighbours");
    // A vector at its centre estimates as the centre's distance.
    std::vector<uint8_t> zero(rec);
    std::vector<float> zr(dim, 0.0f);
    ivf_excode_encode(zr.data(), rc.data(), dim, zero.data());
    ivf_excode_prepare_query(ex, dim, queries.data(), eq);
    t.check(ivf_excode_estimate(zero.data(), dim, eq, 123.0f) == 123.0f, "a zero residual does not estimate as ||q-c||^2");
    return t.done();
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), {});
}

template<typename T>
void patch(const std::string& path, size_t offset, T value) {
    const int fd = ::open(path.c_str(), O_WRONLY);
    if (fd < 0 || ::pwrite(fd, &value, sizeof(T), off_t(offset)) != ssize_t(sizeof(T))) throw std::runtime_error("patch failed");
    ::close(fd);
}

// The store's file against records encoded here in posting order, laid out
// as compute_raw_vector_heap_layout says; then the index file round trip.
bool test_store_file(const std::string& prefix, Built& b, const std::vector<float>& base, uint32_t dim) {
    TestCase t("ex-code store file and index round trip");
    const std::string base_bin = prefix + "_base.bin";
    diskann::save_bin<float>(base_bin, const_cast<float*>(base.data()), N, dim);
    IVFPQIndex& ix = b.index;
    const uint32_t rec = ivf_excode_record_bytes(dim);
    // One window, then windows of 7 records: the same file.
    build_ivf_excode_store<float>(base_bin, prefix, ix, 4096, SEED);
    const std::string one = read_file(ivf_excodes_path(prefix));
    build_ivf_excode_store<float>(base_bin, prefix, ix, 4096, SEED, 7 * rec);
    const std::string many = read_file(ivf_excodes_path(prefix));
    t.check(one == many, "the store depends on the build's window size");

    const RawVectorHeapLayout L = ix.excodes.layout;
    t.check(L.elem_size == rec && ix.excodes.next_slot == N && ix.excodes.bits == 9, "store descriptor is wrong");
    t.check(one.size() == size_t(ix.excodes.pages) * 4096 && ix.excodes.pages == (N + L.slots_per_page - 1) / L.slots_per_page,
            "store file has " + std::to_string(one.size()) + " bytes");
    const std::vector<float> P = ix.excodes.rotation;
    size_t bad = 0;
    std::vector<uint8_t> want(rec);
    for (uint32_t p = 0; p < N; ++p) {
        const uint32_t id = ix.lists.ids[p], list = ix.assignments.cluster_id[id];
        const float* c = ix.meta.centroids.data() + size_t(list) * ix.meta.aligned_dim;
        std::vector<float> res(dim);
        for (uint32_t d = 0; d < dim; ++d) res[d] = base[size_t(id) * dim + d] - c[d];
        const std::vector<float> r = rotate(P, res.data(), dim), rc = rotate(P, c, dim);
        ivf_excode_encode(r.data(), rc.data(), dim, want.data());
        const size_t page = size_t(p / L.slots_per_page) * 4096, idx = p % L.slots_per_page;
        uint32_t owner;
        std::memcpy(&owner, one.data() + page + L.owners_offset + idx * 4, 4);
        const bool occupied = (uint8_t(one[page + 16 + idx / 8]) >> (idx % 8)) & 1;
        // The rotations here and in the build's GEMM round differently, and
        // a rescaling factor near a tie may then flip, so the record need
        // not equal `want` bit for bit; it must encode this vector as well:
        // x as close to r in angle, and alpha <x, r> = ||r||^2.
        const uint8_t* got = reinterpret_cast<const uint8_t*>(one.data() + page + L.slots_offset + idx * rec);
        const std::vector<double> xg = decode_x(got, dim), xw = decode_x(want.data(), dim);
        double gr = 0, gg = 0, wr = 0, ww = 0, rr = 0;
        for (uint32_t d = 0; d < dim; ++d) {
            gr += xg[d] * r[d], gg += xg[d] * xg[d], wr += xw[d] * r[d], ww += xw[d] * xw[d], rr += double(r[d]) * r[d];
        }
        float f_rescale;
        std::memcpy(&f_rescale, got + 4, 4);
        const double cos_g = gr / std::sqrt(gg * rr), cos_w = wr / std::sqrt(ww * rr);
        bad += owner != id || !occupied || cos_g < cos_w - 1e-6 ||
               std::fabs(-0.5 * f_rescale * gr - rr) > 1e-4 * rr;
    }
    t.check(bad == 0, std::to_string(bad) + " records differ from direct posting-order encoding");

    write_ivf_pq_index(prefix, ix);
    IVFPQIndex loaded = load_ivf_pq_index(prefix);
    t.check(loaded.excodes.bits == 9 && loaded.excodes.rotation == P && loaded.excodes.pages == ix.excodes.pages &&
                loaded.excodes.next_slot == N && loaded.excodes.layout.slots_per_page == L.slots_per_page,
            "ex-code descriptor did not round-trip");
    IVFPQIndexFileHeader h;
    std::memcpy(&h, read_file(ivf_pq_index_path(prefix)).data(), sizeof(h));
    t.check(h.version == 6 && h.excode_rotation_offset == h.rid_table_offset + h.rid_table_bytes, "rotation section misplaced");
    const std::string path = ivf_pq_index_path(prefix);
    auto corrupt = [&](const char* what, auto&& fn) {
        write_ivf_pq_index(prefix, ix);
        fn();
        t.expect_throw(what, [&] { load_ivf_pq_index(prefix); });
    };
    corrupt("ex bits 8", [&] { patch<uint32_t>(path, offsetof(IVFPQIndexFileHeader, excode_bits), 8); });
    corrupt("ex record size", [&] { patch<uint32_t>(path, offsetof(IVFPQIndexFileHeader, excode_elem_size), rec + 1); });
    corrupt("ex slot cursor short of N",
            [&] { patch<uint64_t>(path, offsetof(IVFPQIndexFileHeader, excode_next_slot), N - 1); });
    corrupt("ex fields without bits", [&] { patch<uint32_t>(path, offsetof(IVFPQIndexFileHeader, excode_bits), 0); });
    corrupt("store file shorter than recorded",
            [&] { ::truncate(ivf_excodes_path(prefix).c_str(), off_t(one.size() - 4096)); });
    corrupt("store file missing", [&] { ::unlink(ivf_excodes_path(prefix).c_str()); });
    // An index without a store still writes and loads as before.
    IVFPQIndex plain = ix;
    plain.excodes = IVFExCodes{};
    write_ivf_pq_index(prefix, plain);
    t.check(load_ivf_pq_index(prefix).excodes.bits == 0, "an index without a store loads one");
    // Leave a good store and index file for the search tests.
    build_ivf_excode_store<float>(base_bin, prefix, ix, 4096, SEED);
    write_ivf_pq_index(prefix, ix);
    ::unlink(base_bin.c_str());
    return t.done();
}

bool test_search(const std::string& prefix, Built& b, const std::vector<float>& base, const std::vector<float>& queries,
                 uint32_t dim) {
    TestCase t("search with the ex-code re-rank");
    IVFPQIndex& ix = b.index;
    RawVectorHeap store;
    store.open_existing(ivf_excodes_path(prefix), ix.excodes.layout, ix.excodes.next_slot, ix.excodes.pages);
    const uint32_t nprobe = 4, rerank_m = 60;
    IVFPQSearchScratch s;
    size_t hits_ex = 0, hits_raw = 0;
    std::vector<uint8_t> dead(N, 0);
    for (int fast = 0; fast < 2; ++fast) {
        set_fastscan_enabled(fast == 1);
        for (uint32_t q = 0; q < NQ; ++q) {
            const float* query = queries.data() + size_t(q) * dim;
            const IVFPQSearchResult raw = ivf_pq_search<float>(ix, b.heap, query, K, nprobe, rerank_m, s);
            const IVFPQSearchResult all =
                ivf_pq_search<float>(ix, b.heap, query, K, nprobe, rerank_m, s, nullptr, {}, {&store, rerank_m});
            t.check(all.ids == raw.ids && all.dists == raw.dists, "exact re-rank of the whole shortlist differs from the raw re-rank");
            const IVFPQSearchResult est =
                ivf_pq_search<float>(ix, b.heap, query, K, nprobe, rerank_m, s, nullptr, {}, {&store, 0});
            t.check(est.ids.size() == K && est.io.raw_vectors == 0 && est.io.raw_pages == 0,
                    "estimate-only re-rank read the raw heap");
            t.check(est.io.ex_records == rerank_m && est.io.ex_pages > 0 && est.io.ex_pages <= est.io.ex_records,
                    "ex-code I/O counts are off");
            // Every record of a page shares it: no more pages than the
            // candidates' distinct pages, which here are bounded by the
            // probed lists' page spans.
            t.check(raw.io.raw_vectors == rerank_m && raw.io.ex_records == 0, "raw re-rank I/O counts are off");
            const IVFPQSearchResult want = brute_force(base, dim, dead, query, K);
            for (uint32_t id : est.ids) hits_ex += std::find(want.ids.begin(), want.ids.end(), id) != want.ids.end();
            for (uint32_t id : raw.ids) hits_raw += std::find(want.ids.begin(), want.ids.end(), id) != want.ids.end();
            // The estimates are near the exact distances of the same ids.
            for (size_t i = 0; i < est.ids.size(); ++i) {
                const float d = sq_dist(query, base.data() + size_t(est.ids[i]) * dim, dim);
                t.check(std::fabs(est.dists[i] - d) <= 0.02f * d + 1.0f, "estimate far from exact distance");
            }
            const IVFPQSearchResult some =
                ivf_pq_search<float>(ix, b.heap, query, K, nprobe, rerank_m, s, nullptr, {}, {&store, 20});
            t.check(some.io.raw_vectors == 20, "exact stage read " + std::to_string(some.io.raw_vectors) + " vectors, not 20");
        }
    }
    set_fastscan_enabled(true);
    t.check(hits_ex + 4 >= hits_raw, "estimate-only recall " + std::to_string(hits_ex) + " trails raw " + std::to_string(hits_raw));
    const float* q0 = queries.data();
    t.expect_throw("exact count below k", [&] { ivf_pq_search<float>(ix, b.heap, q0, K, 4, 60, s, nullptr, {}, {&store, K - 1}); });
    {
        IVFPQIndex none = ix;
        none.excodes = IVFExCodes{};
        t.expect_throw("store on an index without one",
                       [&] { ivf_pq_search<float>(none, b.heap, q0, K, 4, 60, s, nullptr, {}, {&store, 0}); });
    }
    t.expect_throw("store with another geometry",
                   [&] { ivf_pq_search<float>(ix, b.heap, q0, K, 4, 60, s, nullptr, {}, {&b.heap, 0}); });

    // An insert has no record: it is re-ranked exactly and found.
    std::vector<float> v(queries.begin(), queries.begin() + dim);
    const uint32_t id = ivf_pq_insert<float>(ix, b.heap, b.delta, v.data());
    for (int fast = 0; fast < 2; ++fast) {
        set_fastscan_enabled(fast == 1);
        const IVFPQSearchResult r =
            ivf_pq_search<float>(ix, b.heap, q0, K, ix.meta.nlist, rerank_m, s, &b.delta, {}, {&store, 0});
        t.check(!r.ids.empty() && r.ids[0] == id && r.dists[0] == 0.0f && r.io.raw_vectors == 1,
                "the inserted copy of the query is not first with distance 0");
    }
    set_fastscan_enabled(true);
    store.close();
    return t.done();
}

template<typename T>
int real_dataset(int argc, char** argv) {
    const std::string base_bin = argv[1], query_bin = argv[2], gt = argv[3];
    const uint32_t nlist = argc > 4 ? uint32_t(std::stoul(argv[4])) : 1024;
    const uint32_t chunks = argc > 5 ? uint32_t(std::stoul(argv[5])) : 32;
    const std::string prefix = temp_path("ivf_excode_real");
    size_t n = 0, dim = 0;
    diskann::get_bin_metadata(base_bin, n, dim);
    IVFPQIndex ix;
    ix.meta = train_ivf_centroids<T>(base_bin, nlist, 0.0, NUM_K_MEANS_ITERS, SEED);
    ix.heap_layout = compute_raw_vector_heap_layout(4096, uint32_t(dim * sizeof(T)));
    RawVectorHeap heap;
    heap.open(ivf_raw_vectors_path(prefix), ix.heap_layout);
    assign_ivf_clusters<T>(base_bin, ix.meta, heap, ix.assignments, ix.rid_table);
    ix.heap_pages = heap.allocated_pages();
    ix.heap_next_slot = heap.next_flat_slot();
    ix.lists = build_ivf_posting_lists(ix.assignments, nlist);
    train_ivf_pq_pivots<T>(base_bin, prefix, chunks, 0.0, NUM_K_MEANS_ITERS, SEED);
    encode_ivf_pq_codes<T>(base_bin, prefix, chunks);
    ix.pq = load_ivf_pq(prefix);
    build_ivf_excode_store<T>(base_bin, prefix, ix, 4096, SEED);
    write_ivf_pq_index(prefix, ix);
    heap.close();
    IVFPQIndex loaded = load_ivf_pq_index(prefix);
    heap.open_existing(ivf_raw_vectors_path(prefix), loaded.heap_layout, loaded.heap_next_slot, loaded.heap_pages);
    RawVectorHeap store;
    store.open_existing(ivf_excodes_path(prefix), loaded.excodes.layout, loaded.excodes.next_slot, loaded.excodes.pages);

    T* qraw = nullptr;
    size_t nq = 0, qdim = 0;
    diskann::load_bin<T>(query_bin, qraw, nq, qdim);
    std::vector<float> queries(qraw, qraw + nq * qdim);
    delete[] qraw;
    uint32_t* gt_ids = nullptr;
    float* gt_d = nullptr;
    size_t gn = 0, gk = 0;
    diskann::load_truthset(gt, gt_ids, gt_d, gn, gk);
    bool ok = true;
    for (uint32_t exact : {0u, 20u, 0xFFFFFFFFu}) {
        size_t hits = 0, ex_pages = 0, raw_pages = 0;
#pragma omp parallel for reduction(+ : hits, ex_pages, raw_pages)
        for (int64_t q = 0; q < int64_t(nq); ++q) {
            static thread_local IVFPQSearchScratch s;
            const float* query = queries.data() + size_t(q) * dim;
            const IVFPQSearchResult r =
                exact == 0xFFFFFFFFu ? ivf_pq_search<T>(loaded, heap, query, K, 64, 200, s)
                                     : ivf_pq_search<T>(loaded, heap, query, K, 64, 200, s, nullptr, {}, {&store, exact});
            for (uint32_t id : r.ids) hits += std::find(gt_ids + q * gk, gt_ids + q * gk + K, id) != gt_ids + q * gk + K;
            ex_pages += r.io.ex_pages, raw_pages += r.io.raw_pages;
        }
        const double recall = double(hits) / double(nq * K);
        std::printf("%s nprobe 64 rerank_m 200: recall@10 %.4f, pages/query ex %.1f raw %.1f\n",
                    exact == 0xFFFFFFFFu ? "raw re-rank" : ("ex9 exact " + std::to_string(exact)).c_str(), recall,
                    double(ex_pages) / nq, double(raw_pages) / nq);
        ok &= recall >= 0.99;
    }
    delete[] gt_ids;
    delete[] gt_d;
    store.close();
    heap.close();
    remove_index_files(prefix);
    std::printf("%s\n", ok ? "PASS" : "FAIL: recall@10 below 0.99");
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc >= 4) {
        const std::string base = argv[1];
        const bool u8 = base.size() > 6 && (base.find("u8") != std::string::npos || base.find("uint8") != std::string::npos);
        return u8 ? real_dataset<uint8_t>(argc, argv) : real_dataset<float>(argc, argv);
    }
    bool all = true;
    const std::string prefix = temp_path("ivf_excode");
    try {
        all &= test_rotation_orthogonal(20);
        all &= test_rotation_orthogonal(128);
        all &= test_estimate_bound(128);
        all &= test_estimate_bound(100);  // not a multiple of 8
        all &= test_estimate_bound(960);
        const uint32_t dim = 64;
        std::vector<float> base = draw_blobs<float>(N, 1, dim, 24), queries = draw_blobs<float>(NQ, 2, dim, 24);
        std::unique_ptr<Built> b = build_index<float>(prefix, base, N, dim, NLIST, CHUNKS, SEED);
        all &= test_store_file(prefix, *b, base, dim);
        all &= test_search(prefix, *b, base, queries, dim);
        b->heap.close();
    } catch (const diskann::ANNException& e) {
        std::cout << "ERROR: " << e.message() << std::endl;
        all = false;
    } catch (const std::exception& e) {
        std::cout << "ERROR: " << e.what() << std::endl;
        all = false;
    }
    remove_index_files(prefix);
    std::cout << (all ? "ALL PASS" : "SOME FAILED") << std::endl;
    return all ? 0 : 1;
}
