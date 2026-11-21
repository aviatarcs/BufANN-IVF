# kmeans_min1.py <two_level_kmeans.py> <args...>
# Runs RaBitQ-SSD's clustering/two_level_kmeans.py unchanged, except that
# faiss.contrib.clustering.two_level_clustering's rebalanced split of the nc2
# fine centroids over the nc1 coarse cells gives every cell at least one.
# faiss 1.16 splits by rounding cumulative cell sizes, so a cell holding fewer
# than n_train / nc2 points gets 0 and k-means asserts 'clus.k > 0' (cohere768:
# a 14-point cell of 2.34M training points at C 9756). The centroids a cell
# gains are taken from the cells with the most, so the total stays nc2.
import runpy, sys
import numpy as np
import faiss.contrib.clustering as fcc



def split_min1(bc, nc2):
    bc_sum = np.cumsum(bc)
    alloc = bc_sum * nc2 // bc_sum[-1]
    alloc[1:] -= alloc[:-1].copy()
    for c in np.flatnonzero(alloc == 0):
        alloc[np.argmax(alloc)] -= 1
        alloc[c] = 1
    assert alloc.sum() == nc2 and alloc.min() >= 1
    return alloc


def two_level_clustering(xt, nc1, nc2, rebalance=True, clustering_niter=25, **args):
    # Same algorithm as faiss 1.16's, with split_min1 for the rebalanced split.
    d = xt.shape[1]
    verbose = args.get("verbose", False)
    log = print if verbose else (lambda *a, **k: None)
    km = fcc.faiss.Kmeans(d, nc1, niter=clustering_niter, max_points_per_centroid=2000, **args)
    km.train(xt)
    stats = [km.iteration_stats]
    _, assign1 = km.assign(xt)
    bc = np.bincount(assign1, minlength=nc1)
    o = assign1.argsort()
    del km
    if rebalance:
        all_nc2 = split_min1(bc, nc2)
    else:
        cc = np.arange(nc1 + 1) * nc2 // nc1
        all_nc2 = cc[1:] - cc[:-1]
    log(f"cells {bc.min()}-{bc.max()} points; 2nd-level centroids {all_nc2.min()}-{all_nc2.max()} (min-1 split)")
    i0, c2 = 0, []
    for c1 in range(nc1):
        i1 = i0 + bc[c1]
        km = fcc.faiss.Kmeans(d, int(all_nc2[c1]), **args)
        km.train(xt[o[i0:i1]])
        stats.append(km.iteration_stats)
        c2.append(km.centroids)
        del km
        i0 = i1
    return np.vstack(c2), stats


if __name__ == "__main__":
    # Self-check of the split on the failing shape before handing over.
    a = split_min1(np.array([14, 2_000_000, 341_426]), 9756)
    assert a[0] == 1 and a.sum() == 9756, a
    fcc.two_level_clustering = two_level_clustering
    script = sys.argv[1]
    sys.argv = sys.argv[1:]
    sys.path.insert(0, __import__("os").path.dirname(script))
    runpy.run_path(script, run_name="__main__")
