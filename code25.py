# short_sparse.py — compact sparse matrix (COO-like) with basic stats
import numpy as np

class CompactSparse:
    def __init__(self, shape, rows=None, cols=None, vals=None):
        self.shape = tuple(shape)
        self.rows = np.asarray(rows, dtype=np.int32) if rows is not None else np.empty(0, dtype=np.int32)
        self.cols = np.asarray(cols, dtype=np.int32) if cols is not None else np.empty(0, dtype=np.int32)
        self.vals = np.asarray(vals, dtype=np.float64) if vals is not None else np.empty(0, dtype=np.float64)
        assert len(self.rows)==len(self.cols)==len(self.vals)

    @classmethod
    def random(cls, shape, density=1e-4, value_range=(0.0,1.0), seed=None):
        rng = np.random.default_rng(seed)
        nrows, ncols = shape
        nnz = int(round(nrows * ncols * density))
        if nnz == 0:
            return cls(shape)
        idx = rng.choice(nrows * ncols, size=nnz, replace=False)
        r = (idx // ncols).astype(np.int32)
        c = (idx % ncols).astype(np.int32)
        v = rng.uniform(value_range[0], value_range[1], size=nnz).astype(np.float64)
        return cls(shape, r, c, v)

    # basic stats
    def nnz(self): return int(self.vals.size)
    def density(self): return self.nnz() / (self.shape[0]*self.shape[1])
    def nonzero_mean(self): return float(self.vals.mean()) if self.nnz()>0 else 0.0
    def nonzero_sum(self): return float(self.vals.sum())
    # memory (bytes)
    def sparse_bytes(self): return self.rows.nbytes + self.cols.nbytes + self.vals.nbytes
    def dense_bytes_equiv(self): return int(np.prod(self.shape) * np.dtype(np.float64).itemsize)
    # safe dense conversion (guard recommended for very large shapes)
    def to_dense(self):
        D = np.zeros(self.shape, dtype=np.float64)
        if self.nnz()>0:
            D[self.rows, self.cols] += self.vals
        return D
    # simple scalar multiply (returns new CompactSparse)
    def scalar_multiply(self, s):
        return CompactSparse(self.shape, self.rows.copy(), self.cols.copy(), (self.vals * s).copy())

# ------------------ example usage ------------------
if __name__ == "__main__":
    # create a 10000x10000 sparse matrix with 0.01% density (~10k nonzeros)
    S = CompactSparse.random((10000,10000), density=1e-4, value_range=(1,100), seed=0)
    print("shape:", S.shape)
    print("nnz:", S.nnz())
    print("density:", f"{S.density():.6f}")
    print("nonzero mean:", f"{S.nonzero_mean():.4f}")
    print("nonzero sum:", f"{S.nonzero_sum():.2f}")
    print("sparse memory (bytes):", S.sparse_bytes())
    print("dense equivalent (bytes):", S.dense_bytes_equiv())
    print("memory saving ≈", f"{S.dense_bytes_equiv() / max(1, S.sparse_bytes()):.1f}x")

    # demonstrate scalar multiply and new mean
    S2 = S.scalar_multiply(2.0)
    print("mean after *2:", f"{S2.nonzero_mean():.4f}")

    # optional sanity check for small matrices only:
    if S.shape[0]*S.shape[1] <= 5_000_000:
        dense = S.to_dense()
        print("dense nnz:", int(np.count_nonzero(dense)))
