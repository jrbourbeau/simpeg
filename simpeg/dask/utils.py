import numpy as np


def compute_chunk_sizes(M, N, target_chunk_size):
    """
    Compute row and collumn chunk sizes for a matrix of shape MxN,
    such that the chunks are below a certain threshold target_chunk_size (in Mb)
    """
    nChunks_col = 1
    nChunks_row = 1
    rowChunk = int(np.ceil(M / nChunks_row))
    colChunk = int(np.ceil(N / nChunks_col))
    chunk_size = rowChunk * colChunk * 8 * 1e-6

    # Add more chunks until memory falls below target
    while chunk_size >= target_chunk_size:
        if rowChunk > colChunk:
            nChunks_row += 1
        else:
            nChunks_col += 1

        rowChunk = int(np.ceil(M / nChunks_row))
        colChunk = int(np.ceil(N / nChunks_col))
        chunk_size = rowChunk * colChunk * 8 * 1e-6  # in Mb
    return rowChunk, colChunk


def is_arraylike(x) -> bool:
    """Is this object a numpy array or something similar?

    This function tests specifically for an object that already has
    array attributes (e.g. np.ndarray, dask.array.Array, cupy.ndarray,
    sparse.COO), **NOT** for something that can be coerced into an
    array object (e.g. Python lists and tuples). It is meant for dask
    developers and developers of downstream libraries.

    Note that this function does not correspond with NumPy's
    definition of array_like, which includes any object that can be
    coerced into an array (see definition in the NumPy glossary):
    https://numpy.org/doc/stable/glossary.html

    Examples
    --------
    >>> import numpy as np
    >>> is_arraylike(np.ones(5))
    True
    >>> is_arraylike(np.ones(()))
    True
    >>> is_arraylike(5)
    False
    >>> is_arraylike('cat')
    False
    """
    # from dask.base import is_dask_collection

    is_duck_array = hasattr(x, "__array_function__") or hasattr(x, "__array_ufunc__")

    return bool(
        hasattr(x, "shape")
        and isinstance(x.shape, tuple)
        and hasattr(x, "dtype")
        # and not any(is_dask_collection(n) for n in x.shape)
        # We special case scipy.sparse and cupyx.scipy.sparse arrays as having partial
        # support for them is useful in scenarios where we mostly call `map_partitions`
        # or `map_blocks` with scikit-learn functions on dask arrays and dask dataframes.
        # https://github.com/dask/dask/pull/3738
        and (is_duck_array or "scipy.sparse" in typename(type(x)))
    )
