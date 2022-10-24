# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

from time import perf_counter

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport free, malloc
import psutil


# forward star 1
# ==============


cpdef void loop_CSR_1(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count):
    """Loop over all vertex outgoing edges using the forward star representation with:
    - a pointer (csr_indptr)
    - a vector storing the head vertex indices (csr_indices)
    - a vector storing the weight (csr_data)
    """

    cdef: 
        ssize_t tail_vert_idx, head_vert_idx, ptr
        cnp.float64_t edge_weight

    with nogil:

        for tail_vert_idx in range(<ssize_t>vertex_count):
            for ptr in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = csr_indices[ptr]
                edge_weight = csr_data[ptr]

    # for tail_vert_idx in range(100):
    #     for ptr in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
    #         head_vert_idx = csr_indices[ptr]
    #         edge_weight = csr_data[ptr]
    #         print(f"tail : {tail_vert_idx:10d}, head {head_vert_idx:10d}, weight : {edge_weight:16.4}")

cpdef csr_1(edges_df, vertex_count, edge_count):
    """Loop over all vertex outgoing edges using the forward star representation with:
    - a pointer (csr_indptr)
    - a vector storing the head vertex indices (csr_indices)
    - a vector storing the weight (csr_data)
    """

    csr_indptr = np.zeros(vertex_count + 1, dtype=np.uint32)
    csr_indices = np.zeros(edge_count, dtype=np.uint32)
    csr_data = np.zeros(edge_count, dtype=np.float64)

    # convert to CSR

    start = perf_counter()

    edges_df.sort_values(by="source", axis=0, ascending=True, inplace=True)

    cdef:

        cnp.uint32_t[::1] tail_view  = edges_df["source"].values
        cnp.uint32_t[::1] head_view = edges_df["target"].values
        cnp.float64_t[::1] data_view = edges_df["weight"].values

        cnp.uint32_t[::1] csr_indptr_view  = csr_indptr
        cnp.uint32_t[::1] csr_indices_view = csr_indices
        cnp.float64_t[::1] csr_data_view = csr_data

        ssize_t i, edge_count_ssize_t = <ssize_t> edge_count
        ssize_t vertex_count_ssize_t = <ssize_t> vertex_count
        ssize_t ptr_ssize_t
        cnp.uint32_t ptr, head_vert_idx
        cnp.float64_t edge_weight


    with nogil:

        for i in range(edge_count_ssize_t):
            csr_indptr_view[<ssize_t>(tail_view[i] + 1)] += 1
            csr_indices_view[i] = head_view[i]
            csr_data_view[i] = data_view[i]

        for i in range(vertex_count_ssize_t):
            csr_indptr_view[i + 1] += csr_indptr_view[i]

    end = perf_counter()
    elapsed_time = end - start
    print(f"convert to CSR - Elapsed time: {elapsed_time:12.8f} s")

    start = perf_counter()

    with nogil:

        for i in range(vertex_count_ssize_t):
            for ptr in range(csr_indptr_view[i], csr_indptr_view[i + 1]):
                ptr_ssize_t = <ssize_t>ptr
                head_vert_idx = csr_indices_view[ptr_ssize_t]
                edge_weight = csr_data_view[ptr_ssize_t]

    # for i in range(<ssize_t>100):
    #     for ptr in range(csr_indptr_view[i], csr_indptr_view[i + 1]):
    #         ptr_ssize_t = <ssize_t>ptr
    #         head_vert_idx = csr_indices_view[ptr_ssize_t]
    #         edge_weight = csr_data_view[ptr_ssize_t]
    #         print(f"tail : {i:10d}, head {head_vert_idx:10d}, weight : {edge_weight:16.4}")

    end = perf_counter()
    elapsed_time = end - start
    print(f"CSR_2 loop - Elapsed time: {elapsed_time:12.8f} s")


# forward star 2
# ==============

cdef struct Edge:
    ssize_t head
    cnp.float64_t weight

cpdef void loop_CSR_2(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count,
    int edge_count):
    """Loop over all vertex outgoing edges using the forward star representation with:
    - a pointer (csr_indptr)
    - a vector of struc storing a head vertvertex indices along the edge weights
    """

    start = perf_counter()

    edge_table = <Edge*> malloc(edge_count * sizeof(Edge))
    for i in range(<ssize_t>edge_count):
        edge_table[i].head = csr_indices[i]
        edge_table[i].weight = csr_data[i]

    end = perf_counter()
    elapsed_time = end - start
    print(f"CSR_2 init - Elapsed time: {elapsed_time:12.8f} s")

    start = perf_counter()

    cdef: 
        ssize_t tail_vert_idx, head_vert_idx, ptr
        cnp.float64_t edge_weight

    with nogil:

        for tail_vert_idx in range(<ssize_t>vertex_count):
            for ptr in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = edge_table[ptr].head
                edge_weight = edge_table[ptr].weight

    end = perf_counter()
    elapsed_time = end - start
    print(f"CSR_2 loop - Elapsed time: {elapsed_time:12.8f} s")

    start = perf_counter()

    free(edge_table)

    end = perf_counter()
    elapsed_time = end - start
    print(f"CSR_2 cleanup - Elapsed time: {elapsed_time:12.8f} s")


# adjacency list
# ==============

cdef struct AdjacencyList:
    ssize_t size
    ssize_t* vertices
    cnp.float64_t* weights

cdef struct AdjacencyLists:
    ssize_t vertex_count
    AdjacencyList* neighbors

cdef void init_AL(AdjacencyLists* adj, ssize_t vertex_count):

    adj.vertex_count = vertex_count
    adj.neighbors = <AdjacencyList*> malloc(vertex_count * sizeof(AdjacencyList))

cdef void create_AL(
    AdjacencyLists* adj,
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int num_threads) nogil:
    
    cdef:
        ssize_t i, tail_vert_idx, size, ptr
        int n_jobs = <int>num_threads

    for tail_vert_idx in prange(
        adj.vertex_count,
        num_threads=n_jobs):
        size = csr_indptr[tail_vert_idx + 1] - csr_indptr[tail_vert_idx]
        adj.neighbors[tail_vert_idx].size = size
        adj.neighbors[tail_vert_idx].vertices = <ssize_t*> malloc(size * sizeof(ssize_t))
        adj.neighbors[tail_vert_idx].weights = <cnp.float64_t*> malloc(size * sizeof(cnp.float64_t))
        for i in range(size):
            ptr = csr_indptr[tail_vert_idx] + i
            adj.neighbors[tail_vert_idx].vertices[i] = csr_indices[ptr]
            adj.neighbors[tail_vert_idx].weights[i] = csr_data[ptr]

cdef void loop_AL_inner(AdjacencyLists* adj) nogil:
    
    cdef:
        ssize_t i, tail_vert_idx, head_vert_idx
        cnp.float64_t edge_weight

    for tail_vert_idx in range(adj.vertex_count):
        for i in range(adj.neighbors[tail_vert_idx].size):
            head_vert_idx = adj.neighbors[tail_vert_idx].vertices[i]
            edge_weight = adj.neighbors[tail_vert_idx].weights[i] 

cdef void free_AL(AdjacencyLists* adj, int num_threads) nogil:

    cdef: 
        ssize_t tail_vert_idx
        int n_jobs = <int>num_threads

    for tail_vert_idx in prange(
        adj.vertex_count,
        num_threads=n_jobs):
        free(adj.neighbors[tail_vert_idx].vertices)
        free(adj.neighbors[tail_vert_idx].weights)

    free(adj.neighbors)

cpdef void loop_AL(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count):

    cdef:
        AdjacencyLists adj
        int n_jobs

    n_jobs = psutil.cpu_count()

    start = perf_counter()

    init_AL(&adj, vertex_count)

    end = perf_counter()
    elapsed_time = end - start
    print(f"AL init - Elapsed time: {elapsed_time:12.8f} s")

    start = perf_counter()

    create_AL(&adj, csr_indptr, csr_indices, csr_data, n_jobs)

    del csr_indptr, csr_indices, csr_data

    end = perf_counter()
    elapsed_time = end - start
    print(f"AL create - Elapsed time: {elapsed_time:12.8f} s")

    start = perf_counter()

    loop_AL_inner(&adj)

    end = perf_counter()
    elapsed_time = end - start
    print(f"AL loop - Elapsed time: {elapsed_time:12.8f} s")

    start = perf_counter()

    free_AL(&adj, n_jobs)

    end = perf_counter()
    elapsed_time = end - start
    print(f"AL free - Elapsed time: {elapsed_time:12.8f} s")
