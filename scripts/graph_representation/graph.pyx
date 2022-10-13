# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False

from time import perf_counter

cimport numpy as cnp
from libc.stdlib cimport free, malloc


cpdef void loop_CSR(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count):
    
    cdef: 
        ssize_t tail_vert_idx, head_vert_idx, ptr
        cnp.float64_t edge_weight

    for tail_vert_idx in range(vertex_count):
    # for tail_vert_idx in range(10):
        # print(f"- {tail_vert_idx}")
        for ptr in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
            head_vert_idx = csr_indices[ptr]
            edge_weight = csr_data[ptr]
            # print(f"{head_vert_idx} : {edge_weight}")          

cdef struct ForwardStar:
    ssize_t size
    ssize_t* vertices
    cnp.float64_t* weights

cdef struct AdjacencyVectors:
    ForwardStar* forward_stars

cdef void init_adjacency_vectors(AdjacencyVectors* adj_vec, int vertex_count):

    adj_vec.forward_stars = <ForwardStar*> malloc(vertex_count * sizeof(ForwardStar))


cpdef void loop_FSV(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count):

    cdef:
        AdjacencyVectors adj_vec
        size_t size, tail_vert_idx, head_vert_idx, i
        cnp.float64_t edge_weight

    init_adjacency_vectors(&adj_vec, vertex_count)
    for tail_vert_idx in range(vertex_count):
        size = csr_indptr[tail_vert_idx + 1] - csr_indptr[tail_vert_idx]
        adj_vec.forward_stars[tail_vert_idx].size = size
        adj_vec.forward_stars[tail_vert_idx].vertices = <ssize_t*> malloc(size * sizeof(ssize_t))
        adj_vec.forward_stars[tail_vert_idx].weights = <cnp.float64_t*> malloc(size * sizeof(cnp.float64_t))
        for i in range(size):
            ptr = csr_indptr[tail_vert_idx] + i
            adj_vec.forward_stars[tail_vert_idx].vertices[i] = csr_indices[ptr]
            adj_vec.forward_stars[tail_vert_idx].weights[i] = csr_data[ptr]

    start = perf_counter()

    for tail_vert_idx in range(vertex_count):
    # for tail_vert_idx in range(10):
        # print(f"- {tail_vert_idx}")
        for i in range(adj_vec.forward_stars[tail_vert_idx].size):
            head_vert_idx = adj_vec.forward_stars[tail_vert_idx].vertices[i]
            edge_weight = adj_vec.forward_stars[tail_vert_idx].weights[i]  
            # print(f"{head_vert_idx} : {edge_weight}")          

    end = perf_counter()
    elapsed_time = end - start
    print(f"FSV loop - Elapsed time: {elapsed_time:6.2f} s")

    for tail_vert_idx in range(vertex_count):
        free(adj_vec.forward_stars[tail_vert_idx].vertices)
        free(adj_vec.forward_stars[tail_vert_idx].weights)

    free(adj_vec.forward_stars)