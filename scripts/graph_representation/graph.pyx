# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False

from time import perf_counter

cimport numpy as cnp
from cython.parallel import prange
from libc.stdlib cimport free, malloc
import psutil


cpdef void loop_CSR(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count):
    
    cdef: 
        ssize_t tail_vert_idx, head_vert_idx, ptr
        cnp.float64_t edge_weight

    for tail_vert_idx in range(<size_t>vertex_count):
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
    size_t vertex_count
    ForwardStar* forward_stars

cdef void init_adjacency_vectors(AdjacencyVectors* adjvec, int vertex_count):

    adjvec.vertex_count = <size_t> vertex_count
    adjvec.forward_stars = <ForwardStar*> malloc(vertex_count * sizeof(ForwardStar))

cdef void create_FSV(
    AdjacencyVectors* adjvec,
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int n_jobs) nogil:
    
    cdef:
        size_t i, tail_vert_idx, size, ptr

    for tail_vert_idx in prange(
        adjvec.vertex_count,
        num_threads=n_jobs):
        size = csr_indptr[tail_vert_idx + 1] - csr_indptr[tail_vert_idx]
        adjvec.forward_stars[tail_vert_idx].size = size
        adjvec.forward_stars[tail_vert_idx].vertices = <ssize_t*> malloc(size * sizeof(ssize_t))
        adjvec.forward_stars[tail_vert_idx].weights = <cnp.float64_t*> malloc(size * sizeof(cnp.float64_t))
        for i in range(size):
            ptr = csr_indptr[tail_vert_idx] + i
            adjvec.forward_stars[tail_vert_idx].vertices[i] = csr_indices[ptr]
            adjvec.forward_stars[tail_vert_idx].weights[i] = csr_data[ptr]

cdef void loop_FSV_inner(AdjacencyVectors* adjvec) nogil:
    
    cdef:
        size_t i, tail_vert_idx, head_vert_idx
        cnp.float64_t edge_weight

    for tail_vert_idx in range(adjvec.vertex_count):
        for i in range(adjvec.forward_stars[tail_vert_idx].size):
            head_vert_idx = adjvec.forward_stars[tail_vert_idx].vertices[i]
            edge_weight = adjvec.forward_stars[tail_vert_idx].weights[i] 

cdef void free_adjacency_vectors(AdjacencyVectors* adjvec, int n_jobs) nogil:

    cdef ssize_t tail_vert_idx

    for tail_vert_idx in prange(
        adjvec.vertex_count,
        num_threads=n_jobs):
        free(adjvec.forward_stars[tail_vert_idx].vertices)
        free(adjvec.forward_stars[tail_vert_idx].weights)

    free(adjvec.forward_stars)

cpdef void loop_FSV(
    ssize_t[::1] csr_indptr,
    ssize_t[::1] csr_indices,
    cnp.float64_t[::1] csr_data,
    int vertex_count):

    cdef:
        AdjacencyVectors adjvec
        int n_jobs

    n_jobs = psutil.cpu_count()

    start = perf_counter()

    init_adjacency_vectors(&adjvec, vertex_count)

    end = perf_counter()
    elapsed_time = end - start
    print(f"FSV init - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    create_FSV(&adjvec, csr_indptr, csr_indices, csr_data, n_jobs)

    end = perf_counter()
    elapsed_time = end - start
    print(f"FSV create - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    loop_FSV_inner(&adjvec)

    end = perf_counter()
    elapsed_time = end - start
    print(f"FSV loop - Elapsed time: {elapsed_time:6.2f} s")

    start = perf_counter()

    free_adjacency_vectors(&adjvec, n_jobs)

    end = perf_counter()
    elapsed_time = end - start
    print(f"FSV free - Elapsed time: {elapsed_time:6.2f} s")
