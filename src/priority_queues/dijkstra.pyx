# cython: boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False


# cimport numpy as cnp
import numpy as np

# from priority_queues.pq_bin_heap cimport *
# from priority_queues.commons cimport DTYPE, DTYPE_t, SCANNED, NOT_IN_HEAP





# cpdef cnp.ndarray path_length_from(
#     UITYPE_t[:] csr_indices,
#     UITYPE_t[:] csr_indptr,
#     DTYPE_t[:] edge_weights,
#     UITYPE_t origin_vert,
#     UITYPE_t n_vertices,
#     int n_jobs=-1):
#     """ Compute single-source shortest path (from one vertex to all vertices).

#        Does not return predecessors
#     """

#     cdef:
#         UITYPE_t tail_vert_idx, head_vert_idx, edge_idx  # vertex and edge indices
#         DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
#         BinaryHeap bheap  # binary heap
#         int vert_state  # vertex state
#         int num_threads = n_jobs

#     if num_threads < 1:
#         num_threads = N_THREADS
#     # initialization (the priority queue is filled with all nodes)
#     # all nodes of INFINITY key
#     init_heap(&bheap, n_vertices, num_threads)

#     # the key is set to zero for the origin vertex
#     min_heap_insert(&bheap, origin_vert, 0.)

#     # main loop
#     while bheap.size > 0:
#         tail_vert_idx = extract_min(&bheap)
#         tail_vert_val = bheap.nodes[tail_vert_idx].key
#         # loop on outgoing edges
#         for edge_idx in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
#             head_vert_idx = csr_indices[edge_idx]
#             vert_state = bheap.nodes[head_vert_idx].state
#             if vert_state != SCANNED:
#                 head_vert_val = tail_vert_val + edge_weights[edge_idx]
#                 if vert_state == NOT_IN_HEAP:
#                     min_heap_insert(&bheap, head_vert_idx, head_vert_val)
#                 elif bheap.nodes[head_vert_idx].key > head_vert_val:
#                     decrease_key_from_node_index(&bheap, head_vert_idx, head_vert_val)

#     # copy the results into a numpy array
#     path_lengths = np.zeros(n_vertices, dtype=DTYPE)
#     cdef:
#         int i  # loop counter
#         DTYPE_t[:] path_lengths_view = path_lengths

#     for i in prange(
#         n_vertices, 
#         schedule=guided, 
#         nogil=True, 
#         num_threads=num_threads):
#         path_lengths_view[i] = bheap.nodes[i].key 

#     free_heap(&bheap)  # cleanup

#     return path_lengths


