# cython: boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False

cimport numpy as cnp

# from libc.stdio cimport printf
from libc.stdlib cimport free, malloc
import numpy as np

from priority_queues.commons cimport (
    DTYPE, DTYPE_INF, NOT_IN_HEAP, SCANNED, DTYPE_t)

from priority_queues.pq_bin_heap cimport (
    BinaryHeap,
    decrease_key_from_element_index,
    extract_min, free_heap,
    init_heap,
    min_heap_insert, 
    copy_keys_to_numpy)

from priority_queues.pq_fib_heap cimport (
    FibonacciNode,
    FibonacciHeap,
    initialize_node,
    insert_node,
    remove_min,
    decrease_val)


cpdef cnp.ndarray path_length_from_bin(
    ssize_t[::1] csr_indices,
    ssize_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

       Does not return predecessors.
    """

    cdef:
        ssize_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        BinaryHeap bheap  # binary heap
        int vert_state  # vertex state
        ssize_t origin_vert = <ssize_t>origin_vert_in

    # initialization of the heap elements 
    # all nodes have INFINITY key and NOT_IN_HEAP state
    init_heap(&bheap, <ssize_t>vertex_count)

    with nogil:

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        min_heap_insert(&bheap, origin_vert, 0.0)

        # main loop
        while bheap.size > 0:
            tail_vert_idx = extract_min(&bheap)
            # printf("%d\n", tail_vert_idx)
            tail_vert_val = bheap.elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = csr_indices[idx]
                vert_state = bheap.elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        min_heap_insert(&bheap, head_vert_idx, head_vert_val)
                    elif bheap.elements[head_vert_idx].key > head_vert_val:
                        decrease_key_from_element_index(&bheap, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = copy_keys_to_numpy(&bheap, <ssize_t>vertex_count)

    # cleanup
    free_heap(&bheap)  

    return path_lengths


cpdef cnp.ndarray path_length_from_fib(
    ssize_t[::1] csr_indices,
    ssize_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a Fibonacci heap.

       Does not return predecessors.
    """

    cdef:
        ssize_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        FibonacciHeap heap
        FibonacciNode *v
        FibonacciNode *current_node
        FibonacciNode *nodes = <FibonacciNode*> malloc(vertex_count * sizeof(FibonacciNode))
        unsigned int vert_state

    # initialization of the heap elements 
    # all nodes have INFINITY key and NOT_IN_HEAP state
    for idx in range(<ssize_t>vertex_count):
        initialize_node(&nodes[idx], <unsigned int>idx, <double>DTYPE_INF)

    # initialization of the heap
    heap.min_node = NULL

    # the key is set to zero for the origin vertex,
    # which is inserted into the heap
    nodes[origin_vert].val = 0.0
    insert_node(&heap, &nodes[origin_vert])

    while heap.min_node:

        v = remove_min(&heap)
        v.state = 1  # SCANNED
        tail_vert_idx = <ssize_t>v.index
        tail_vert_val = v.val

        # loop on outgoing edges
        for idx in range(csr_indptr[tail_vert_idx], csr_indptr[tail_vert_idx + 1]):
            head_vert_idx = csr_indices[idx]
            current_node = &nodes[head_vert_idx]
            vert_state = current_node.state
            if vert_state != 1:
                head_vert_val = tail_vert_val + csr_data[idx]
                if vert_state == 2:
                    current_node.state = 3
                    current_node.val = head_vert_val
                    insert_node(&heap, current_node)
                elif current_node.val > head_vert_val:
                    decrease_val(&heap, current_node, head_vert_val)

    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)
        
    for idx in range(vertex_count):
        path_lengths[idx] = nodes[idx].val
    free(nodes)

    return path_lengths