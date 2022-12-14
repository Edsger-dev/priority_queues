# cython: boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False

cimport numpy as cnp

####
# from libc.stdio cimport printf, fflush, stdout
####
from libc.stdlib cimport free, malloc
import numpy as np

from priority_queues.commons cimport (
    DTYPE, DTYPE_INF, IN_HEAP, NOT_IN_HEAP, SCANNED, DTYPE_t, ElementState)

cimport priority_queues.pq_bin_heap_basic as bhb
cimport priority_queues.pq_3ary_heap as threeh
cimport priority_queues.pq_4ary_heap as fourh
cimport priority_queues.pq_bin_heap_length as bhl

from priority_queues.pq_bin_heap cimport (
    PriorityQueue,
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


cpdef cnp.ndarray path_length_from_3ary(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        threeh.PriorityQueue pqueue 
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in


    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        threeh.init_heap(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        threeh.insert(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = threeh.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        threeh.insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        threeh.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    cdef:
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(<size_t>vertex_count):
            path_lengths_view[i] = pqueue.Elements[i].key

    # cleanup
    threeh.free_heap(&pqueue)  

    return path_lengths



cpdef cnp.ndarray path_length_from_4ary(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        fourh.PriorityQueue pqueue 
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in


    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        fourh.init_heap(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        fourh.insert(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = fourh.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        fourh.insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        fourh.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    cdef:
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(<size_t>vertex_count):
            path_lengths_view[i] = pqueue.Elements[i].key

    # cleanup
    fourh.free_heap(&pqueue)  

    return path_lengths



cpdef cnp.ndarray path_length_from_bin_basic(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        bhb.PriorityQueue pqueue 
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in
        ####
        # size_t line = 0
        ####


    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        bhb.init_pqueue(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        bhb.insert(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = bhb.extract_min(&pqueue)
            ####
            # printf("%9d ", line)
            # printf("SCANNED %9d IN_HEAP ", tail_vert_idx)
            ####
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        bhb.insert(&pqueue, head_vert_idx, head_vert_val)
                        ####
                        # printf("%9d ", head_vert_idx)
                        ####
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        bhb.decrease_key(&pqueue, head_vert_idx, head_vert_val)
            ####
            # printf("\n")
            # line += 1
            ####
    ####
    # fflush(stdout)
    ####

    # copy the results into a numpy array
    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    ##
    cdef:
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(<size_t>vertex_count):
            path_lengths_view[i] = pqueue.Elements[i].key

    # cleanup
    bhb.free_pqueue(&pqueue)  

    return path_lengths



cpdef cnp.ndarray path_length_from_bin_basic_insert_all(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

        insert all nodes in the init.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        bhb.PriorityQueue pqueue 
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in


    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and IN_HEAP state
        bhb.init_pqueue_insert_all(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        bhb.decrease_key(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = bhb.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if pqueue.Elements[head_vert_idx].key > head_vert_val:
                        bhb.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    cdef:
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(<size_t>vertex_count):
            path_lengths_view[i] = pqueue.Elements[i].key

    # cleanup
    bhb.free_pqueue(&pqueue)  

    return path_lengths


cpdef cnp.ndarray path_length_from_bin(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        PriorityQueue pqueue  # binary heap
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in

    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        init_heap(&pqueue, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        min_heap_insert(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        min_heap_insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        decrease_key_from_element_index(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = copy_keys_to_numpy(&pqueue, <size_t>vertex_count)

    # cleanup
    free_heap(&pqueue)  

    return path_lengths


cpdef cnp.ndarray path_length_from_fib(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert,
    int vertex_count):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a Fibonacci heap.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        FibonacciHeap heap
        FibonacciNode *v
        FibonacciNode *current_node
        FibonacciNode *nodes = <FibonacciNode*> malloc(vertex_count * sizeof(FibonacciNode))
        unsigned int vert_state

    # initialization of the heap elements 
    # all nodes have INFINITY key and NOT_IN_HEAP state
    for idx in range(<size_t>vertex_count):
        initialize_node(&nodes[idx], <unsigned int>idx, <double>DTYPE_INF)

    # initialization of the heap
    heap.min_node = NULL

    # the key is set to zero for the origin vertex,
    # which is inserted into the heap
    nodes[origin_vert].val = 0.0
    insert_node(&heap, &nodes[origin_vert])

    while heap.min_node:

        v = remove_min(&heap)
        v.state = SCANNED
        tail_vert_idx = <size_t>v.index
        tail_vert_val = v.val

        # loop on outgoing edges
        for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
            head_vert_idx = <size_t>csr_indices[idx]
            current_node = &nodes[head_vert_idx]
            vert_state = current_node.state
            if vert_state != SCANNED:
                head_vert_val = tail_vert_val + csr_data[idx]
                if vert_state == NOT_IN_HEAP:
                    current_node.state = IN_HEAP
                    current_node.val = head_vert_val
                    insert_node(&heap, current_node)
                elif current_node.val > head_vert_val:
                    decrease_val(&heap, current_node, head_vert_val)

    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)
        
    for idx in range(vertex_count):
        path_lengths[idx] = nodes[idx].val
    free(nodes)

    return path_lengths


cpdef cnp.ndarray path_length_from_bhl(
    cnp.uint32_t[::1] csr_indices,
    cnp.uint32_t[::1] csr_indptr,
    DTYPE_t[::1] csr_data,
    int origin_vert_in,
    int vertex_count,
    int heap_length):
    """ Compute single-source shortest path (from one vertex to all vertices)
        using a priority queue based on a bin heap.

       Does not return predecessors.
    """

    cdef:
        size_t tail_vert_idx, head_vert_idx, idx  # indices
        DTYPE_t tail_vert_val, head_vert_val  # vertex travel times
        bhl.PriorityQueue pqueue 
        ElementState vert_state  # vertex state
        size_t origin_vert = <size_t>origin_vert_in


    with nogil:

        # initialization of the heap elements 
        # all nodes have INFINITY key and NOT_IN_HEAP state
        bhl.init_pqueue(&pqueue, <size_t>heap_length, <size_t>vertex_count)

        # the key is set to zero for the origin vertex,
        # which is inserted into the heap
        bhl.insert(&pqueue, origin_vert, 0.0)

        # main loop
        while pqueue.size > 0:
            tail_vert_idx = bhl.extract_min(&pqueue)
            tail_vert_val = pqueue.Elements[tail_vert_idx].key

            # loop on outgoing edges
            for idx in range(<size_t>csr_indptr[tail_vert_idx], <size_t>csr_indptr[tail_vert_idx + 1]):
                head_vert_idx = <size_t>csr_indices[idx]
                vert_state = pqueue.Elements[head_vert_idx].state
                if vert_state != SCANNED:
                    head_vert_val = tail_vert_val + csr_data[idx]
                    if vert_state == NOT_IN_HEAP:
                        bhl.insert(&pqueue, head_vert_idx, head_vert_val)
                    elif pqueue.Elements[head_vert_idx].key > head_vert_val:
                        bhl.decrease_key(&pqueue, head_vert_idx, head_vert_val)

    # copy the results into a numpy array
    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    cdef:
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(<size_t>vertex_count):
            path_lengths_view[i] = pqueue.Elements[i].key

    # cleanup
    bhl.free_pqueue(&pqueue)  

    return path_lengths


cpdef void coo_tocsr(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.float64_t[::1] Ax,
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bj,
    cnp.float64_t[::1] Bx) nogil:

    cdef:
        size_t i, row, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bj.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Ai[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[n_vert] = <cnp.uint32_t>n_edge 

    for i in range(n_edge):
        row  = <size_t>Ai[i]
        dest = <size_t>Bp[row]
        Bj[dest] = Aj[i]
        Bx[dest] = Ax[i]
        Bp[row] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp


cpdef void coo_tocsc(
    cnp.uint32_t [::1] Ai,
    cnp.uint32_t [::1] Aj,
    cnp.float64_t[::1] Ax,
    cnp.uint32_t [::1] Bp,
    cnp.uint32_t [::1] Bi,
    cnp.float64_t[::1] Bx) nogil:

    cdef:
        size_t i, col, dest
        size_t n_vert = <size_t>(Bp.shape[0] - 1)
        size_t n_edge = <size_t>Bi.shape[0]
        cnp.uint32_t temp, cumsum, last

    for i in range(n_edge):
        Bp[<size_t>Aj[i]] += 1

    cumsum = 0
    for i in range(n_vert):
        temp = Bp[i]
        Bp[i] = cumsum
        cumsum += temp
    Bp[<size_t>n_vert] = <cnp.uint32_t>n_edge 

    for i in range(n_edge):
        col  = <size_t>Aj[i]
        dest = <size_t>Bp[col]
        Bi[dest] = Ai[i]
        Bx[dest] = Ax[i]
        Bp[col] += 1

    last = 0
    for i in range(n_vert + 1):
        temp = Bp[i]
        Bp[i] = last
        last = temp