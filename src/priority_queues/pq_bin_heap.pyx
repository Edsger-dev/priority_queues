# cython: boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False


""" Priority queue based on a minimum binary heap.
    
    Binary heap implemented with a static array.

    Tree elements also stored in a static array.
    
author : François Pacull
copyright : Architecture & Performance
license : MIT
"""

cimport numpy as cnp
from cython.parallel import prange


# from libc.stdio cimport printf

from libc.stdlib cimport free, malloc

from priority_queues.commons cimport (DTYPE, DTYPE_INF, IN_HEAP, N_THREADS,
                                      NOT_IN_HEAP, SCANNED, DTYPE_t)


cdef void init_heap(
    BinaryHeap* bheap,
    size_t length) nogil:
    """Initialize the binary heap.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t length : length (maximum size) of the heap
    """
    cdef size_t i

    bheap.length = length
    bheap.size = 0
    bheap.A = <size_t*> malloc(length * sizeof(size_t))
    bheap.elements = <Element*> malloc(length * sizeof(Element))

    for i in range(length):
        bheap.A[i] = length
        _initialize_element(bheap, i)


cdef void init_heap_para(
    BinaryHeap* bheap,
    size_t length,
    int num_threads) nogil:
    """Initialize the binary heap with a parallel loop.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t length : length (maximum size) of the heap
    * int num_threads :  number of threads for the parallel job
    """
    cdef size_t i

    bheap.length = length
    bheap.size = 0
    bheap.A = <size_t*> malloc(length * sizeof(size_t))
    bheap.elements = <Element*> malloc(length * sizeof(Element))

    for i in prange(
        length, 
        schedule='static', 
        nogil=True, 
        num_threads=num_threads):
        bheap.A[i] = length
        _initialize_element(bheap, i)
        # bheap.elements[i].key = DTYPE_INF
        # bheap.elements[i].state = NOT_IN_HEAP
        # bheap.elements[i].node_idx = length


cdef void _initialize_element(
    BinaryHeap* bheap,
    size_t element_idx) nogil:
    """Initialize a single element.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t element_idx : index of the element in the element array
    """
    bheap.elements[element_idx].key = DTYPE_INF
    bheap.elements[element_idx].state = NOT_IN_HEAP
    bheap.elements[element_idx].node_idx = bheap.length


cdef void free_heap(
    BinaryHeap* bheap) nogil:
    """Free the binary heap.

    input
    =====
    * BinaryHeap* bheap : binary heap
    """
    free(bheap.A)
    free(bheap.elements)


cdef void min_heap_insert(
    BinaryHeap* bheap,
    size_t element_idx,
    DTYPE_t key) nogil:
    """Insert an element into the heap and reorder the heap.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key : key value of the element

    assumptions
    ===========
    * the element bheap.elements[element_idx] is not in the heap
    * its new key is smaller than DTYPE_INF
    """
    cdef size_t node_idx = bheap.size

    bheap.size += 1
    bheap.elements[element_idx].state = IN_HEAP
    bheap.elements[element_idx].node_idx = node_idx
    bheap.A[node_idx] = element_idx
    _decrease_key_from_node_index(bheap, node_idx, key)


cdef void decrease_key_from_element_index(
    BinaryHeap* bheap, 
    size_t element_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of a element in the heap, given its element index.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key_new : new value of the element key 

    assumption
    ==========
    * bheap.elements[idx] is in the heap
    """
    _decrease_key_from_node_index(
        bheap, 
        bheap.elements[element_idx].node_idx, 
        key_new)


cdef DTYPE_t peek(BinaryHeap* bheap) nogil:
    """Find heap min key.

    input
    =====
    * BinaryHeap* bheap : binary heap

    output
    ======
    * DTYPE_t : key value of the min element

    assumption
    ==========
    * bheap.size > 0
    * heap is heapified
    """
    return bheap.elements[bheap.A[0]].key


cdef bint is_empty(BinaryHeap* bheap) nogil:
    """Check whether the heap is empty.

    input
    =====
    * BinaryHeap* bheap : binary heap 
    """
    cdef bint isempty = 0

    if bheap.size == 0:
        isempty = 1

    return isempty


cdef size_t extract_min(BinaryHeap* bheap) nogil:
    """Extract element with min keay from the heap, 
    and return its element index.

    input
    =====
    * BinaryHeap* bheap : binary heap

    output
    ======
    * size_t : element index with min key

    assumption
    ==========
    * bheap.size > 0
    """
    cdef: 
        size_t element_idx = bheap.A[0]  # min element index
        size_t node_idx = bheap.size - 1  # last leaf node index

    # printf("%d\n", bheap.size)

    # exchange the root node with the last leaf node
    _exchange_nodes(bheap, 0, node_idx)

    # remove this element from the heap
    bheap.elements[element_idx].state = SCANNED
    bheap.elements[element_idx].node_idx = bheap.length
    bheap.A[node_idx] = bheap.length
    bheap.size -= 1

    # reorder the tree elements from the root node
    _min_heapify(bheap, 0)

    return element_idx

cdef void _exchange_nodes(
    BinaryHeap* bheap, 
    size_t node_i,
    size_t node_j) nogil:
    """Exchange two nodes in the heap.

    input
    =====
    * BinaryHeap* bheap: binary heap
    * size_t node_i: first node index
    * size_t node_j: second node index
    """
    cdef: 
        size_t element_i = bheap.A[node_i]
        size_t element_j = bheap.A[node_j]
    
    # exchange element indices in the heap array
    bheap.A[node_i] = element_j
    bheap.A[node_j] = element_i

    # exchange node indices in the element array
    bheap.elements[element_j].node_idx = node_i
    bheap.elements[element_i].node_idx = node_j


cdef void _min_heapify(
    BinaryHeap* bheap,
    size_t node_idx) nogil:
    """Re-order sub-tree under a given node (given its node index) 
    until it satisfies the heap property.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t node_idx : node index
    """
    cdef: 
        size_t l, r, i = node_idx, s
        # DTYPE_t key_i, key_l, key_s

    while True:

        l =  2 * i + 1  
        # l = (i << 1) + 1
        r = l + 1
        
        # key_i = bheap.elements[bheap.A[i]].key
        # if (l < bheap.size):
        #     key_l = bheap.elements[bheap.A[l]].key
        #     if (key_l < key_i):
        #         s = l
        #         key_s = key_l
        #     else:
        #         s = i
        #         key_s = key_i
        # else:
        #     s = i
        #     key_s = key_i

        # if (
        #     (r < bheap.size) and 
        #     (bheap.elements[bheap.A[r]].key < key_s)
        # ):
        #     s = r

        # key_i = bheap.elements[bheap.A[i]].key
        # s = i
        # key_s = key_i
        # if (r < bheap.size):
        #     key_r = bheap.elements[bheap.A[r]].key
        #     if (key_r < key_i):
        #         s = r
        #         key_s = key_r
        #     if (bheap.elements[bheap.A[l]].key < key_s):
        #         s = l
        # else:
        #     if (
        #         (l < bheap.size) and 
        #         (bheap.elements[bheap.A[l]].key < bheap.elements[bheap.A[s]].key)
        #     ):
        #         s = l


        # optim
        s = i
        if (r < bheap.size):
            if (bheap.elements[bheap.A[r]].key < bheap.elements[bheap.A[s]].key):
                s = r
            if (bheap.elements[bheap.A[l]].key < bheap.elements[bheap.A[s]].key):
                s = l
        else:
            if (
                (l < bheap.size) and 
                (bheap.elements[bheap.A[l]].key < bheap.elements[bheap.A[s]].key)
            ):
                s = l

        # if (
        #     (l < bheap.size) and 
        #     (bheap.elements[bheap.A[l]].key < bheap.elements[bheap.A[i]].key)
        # ):
        #     s = l
        # else:
        #     s = i

        # if (
        #     (r < bheap.size) and 
        #     (bheap.elements[bheap.A[r]].key < bheap.elements[bheap.A[s]].key)
        # ):
        #     s = r

        if s != i:
            _exchange_nodes(bheap, i, s)
            i = s
        else:
            break
        

cdef void _decrease_key_from_node_index(
    BinaryHeap* bheap,
    size_t node_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of an element in the heap, given its tree index.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * size_t node_idx : node index
    * DTYPE_t key_new : new key value

    assumptions
    ===========
    * bheap.elements[bheap.A[node_idx]] is in the heap (node_idx < bheap.size)
    * key_new < bheap.elements[bheap.A[node_idx]].key
    """
    cdef:
        size_t i = node_idx, j
        DTYPE_t key_j

    bheap.elements[bheap.A[i]].key = key_new
    while i > 0: 
        j = (i - 1) // 2  
        # j = (i - 1) >> 1
        key_j = bheap.elements[bheap.A[j]].key
        if key_j > key_new:
            _exchange_nodes(bheap, i, j)
            i = j
        else:
            break

cdef cnp.ndarray copy_keys_to_numpy_para(
    BinaryHeap* bheap,
    int vertex_count,
    int num_threads
):
    """Copy the keys into a numpy array, in parallel.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * int vertex_count : vertex count
    * int num_threads : number of threads for the parallel job

    output
    ======
    * cnp.ndarray : NumPy array with all the keys
    """

    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    cdef:
        int i  # loop counter
        DTYPE_t[::1] path_lengths_view = path_lengths
        

    for i in prange(
        vertex_count, 
        schedule='static', 
        nogil=True, 
        num_threads=num_threads):
        path_lengths_view[i] = bheap.elements[i].key

    return path_lengths


cdef cnp.ndarray copy_keys_to_numpy(
    BinaryHeap* bheap,
    size_t vertex_count
):
    """Copy the keys into a numpy array.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * int vertex_count : vertex count
    * int num_threads : number of threads for the parallel job

    output
    ======
    * cnp.ndarray : NumPy array with all the keys
    """

    path_lengths = cnp.ndarray(vertex_count, dtype=DTYPE)

    cdef:
        size_t i  # loop counter
        DTYPE_t[::1] path_lengths_view = path_lengths

    with nogil:

        for i in range(vertex_count):
            path_lengths_view[i] = bheap.elements[i].key

    return path_lengths