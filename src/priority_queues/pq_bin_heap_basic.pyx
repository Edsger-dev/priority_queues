# cython: boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False


""" Priority queue based on a minimum binary heap.
    
    Binary heap implemented with a static array.

    Tree elements also stored in a static array.
    
author : François Pacull
email: francois.pacull@architecture-performance.fr
copyright : Architecture & Performance
license : MIT
"""

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport free, malloc

from priority_queues.commons cimport DTYPE_INF, DTYPE_t, SCANNED, IN_HEAP, NOT_IN_HEAP


cdef void init_pqueue(
    PriorityQueue* pqueue,
    size_t length) nogil:
    """Initialize the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t length : length (maximum size) of the heap
    """
    cdef size_t i

    pqueue.length = length
    ##
    pqueue.size = 0
    ##
    # pqueue.size = length
    ##
    pqueue.A = <size_t*> malloc(length * sizeof(size_t))
    pqueue.Elements = <Element*> malloc(length * sizeof(Element))

    for i in range(length):
        ##
        # pqueue.A[i] = i
        # _initialize_element(pqueue, i)
        ##
        pqueue.A[i] = length
        _initialize_element(pqueue, i)
        ##
        # insert(pqueue, i, DTYPE_INF)
        ##

cdef void init_pqueue_insert_all(
    PriorityQueue* pqueue,
    size_t length) nogil:
    """Initialize the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t length : length (maximum size) of the heap
    """
    cdef size_t i

    pqueue.length = length
    pqueue.size = length

    pqueue.A = <size_t*> malloc(length * sizeof(size_t))
    pqueue.Elements = <Element*> malloc(length * sizeof(Element))

    for i in range(length):
        pqueue.A[i] = i
        # insert(pqueue, i, DTYPE_INF)
        pqueue.Elements[i].key = DTYPE_INF
        pqueue.Elements[i].state = IN_HEAP
        pqueue.Elements[i].node_idx = i

cdef inline void _initialize_element(
    PriorityQueue* pqueue,
    size_t element_idx) nogil:
    """Initialize a single element.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    """
    pqueue.Elements[element_idx].key = DTYPE_INF
    pqueue.Elements[element_idx].state = NOT_IN_HEAP
    pqueue.Elements[element_idx].node_idx = pqueue.length


cdef void free_pqueue(
    PriorityQueue* pqueue) nogil:
    """Free the priority queue.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    """
    free(pqueue.A)
    free(pqueue.Elements)

cdef void insert(
    PriorityQueue* pqueue,
    size_t element_idx,
    DTYPE_t key) nogil:
    """Insert an element into the priority queue and reorder the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key : key value of the element

    assumptions
    ===========
    * the element pqueue.Elements[element_idx] is not in the heap
    * its new key is smaller than DTYPE_INF
    """
    cdef size_t node_idx = pqueue.size

    pqueue.size += 1
    pqueue.Elements[element_idx].state = IN_HEAP
    pqueue.Elements[element_idx].node_idx = node_idx
    pqueue.A[node_idx] = element_idx
    _decrease_key_from_node_index(pqueue, node_idx, key)

cdef void decrease_key(
    PriorityQueue* pqueue,
    size_t element_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of a element in the priority queue, 
    given its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t element_idx : index of the element in the element array
    * DTYPE_t key_new : new value of the element key 

    assumption
    ==========
    * pqueue.Elements[idx] is in the heap
    """
    _decrease_key_from_node_index(
        pqueue, 
        pqueue.Elements[element_idx].node_idx, 
        key_new)

cdef size_t extract_min(PriorityQueue* pqueue) nogil:
    """Extract element with min key from the priority queue, 
    and return its element index.

    input
    =====
    * PriorityQueue* pqueue : priority queue

    output
    ======
    * size_t : element index with min key

    assumption
    ==========
    * pqueue.size > 0
    """
    cdef: 
        size_t element_idx = pqueue.A[0]  # min element index
        size_t node_idx = pqueue.size - 1  # last leaf node index

    # exchange the root node with the last leaf node
    _exchange_nodes(pqueue, 0, node_idx)

    # remove this element from the heap
    pqueue.Elements[element_idx].state = SCANNED
    pqueue.Elements[element_idx].node_idx = pqueue.length
    pqueue.A[node_idx] = pqueue.length
    pqueue.size -= 1

    # reorder the tree elements from the root node
    _min_heapify(pqueue, 0)

    return element_idx

cdef inline void _exchange_nodes(
    PriorityQueue* pqueue, 
    size_t node_i,
    size_t node_j) nogil:
    """Exchange two nodes in the heap.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_i: first node index
    * size_t node_j: second node index
    """
    cdef: 
        size_t element_i = pqueue.A[node_i]
        size_t element_j = pqueue.A[node_j]
    
    # exchange element indices in the heap array
    pqueue.A[node_i] = element_j
    pqueue.A[node_j] = element_i

    # exchange node indices in the element array
    pqueue.Elements[element_j].node_idx = node_i
    pqueue.Elements[element_i].node_idx = node_j

    
cdef inline void _min_heapify(
    PriorityQueue* pqueue,
    size_t node_idx) nogil:
    """Re-order sub-tree under a given node (given its node index) 
    until it satisfies the heap property.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    """
    cdef: 
        size_t l, r, i = node_idx, s

    while True:

        l =  2 * i + 1  
        r = l + 1
        
        if (
            (l < pqueue.size) and 
            (pqueue.Elements[pqueue.A[l]].key < pqueue.Elements[pqueue.A[i]].key)
        ):
            s = l
        else:
            s = i

        if (
            (r < pqueue.size) and 
            (pqueue.Elements[pqueue.A[r]].key < pqueue.Elements[pqueue.A[s]].key)
        ):
            s = r

        if s != i:
            _exchange_nodes(pqueue, i, s)
            i = s
        else:
            break
    
cdef inline void _decrease_key_from_node_index(
    PriorityQueue* pqueue,
    size_t node_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of an element in the priority queue, given its tree index.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    * DTYPE_t key_new : new key value

    assumptions
    ===========
    * pqueue.elements[pqueue.A[node_idx]] is in the heap (node_idx < pqueue.size)
    * key_new < pqueue.elements[pqueue.A[node_idx]].key
    """
    cdef:
        size_t i = node_idx, j
        DTYPE_t key_j

    pqueue.Elements[pqueue.A[i]].key = key_new
    while i > 0: 
        j = (i - 1) // 2  
        key_j = pqueue.Elements[pqueue.A[j]].key
        if key_j > key_new:
            _exchange_nodes(pqueue, i, j)
            i = j
        else:
            break
