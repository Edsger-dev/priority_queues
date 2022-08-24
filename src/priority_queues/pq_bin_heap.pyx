# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, cython: cdivision=True, initializedcheck=False


""" Priority queue based on a minimum binary heap.
    
    Binary heap implemented with a static array.

    Tree elements also stored in a static array.
"""

cimport cython

import numpy as np

cimport numpy as cnp
from libc.stdlib cimport free, malloc

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t INFINITY = <DTYPE_t>np.finfo(dtype=DTYPE).max

cdef enum ElementState:
   SCANNED = 1
   NOT_IN_HEAP = 2
   IN_HEAP = 3

cdef struct Element:
    DTYPE_t key
    ElementState state
    ssize_t node_idx

cdef struct BinaryHeap:
    ssize_t length  # number of elements in the array
    ssize_t size  # number of elements in the heap
    ssize_t* A  # array storing the binary tree
    Element* elements  # array storing the elements


cdef void init_heap(
    BinaryHeap* bheap,
    ssize_t length) nogil:
    """Initialize the binary heap.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * ssize_t length : length (maximum size) of the heap
    """
    cdef ssize_t i

    bheap.length = length
    bheap.size = 0
    bheap.A = <ssize_t*> malloc(length * sizeof(ssize_t))
    bheap.elements = <Element*> malloc(length * sizeof(Element))
    for i in range(length):
        bheap.A[i] = length
        _initialize_element(bheap, i)


cdef void _initialize_element(
    BinaryHeap* bheap,
    ssize_t element_idx) nogil:
    """Initialize a single element.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * ssize_t element_idx : index of the element in the element array
    """
    bheap.elements[element_idx].key = INFINITY
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
    ssize_t element_idx,
    DTYPE_t key) nogil:
    """Insert an element into the heap and reorder the heap.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * ssize_t element_idx : index of the element in the element array
    * DTYPE_t key : key value of the element

    assumptions
    ===========
    * the element bheap.elements[element_idx] is not in the heap
    * its new key is smaller than INFINITY
    """
    cdef ssize_t node_idx = bheap.size

    bheap.size += 1
    bheap.elements[element_idx].state = IN_HEAP
    bheap.elements[element_idx].node_idx = node_idx
    bheap.A[node_idx] = element_idx
    _decrease_key_from_node_index(bheap, node_idx, key)


cdef void decrease_key_from_element_index(
    BinaryHeap* bheap, 
    ssize_t element_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of a element in the heap, given its element index.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * ssize_t element_idx : index of the element in the element array
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


cdef ssize_t extract_min(BinaryHeap* bheap) nogil:
    """Extract element with min keay from the heap, 
    and return its element index.

    input
    =====
    * BinaryHeap* bheap : binary heap

    output
    ======
    * ssize_t : element index with min key

    assumption
    ==========
    * bheap.size > 0
    """
    cdef: 
        ssize_t element_idx = bheap.A[0]  # min element index
        ssize_t node_idx = bheap.size - 1  # last leaf node index

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


cdef DTYPE_t extract_min_key(BinaryHeap* bheap) nogil:
    """Extract element with min keay from the heap, 
    and return its element key.

    input
    =====
    * BinaryHeap* bheap : binary heap

    output
    ======
    * DTYPE_t : min element key

    assumption
    ==========
    * bheap.size > 0
    """
    cdef: 
        ssize_t element_idx = bheap.A[0]  # min element index
        ssize_t node_idx = bheap.size - 1  # last leaf node index
        DTYPE_t min_key

    # exchange the root node with the last leaf node
    _exchange_nodes(bheap, 0, node_idx)

    # remove this element from the heap
    min_key = bheap.elements[element_idx].key
    bheap.elements[element_idx].state = SCANNED
    bheap.elements[element_idx].node_idx = bheap.length
    bheap.A[node_idx] = bheap.length
    bheap.size -= 1

    # reorder the tree elements from the root node
    _min_heapify(bheap, 0)

    return min_key


cdef ssize_t _parent(ssize_t node_idx) nogil:
    """Get the parent node index.

    input
    =====
    ssize_t node_idx: node index

    assumption
    ==========
    * node_idx > 0
    """
    return (node_idx - 1) // 2


cdef ssize_t _left_child(ssize_t node_idx) nogil:
    """Returns the left child node.

    input
    =====
    * ssize_t node_idx : node index
    """
    return 2 * node_idx + 1


cdef ssize_t _right_child(ssize_t node_idx) nogil:
    """Returns the right child node.

    input
    =====
    * ssize_t node_idx : tree index
    """
    return 2 * (node_idx + 1)


cdef void _exchange_nodes(
    BinaryHeap* bheap, 
    ssize_t node_i,
    ssize_t node_j) nogil:
    """Exchange two nodes in the heap.

    input
    =====
    * BinaryHeap* bheap: binary heap
    * ssize_t node_i: first node index
    * ssize_t node_j: second node index
    """
    cdef: 
        ssize_t element_i = bheap.A[node_i]
        ssize_t element_j = bheap.A[node_j]
    
    # exchange element indices in the heap array
    bheap.A[node_i] = element_j
    bheap.A[node_j] = element_i

    # exchange node indices in the element array
    bheap.elements[element_j].node_idx = node_i
    bheap.elements[element_i].node_idx = node_j


cdef void _min_heapify(
    BinaryHeap* bheap,
    ssize_t node_idx) nogil:
    """Re-order sub-tree under a given node (given its node index) 
    until it satisfies the heap property.

    Note that this function is recursive.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * ssize_t node_idx : tree index
    """
    cdef: 
        ssize_t l, r, s = node_idx

    l = _left_child(s)
    r = _right_child(s)

    if (l < bheap.size) and (bheap.elements[bheap.A[l]].key < bheap.elements[bheap.A[s]].key):
        s = l

    if (r < bheap.size) and (bheap.elements[bheap.A[r]].key < bheap.elements[bheap.A[s]].key):
        s = r

    if s != node_idx:
        _exchange_nodes(bheap, node_idx, s)
        _min_heapify(bheap, s)


cdef void _decrease_key_from_node_index(
    BinaryHeap* bheap,
    ssize_t node_idx, 
    DTYPE_t key_new) nogil:
    """Decrease the key of an element in the heap, given its tree index.

    input
    =====
    * BinaryHeap* bheap : binary heap
    * ssize_t node_idx : node index

    assumptions
    ===========
    * bheap.elements[bheap.A[i]] is in the heap (i < bheap.size)
    * key_new < bheap.elements[bheap.A[i]].key
    """
    cdef:
        ssize_t i = node_idx
        ssize_t j
        DTYPE_t key_j

    bheap.elements[bheap.A[i]].key = key_new
    while i > 0: 
        j = _parent(i)
        key_j = bheap.elements[bheap.A[j]].key
        if key_j > key_new:
            _exchange_nodes(bheap, i, j)
            i = j
        else:
            break