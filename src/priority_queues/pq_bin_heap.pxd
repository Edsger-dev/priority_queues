
cimport numpy as cnp

from priority_queues.commons cimport DTYPE_t, ElementState


cdef struct Element:
    DTYPE_t key
    ElementState state
    ssize_t node_idx

cdef struct BinaryHeap:
    ssize_t length  # number of elements in the array
    ssize_t size  # number of elements in the heap
    ssize_t* A  # array storing the binary tree
    Element* elements  # array storing the elements

cdef void init_heap(BinaryHeap*, ssize_t) nogil
cdef void init_heap_para(BinaryHeap*, ssize_t, int) nogil
cdef void free_heap(BinaryHeap*) nogil
cdef void min_heap_insert(BinaryHeap*, ssize_t, DTYPE_t) nogil
cdef DTYPE_t peek(BinaryHeap*) nogil
cdef ssize_t extract_min(BinaryHeap*) nogil
cdef bint is_empty(BinaryHeap*) nogil
cdef void decrease_key_from_element_index(BinaryHeap*, ssize_t, DTYPE_t) nogil
cdef cnp.ndarray copy_keys_to_numpy_para(BinaryHeap*, int, int)