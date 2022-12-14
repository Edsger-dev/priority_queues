
cimport numpy as cnp

from priority_queues.commons cimport DTYPE_t, ElementState


cdef struct Element:
    DTYPE_t key
    ElementState state
    size_t node_idx

cdef struct PriorityQueue:
    size_t length  # number of elements in the array
    size_t size  # number of elements in the heap
    size_t* A  # array storing the binary tree
    Element* Elements  # array storing the elements

cdef void init_heap(PriorityQueue*, size_t) nogil
cdef void init_heap_para(PriorityQueue*, size_t, int) nogil
cdef void free_heap(PriorityQueue*) nogil
cdef void min_heap_insert(PriorityQueue*, size_t, DTYPE_t) nogil
cdef DTYPE_t peek(PriorityQueue*) nogil
cdef size_t extract_min(PriorityQueue*) nogil
cdef bint is_empty(PriorityQueue*) nogil
cdef void decrease_key_from_element_index(PriorityQueue*, size_t, DTYPE_t) nogil
cdef cnp.ndarray copy_keys_to_numpy(PriorityQueue*, size_t)
cdef cnp.ndarray copy_keys_to_numpy_para(PriorityQueue*, int, int)