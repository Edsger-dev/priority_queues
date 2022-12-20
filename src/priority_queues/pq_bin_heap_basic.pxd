
cimport numpy as cnp

from priority_queues.commons cimport DTYPE_t, ElementState


cdef struct Element:
    ElementState state # element state wrt the heap
    size_t node_idx   # index of the corresponding node in the tree
    DTYPE_t key        # key value

cdef struct PriorityQueue:
    size_t  length    # maximum heap size
    size_t  size      # number of elements in the heap
    size_t* A         # array storing the binary tree
    Element* Elements  # array storing the elements

cdef void init_pqueue(PriorityQueue*, size_t) nogil
cdef void init_pqueue_insert_all(PriorityQueue*, size_t) nogil
cdef void free_pqueue(PriorityQueue*) nogil
cdef void insert(PriorityQueue*, size_t, DTYPE_t) nogil
cdef void decrease_key(PriorityQueue*, size_t, DTYPE_t) nogil
cdef size_t extract_min(PriorityQueue*) nogil

