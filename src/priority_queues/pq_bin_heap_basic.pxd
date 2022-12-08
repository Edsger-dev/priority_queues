
cimport numpy as cnp

from priority_queues.commons cimport DTYPE_t, ElementState


cdef struct Element:
    ElementState state # element state wrt the heap
    ssize_t node_idx   # index of the corresponding node in the tree
    DTYPE_t key        # key value

cdef struct PriorityQueue:
    ssize_t  length    # maximum heap size
    ssize_t  size      # number of elements in the heap
    ssize_t* A         # array storing the binary tree
    Element* Elements  # array storing the elements

cdef void init_pqueue(PriorityQueue*, ssize_t) nogil
cdef void free_pqueue(PriorityQueue*) nogil
cdef void insert(PriorityQueue*, ssize_t, DTYPE_t) nogil
cdef void decrease_key(PriorityQueue*, ssize_t, DTYPE_t) nogil
cdef ssize_t extract_min(PriorityQueue*) nogil

