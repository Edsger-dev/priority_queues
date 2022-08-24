from priority_queues.commons cimport (ElementState, DTYPE_t)


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
cdef void free_heap(BinaryHeap*) nogil

