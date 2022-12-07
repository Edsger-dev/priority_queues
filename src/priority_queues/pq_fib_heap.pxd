
cimport numpy as cnp

ctypedef cnp.int64_t ITYPE_t

from priority_queues.commons cimport DTYPE_t

#cdef enum FibonacciState:
#    SCANNED=1
#    NOT_IN_HEAP=2
#    IN_HEAP=3

cdef struct FibonacciNode:
    ITYPE_t index
    unsigned int rank
    #FibonacciState state
    unsigned int state
    DTYPE_t val
    FibonacciNode* parent
    FibonacciNode* left_sibling
    FibonacciNode* right_sibling
    FibonacciNode* children

ctypedef FibonacciNode* pFibonacciNode

cdef struct FibonacciHeap:
    FibonacciNode* min_node
    pFibonacciNode[100] roots_by_rank  # maximum number of nodes is ~2^100.


cdef void initialize_node(FibonacciNode*, unsigned int, double) nogil
cdef void insert_node(FibonacciHeap*, FibonacciNode*) nogil