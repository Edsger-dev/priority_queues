
from priority_queues.pq_bin_heap cimport (BinaryHeap, init_heap, free_heap)
# from priority_queues.commons cimport (DTYPE_INF, IN_HEAP, NOT_IN_HEAP, SCANNED)


cpdef test_init_01():

    cdef: 
        BinaryHeap bheap
        size_t l = 4

    init_heap(&bheap, l)
    # assert bheap.length == l
    # assert bheap.size == 0
    # for i in range(l):
    #     assert bheap.A[i] == bheap.length
    #     assert bheap.elements[i].key == DTYPE_INF
    #     assert bheap.elements[i].state == NOT_IN_HEAP
    #     assert bheap.elements[i].node_idx == bheap.length
    free_heap(&bheap)