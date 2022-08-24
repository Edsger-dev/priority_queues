
from priority_queues.pq_bin_heap cimport (BinaryHeap, init_heap, free_heap, 
    min_heap_insert, peek)
from priority_queues.commons cimport (DTYPE_INF, DTYPE_t, IN_HEAP, NOT_IN_HEAP, 
    SCANNED)


cpdef init_01():

    cdef: 
        BinaryHeap bheap
        size_t l = 4

    init_heap(&bheap, l)
    assert bheap.length == l
    assert bheap.size == 0
    for i in range(l):
        assert bheap.A[i] == bheap.length
        assert bheap.elements[i].key == DTYPE_INF
        assert bheap.elements[i].state == NOT_IN_HEAP
        assert bheap.elements[i].node_idx == bheap.length
    free_heap(&bheap)


cpdef insert_01():

    cdef: 
        BinaryHeap bheap
        DTYPE_t key

    init_heap(&bheap, 4)


    key = 3.0
    min_heap_insert(&bheap, 1, key)
    
    A_ref = [1, 4, 4, 4]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[1].key == key
    assert bheap.elements[1].state == IN_HEAP
    assert bheap.size == 1

    key = 2.0
    min_heap_insert(&bheap, 0, key)

    A_ref = [0, 1, 4, 4]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[0].key == key
    assert bheap.elements[0].state == IN_HEAP
    assert bheap.size == 2

    key = 4.0
    min_heap_insert(&bheap, 3, key)

    A_ref = [0, 1, 3, 4]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[3].key == key
    assert bheap.elements[3].state == IN_HEAP
    assert bheap.size == 3

    key = 1.0
    min_heap_insert(&bheap, 2, key)

    A_ref = [2, 0, 3, 1]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[2].key == key
    assert bheap.elements[2].state == IN_HEAP
    assert bheap.size == 4

    free_heap(&bheap)


cpdef peek_01():

    cdef: 
        BinaryHeap bheap

    init_heap(&bheap, 6)
    min_heap_insert(&bheap, 0, 9.0)
    min_heap_insert(&bheap, 1, 9.0)
    min_heap_insert(&bheap, 2, 9.0)
    min_heap_insert(&bheap, 3, 5.0)
    min_heap_insert(&bheap, 4, 3.0)
    min_heap_insert(&bheap, 5, 1.0)
    assert peek(&bheap) == 1
    free_heap(&bheap)