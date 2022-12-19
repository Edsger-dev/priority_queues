

import numpy as np

from priority_queues.commons cimport (
    DTYPE, DTYPE_INF, IN_HEAP, NOT_IN_HEAP,
    SCANNED, DTYPE_t)
from priority_queues.pq_bin_heap cimport (
    PriorityQueue,
    decrease_key_from_element_index,
    extract_min, free_heap, init_heap,
    init_heap_para, is_empty,
    min_heap_insert, peek)


cpdef init_01():

    cdef: 
        PriorityQueue pqueue
        size_t l = 4

    init_heap(&pqueue, l)

    assert pqueue.length == l
    assert pqueue.size == 0
    for i in range(l):
        assert pqueue.A[i] == pqueue.length
        assert pqueue.Elements[i].key == DTYPE_INF
        assert pqueue.Elements[i].state == NOT_IN_HEAP
        assert pqueue.Elements[i].node_idx == pqueue.length

    free_heap(&pqueue)


cpdef init_02():

    cdef: 
        PriorityQueue pqueue
        size_t l = 40

    init_heap_para(&pqueue, l, 4)

    assert pqueue.length == l
    assert pqueue.size == 0
    for i in range(l):
        assert pqueue.A[i] == pqueue.length
        assert pqueue.Elements[i].key == DTYPE_INF
        assert pqueue.Elements[i].state == NOT_IN_HEAP
        assert pqueue.Elements[i].node_idx == pqueue.length

    free_heap(&pqueue)


cpdef insert_01():
    """ Testing a single insertion into an empty binary heap 
    of length 1.
    """

    cdef: 
        PriorityQueue pqueue
        DTYPE_t key

    init_heap(&pqueue, 1)
    assert pqueue.length == 1
    key = 1.0
    min_heap_insert(&pqueue, 0, key)
    assert pqueue.size == 1
    assert pqueue.A[0] == 0
    assert pqueue.Elements[0].key == key
    assert pqueue.Elements[0].state == IN_HEAP
    assert pqueue.Elements[0].node_idx == 0

    free_heap(&pqueue)


cpdef insert_02():

    cdef: 
        PriorityQueue pqueue
        DTYPE_t key

    init_heap(&pqueue, 4)

    elem_idx = 1
    key = 3.0
    min_heap_insert(&pqueue, elem_idx, key)
    A_ref = [1, 4, 4, 4]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[elem_idx].key == key
    assert pqueue.Elements[elem_idx].state == IN_HEAP
    assert pqueue.Elements[1].node_idx == 0
    assert pqueue.size == 1

    elem_idx = 0
    key = 2.0
    min_heap_insert(&pqueue, elem_idx, key)
    A_ref = [0, 1, 4, 4]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[elem_idx].key == key
    assert pqueue.Elements[elem_idx].state == IN_HEAP
    assert pqueue.Elements[0].node_idx == 0
    assert pqueue.Elements[1].node_idx == 1
    assert pqueue.size == 2

    elem_idx = 3
    key = 4.0
    min_heap_insert(&pqueue, elem_idx, key)
    A_ref = [0, 1, 3, 4]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[elem_idx].key == key
    assert pqueue.Elements[elem_idx].state == IN_HEAP
    assert pqueue.Elements[0].node_idx == 0
    assert pqueue.Elements[1].node_idx == 1
    assert pqueue.Elements[3].node_idx == 2
    assert pqueue.size == 3

    elem_idx = 2
    key = 1.0
    min_heap_insert(&pqueue, elem_idx, key)
    A_ref = [2, 0, 3, 1]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
    assert pqueue.Elements[2].key == key
    assert pqueue.Elements[2].state == IN_HEAP
    assert pqueue.Elements[0].node_idx == 1
    assert pqueue.Elements[1].node_idx == 3
    assert pqueue.Elements[2].node_idx == 0
    assert pqueue.Elements[3].node_idx == 2
    assert pqueue.size == 4

    free_heap(&pqueue)

cpdef insert_03(n=4):
    """ Inserting nodes with identical keys.
    """
    cdef: 
        PriorityQueue pqueue
        size_t i
        DTYPE_t key = 1.0

    init_heap(&pqueue, n)
    for i in range(n):
        min_heap_insert(&pqueue, i, key)
    for i in range(n):
        assert pqueue.A[i] == i

    free_heap(&pqueue)

cpdef peek_01():

    cdef PriorityQueue pqueue

    init_heap(&pqueue, 6)

    min_heap_insert(&pqueue, 0, 9.0)
    assert peek(&pqueue) == 9.0
    min_heap_insert(&pqueue, 1, 9.0)
    assert peek(&pqueue) == 9.0
    min_heap_insert(&pqueue, 2, 9.0)
    assert peek(&pqueue) == 9.0
    min_heap_insert(&pqueue, 3, 5.0)
    assert peek(&pqueue) == 5.0
    min_heap_insert(&pqueue, 4, 3.0)
    assert peek(&pqueue) == 3.0
    min_heap_insert(&pqueue, 5, 1.0)
    assert peek(&pqueue) == 1.0

    free_heap(&pqueue)

cpdef extract_min_01():
    
    cdef PriorityQueue pqueue

    init_heap(&pqueue, 4)
    min_heap_insert(&pqueue, 1, 3.0)
    min_heap_insert(&pqueue, 0, 2.0)
    min_heap_insert(&pqueue, 3, 4.0)
    min_heap_insert(&pqueue, 2, 1.0)
    idx = extract_min(&pqueue)
    assert idx == 2
    assert pqueue.size == 3
    assert pqueue.Elements[idx].state == SCANNED
    idx = extract_min(&pqueue)
    assert idx == 0
    assert pqueue.size == 2
    assert pqueue.Elements[idx].state == SCANNED
    idx = extract_min(&pqueue)
    assert idx == 1
    assert pqueue.size == 1
    assert pqueue.Elements[idx].state == SCANNED
    idx = extract_min(&pqueue)
    assert idx == 3
    assert pqueue.size == 0
    assert pqueue.Elements[idx].state == SCANNED

    free_heap(&pqueue)

cpdef is_empty_01():
    
    cdef PriorityQueue pqueue

    init_heap(&pqueue, 4)

    assert is_empty(&pqueue) == 1
    min_heap_insert(&pqueue, 1, 3.0)
    assert is_empty(&pqueue) == 0
    idx = extract_min(&pqueue)
    assert is_empty(&pqueue) == 1

    free_heap(&pqueue)


cpdef decrease_key_from_element_index_01():

    cdef PriorityQueue pqueue

    init_heap(&pqueue, 4)

    min_heap_insert(&pqueue, 1, 3.0)
    min_heap_insert(&pqueue, 0, 2.0)
    min_heap_insert(&pqueue, 3, 4.0)
    min_heap_insert(&pqueue, 2, 1.0)

    assert pqueue.size == 4
    A_ref = [2, 0, 3, 1]
    n_ref = [1, 3, 0, 2]
    key_ref = [2.0, 3.0, 1.0, 4.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == IN_HEAP
        assert pqueue.Elements[i].key == key_ref[i]

    decrease_key_from_element_index(&pqueue, 3, 0.0)

    assert pqueue.size == 4
    A_ref = [3, 0, 2, 1]
    n_ref = [1, 3, 2, 0]
    key_ref = [2.0, 3.0, 1.0, 0.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == IN_HEAP
        assert pqueue.Elements[i].key == key_ref[i]


    decrease_key_from_element_index(&pqueue, 1, -1.0)

    assert pqueue.size == 4
    A_ref = [1, 3, 2, 0]
    n_ref = [3, 0, 2, 1]
    key_ref = [2.0, -1.0, 1.0, 0.0]
    for i in range(4):
        assert pqueue.A[i] == A_ref[i]
        assert pqueue.Elements[i].node_idx == n_ref[i]
        assert pqueue.Elements[i].state == IN_HEAP
        assert pqueue.Elements[i].key == key_ref[i]

    free_heap(&pqueue)


cdef void heapsort(DTYPE_t[:] values_in, DTYPE_t[:] values_out) nogil:

    cdef:
        size_t i, l = <size_t>values_in.shape[0]
        PriorityQueue pqueue
    
    init_heap(&pqueue, l)
    for i in range(l):
        min_heap_insert(&pqueue, i, values_in[i])
    for i in range(l):
        values_out[i] = pqueue.Elements[extract_min(&pqueue)].key
    free_heap(&pqueue)


cpdef sort_01(int n, random_seed=124):
    
    cdef PriorityQueue pqueue

    np.random.seed(random_seed)
    values_in = np.random.sample(size=n)
    values_out = np.empty_like(values_in, dtype=DTYPE)
    heapsort(values_in, values_out)
    values_in_sorted = np.sort(values_in)
    np.testing.assert_array_equal(values_in_sorted, values_out)
