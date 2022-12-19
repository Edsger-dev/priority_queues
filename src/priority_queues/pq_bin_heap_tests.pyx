

import numpy as np

from priority_queues.commons cimport (
    DTYPE, DTYPE_INF, IN_HEAP, NOT_IN_HEAP,
    SCANNED, DTYPE_t)
from priority_queues.pq_bin_heap cimport (
    BinaryHeap,
    decrease_key_from_element_index,
    extract_min, free_heap, init_heap,
    init_heap_para, is_empty,
    min_heap_insert, peek)


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


cpdef init_02():

    cdef: 
        BinaryHeap bheap
        size_t l = 40

    init_heap_para(&bheap, l, 4)

    assert bheap.length == l
    assert bheap.size == 0
    for i in range(l):
        assert bheap.A[i] == bheap.length
        assert bheap.elements[i].key == DTYPE_INF
        assert bheap.elements[i].state == NOT_IN_HEAP
        assert bheap.elements[i].node_idx == bheap.length

    free_heap(&bheap)


cpdef insert_01():
    """ Testing a single insertion into an empty binary heap 
    of length 1.
    """

    cdef: 
        BinaryHeap bheap
        DTYPE_t key

    init_heap(&bheap, 1)
    assert bheap.length == 1
    key = 1.0
    min_heap_insert(&bheap, 0, key)
    assert bheap.size == 1
    assert bheap.A[0] == 0
    assert bheap.elements[0].key == key
    assert bheap.elements[0].state == IN_HEAP
    assert bheap.elements[0].node_idx == 0

    free_heap(&bheap)


cpdef insert_02():

    cdef: 
        BinaryHeap bheap
        DTYPE_t key

    init_heap(&bheap, 4)

    elem_idx = 1
    key = 3.0
    min_heap_insert(&bheap, elem_idx, key)
    A_ref = [1, 4, 4, 4]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[elem_idx].key == key
    assert bheap.elements[elem_idx].state == IN_HEAP
    assert bheap.elements[1].node_idx == 0
    assert bheap.size == 1

    elem_idx = 0
    key = 2.0
    min_heap_insert(&bheap, elem_idx, key)
    A_ref = [0, 1, 4, 4]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[elem_idx].key == key
    assert bheap.elements[elem_idx].state == IN_HEAP
    assert bheap.elements[0].node_idx == 0
    assert bheap.elements[1].node_idx == 1
    assert bheap.size == 2

    elem_idx = 3
    key = 4.0
    min_heap_insert(&bheap, elem_idx, key)
    A_ref = [0, 1, 3, 4]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[elem_idx].key == key
    assert bheap.elements[elem_idx].state == IN_HEAP
    assert bheap.elements[0].node_idx == 0
    assert bheap.elements[1].node_idx == 1
    assert bheap.elements[3].node_idx == 2
    assert bheap.size == 3

    elem_idx = 2
    key = 1.0
    min_heap_insert(&bheap, elem_idx, key)
    A_ref = [2, 0, 3, 1]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
    assert bheap.elements[2].key == key
    assert bheap.elements[2].state == IN_HEAP
    assert bheap.elements[0].node_idx == 1
    assert bheap.elements[1].node_idx == 3
    assert bheap.elements[2].node_idx == 0
    assert bheap.elements[3].node_idx == 2
    assert bheap.size == 4

    free_heap(&bheap)

cpdef insert_03(n=4):
    """ Inserting nodes with identical keys.
    """
    cdef: 
        BinaryHeap bheap
        size_t i
        DTYPE_t key = 1.0

    init_heap(&bheap, n)
    for i in range(n):
        min_heap_insert(&bheap, i, key)
    for i in range(n):
        assert bheap.A[i] == i

    free_heap(&bheap)

cpdef peek_01():

    cdef BinaryHeap bheap

    init_heap(&bheap, 6)

    min_heap_insert(&bheap, 0, 9.0)
    assert peek(&bheap) == 9.0
    min_heap_insert(&bheap, 1, 9.0)
    assert peek(&bheap) == 9.0
    min_heap_insert(&bheap, 2, 9.0)
    assert peek(&bheap) == 9.0
    min_heap_insert(&bheap, 3, 5.0)
    assert peek(&bheap) == 5.0
    min_heap_insert(&bheap, 4, 3.0)
    assert peek(&bheap) == 3.0
    min_heap_insert(&bheap, 5, 1.0)
    assert peek(&bheap) == 1.0

    free_heap(&bheap)

cpdef extract_min_01():
    
    cdef BinaryHeap bheap

    init_heap(&bheap, 4)
    min_heap_insert(&bheap, 1, 3.0)
    min_heap_insert(&bheap, 0, 2.0)
    min_heap_insert(&bheap, 3, 4.0)
    min_heap_insert(&bheap, 2, 1.0)
    idx = extract_min(&bheap)
    assert idx == 2
    assert bheap.size == 3
    assert bheap.elements[idx].state == SCANNED
    idx = extract_min(&bheap)
    assert idx == 0
    assert bheap.size == 2
    assert bheap.elements[idx].state == SCANNED
    idx = extract_min(&bheap)
    assert idx == 1
    assert bheap.size == 1
    assert bheap.elements[idx].state == SCANNED
    idx = extract_min(&bheap)
    assert idx == 3
    assert bheap.size == 0
    assert bheap.elements[idx].state == SCANNED

    free_heap(&bheap)

cpdef is_empty_01():
    
    cdef BinaryHeap bheap

    init_heap(&bheap, 4)

    assert is_empty(&bheap) == 1
    min_heap_insert(&bheap, 1, 3.0)
    assert is_empty(&bheap) == 0
    idx = extract_min(&bheap)
    assert is_empty(&bheap) == 1

    free_heap(&bheap)


cpdef decrease_key_from_element_index_01():

    cdef BinaryHeap bheap

    init_heap(&bheap, 4)

    min_heap_insert(&bheap, 1, 3.0)
    min_heap_insert(&bheap, 0, 2.0)
    min_heap_insert(&bheap, 3, 4.0)
    min_heap_insert(&bheap, 2, 1.0)

    assert bheap.size == 4
    A_ref = [2, 0, 3, 1]
    n_ref = [1, 3, 0, 2]
    key_ref = [2.0, 3.0, 1.0, 4.0]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
        assert bheap.elements[i].node_idx == n_ref[i]
        assert bheap.elements[i].state == IN_HEAP
        assert bheap.elements[i].key == key_ref[i]

    decrease_key_from_element_index(&bheap, 3, 0.0)

    assert bheap.size == 4
    A_ref = [3, 0, 2, 1]
    n_ref = [1, 3, 2, 0]
    key_ref = [2.0, 3.0, 1.0, 0.0]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
        assert bheap.elements[i].node_idx == n_ref[i]
        assert bheap.elements[i].state == IN_HEAP
        assert bheap.elements[i].key == key_ref[i]


    decrease_key_from_element_index(&bheap, 1, -1.0)

    assert bheap.size == 4
    A_ref = [1, 3, 2, 0]
    n_ref = [3, 0, 2, 1]
    key_ref = [2.0, -1.0, 1.0, 0.0]
    for i in range(4):
        assert bheap.A[i] == A_ref[i]
        assert bheap.elements[i].node_idx == n_ref[i]
        assert bheap.elements[i].state == IN_HEAP
        assert bheap.elements[i].key == key_ref[i]

    free_heap(&bheap)


cdef void heapsort(DTYPE_t[:] values_in, DTYPE_t[:] values_out) nogil:

    cdef:
        size_t i, l = <size_t>values_in.shape[0]
        BinaryHeap bheap
    
    init_heap(&bheap, l)
    for i in range(l):
        min_heap_insert(&bheap, i, values_in[i])
    for i in range(l):
        values_out[i] = bheap.elements[extract_min(&bheap)].key
    free_heap(&bheap)


cpdef sort_01(int n, random_seed=124):
    
    cdef BinaryHeap bheap

    np.random.seed(random_seed)
    values_in = np.random.sample(size=n)
    values_out = np.empty_like(values_in, dtype=DTYPE)
    heapsort(values_in, values_out)
    values_in_sorted = np.sort(values_in)
    np.testing.assert_array_equal(values_in_sorted, values_out)
