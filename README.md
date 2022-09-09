# priority_queues
Priority queues for some path algorithms

## Tests

Run from the home directory:

```Python
$ pytest
```

To execute a single test:

```Python
$ py.test tests/test_pq_bin_heaps.py -k 'test_init_01'
```



## Road networks

|  | vertices | edges |
|---|---:|---:|
| CTR | 14081817 | 33866826 |
| USA | 23947348 | 57708624 |

## Elapsed time (s)

### commit c5aad3bce6d5a7e35d0d6f73d4ffb1531f702232

* convert the graph to CSR

|  | CTR | USA |
|---|---:|---:|
| SciPy | 0.51 | 0.89 |
| priority_queues |  0.56 | 0.94 |

* shortest path

|  | CTR | USA |
|---|---:|---:|
| SciPy | 10.46 | 15.69 |
| priority_queues |  4.94 | 6.89 |

### commit f5030123502d84d9d7141b616b59080d3be1a72f

* convert the graph to CSR or specific format

|  | CTR | USA |
|---|---:|---:|
| SciPy | 0.40 | 0.64 |
| graph-tool | 5.87 | 6.94 |
| priority_queues | 0.56 | 0.94 |

* shortest path

|  | CTR | USA |
|---|---:|---:|
| SciPy | 10.79 | 15.80 |
| graph-tool | 6.31| 8.71 |
| priority_queues | 4.27 | 5.90 |