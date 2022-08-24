cimport numpy as cnp

cdef DTYPE
ctypedef cnp.float64_t DTYPE_t

cdef enum ElementState:
   SCANNED = 1
   NOT_IN_HEAP = 2
   IN_HEAP = 3