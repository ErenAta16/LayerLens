cdef struct LayerInfo:
    double hidden_size
    double utility

cdef struct AllocationInfo:
    double method_code
    double rank
    double cost

cpdef int select_method(double utility, double[:] thresholds, double[:] method_codes)
cpdef double estimate_rank(double hidden_size, double utility)
cpdef double estimate_cost(double rank, double penalty)

