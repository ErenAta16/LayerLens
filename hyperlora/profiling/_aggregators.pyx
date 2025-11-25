# cython: language_level=3, boundscheck=False, wraparound=False
"""
Cython implementation: takes scores as double array and returns weighted average.
"""

cdef double aggregate_scores_c(double[:] metrics, double[:] weights):
    cdef Py_ssize_t i, size = metrics.shape[0]
    cdef double total_weight = 0.0
    cdef double weighted_sum = 0.0

    for i in range(size):
        total_weight += weights[i]
        weighted_sum += metrics[i] * weights[i]

    if total_weight == 0:
        return 0.0

    return weighted_sum / total_weight

