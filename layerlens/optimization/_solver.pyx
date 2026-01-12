# cython: language_level=3, boundscheck=False, wraparound=False
"""
Simple rank and cost estimation functions at C level.
"""

cpdef int select_method(double utility, double[:] thresholds, double[:] method_codes):
    """
    Selects method based on utility value.
    
    Args:
        utility: Layer utility score
        thresholds: Threshold values (sorted in ascending order)
        method_codes: Method codes for each threshold
        
    Returns:
        Index of the selected method
        
    Note:
        If utility is greater than all thresholds, returns the last method.
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t size = thresholds.shape[0]
    cdef Py_ssize_t codes_size = method_codes.shape[0]
    
    # Bounds checking: method_codes must contain at least size+1 elements
    if codes_size == 0:
        return 0  # Default: first method
    
    # Check threshold values
    for i in range(size):
        if utility < thresholds[i]:
            # Bounds check: i < codes_size
            if i < codes_size:
                return <int>method_codes[i]
            else:
                return <int>method_codes[codes_size - 1]  # Last method
    
    # If greater than all thresholds, return last method
    # Bounds check: size < codes_size must hold (for last method)
    if size < codes_size:
        return <int>method_codes[size]
    else:
        return <int>method_codes[codes_size - 1]  # Safe fallback


cpdef double estimate_rank(double hidden_size, double utility):
    """
    Estimates rank based on utility and hidden size.
    Improved formula: base_rank = hidden_size * utility * scale_factor
    Scale factor increased from 0.1 to 0.15 for better budget utilization.
    """
    # Increased scale factor from 0.1 to 0.15 for better utilization
    # This gives: utility=0.1 -> rank = 768 * 0.1 * 0.15 = 11.52 (was 7.68)
    cdef double base_rank = hidden_size * utility * 0.15
    if base_rank < 1.0:
        base_rank = 1.0
    if base_rank > hidden_size:
        base_rank = hidden_size
    return base_rank


cpdef double estimate_cost(double rank, double penalty):
    return rank * penalty

