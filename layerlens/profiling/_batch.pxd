cdef void gradient_energy_batch(double[:, :] grads, double[:] output)
cdef void fisher_trace_batch(double[:, :] fisher, double[:] output)
cdef void hutchinson_trace_batch(double[:, :, :] tensor, double[:] output, int samples, int seed)

