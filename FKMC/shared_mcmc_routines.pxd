cpdef initialise_state_representations(double [::1] state, double [:, ::1] interaction_matrix)
cpdef double c_classical_energy(
                                    double mu,
                                    double[::1] state,
                                    double[::1] t,
                                    double [::1] background) nogil
cpdef void invert_site_inplace(
                        long i,
                        double [::1] alternating_signs,
                        double [::1] state,
                        double [::1] ut,
                        double [::1] t,
                        double [::1] background,
                        double [:, ::1] interaction_matrix,
                       ) nogil

cpdef double incremental_energy_difference(long i,
                                    double mu,
                                    double[::1] ut,
                                    double[::1] t,
                                    double [::1] background) nogil

cdef void spin_spin_correlation(double[::1] correlation, double[::1] state) nogil