Author: Samuel Begg 
13/08/2025

Calculate observables, Fisher information (via reduced density matrix) for a general XY spin chain in a transverse field by numerical integration of fermions (N x N matrices). Code can be applied to periodically driven problems. 

-XY_time_evolution.ipynb calculates Fisher information during the time-evolution

-XY_ground_states.ipynb calculates Fisher information for ground-states

-Floquet_time_evolution.ipnb does this by constructing the Floquet unitary over 1 period and then mapping forward with the Floquet rule.

-Matrix integrator and ED codes are brute force, the matrix integrator is Euler method and requires small step size.

-Analysis files for personal use in checking code outputs. 

