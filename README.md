# netket_qsl
This package allows for the simulation of the article "Variational Simulation of a topological spin liquid"

## Main classes : 
- lattice : used to describe the Kagome lattice of the experiment. Multiple shapes are possible (rectangular or ruby) with multiple boundary conditions for the rectangular-shaped ones. 

- hilbert : custom retricted hilbert space with perfect first neighbor blockade

- operator : all operators necessary, i.e. Rydberg hamiltonian as well as topological operators. 

- frequencies : callables of the frequency schedules one can use for Ω(t) and Δ(t). already implemented : Linear or Cubic schedule. 

- models : NQS models used for the time evolution. Mainly mean-field and jastrow ansatze. 

- rules : custom sampling rules (restricted means compatible with the restricted hilbert space).

- driver : mean-field driver for the analytical resolution of the TDVP. 

## On the side
Beside these main classes, there are a few utilities one can use : 
- callbacks to use during evolution
- utils with some often used utilities
- toy_model for models working specifically only on the lattice Torus(a,2,4) with restricted Hilbert space (where things can be computed exactly)