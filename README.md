# netket_qsl
This package allows for the simulation of the article "Variational Simulation of a topological spin liquid"

Main classes : 
lattice : used to describe the Kagomé lattice of the experiment. Multiple shapes are possible (rectangular or ruby) with multiple boundary conditions for the rectangular ones. 

operator : all operators necessary, i.e. Rydberg hamiltonian as well as topological operators. 

hilbert : custom retricted hilbert space with perfect first neighbor blockade

rules : custom sampling rules (restricted means compatible with the restricted hilbert space)

driver : mean-field driver for the analytical resolution of the TDVP. 
