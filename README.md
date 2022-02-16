## Hybrid Quantum Network Symmetry Learner (HQNSL) -- v0.2.1
This is HQNSL, a hybrid quantum scheme to learn the stabilizing operators, aka symmetries, 
of many-body quantum systems. Currently under development.

A unit test of the classical deep network regularizer is shown in `CNet_unit_test.ipynb` and the simplest
example of a HQNSL scheme is shown in `HQN_1gates.ipynb`. Tests are actively being performed
on the GHZ state, XY-Hamiltonian ground state, generalized cluster states, and eventually more.

### Current work
1. Analyze the scalability of the scheme with respect to number of qubits `L` and the generalizability with respect to different families of quantum states, such as the XY Hamiltonian states and the cluster states. Determine the quantum loss query complexity as a function of `L` and the quantum cross validation error (take the average of 3 random bases, for example). 
2. What optimizers are best? Global, or local? Do some perform better than others with larger `L`?
3. For a linearly entangled family of quantum circuits, classify the complexity of the symmetry by the minimum linear circuyit depth required to find it.
4. Does the scheme still learn just as efficiently (or at all) with the regularizer? Can we show a simple example in which the regularization prevents the network from traversing the same local part of a manifold, i.e. a family a symmetries that is sampled sparsely when regularized but densely without regularization? See the notes for an example of such a family in the 3-qubit GHZ state.

### Future (next few weeks) work
1. Add noise into the state generation process and the quantum circuit itself, and measure the increase in the query complexity. Does the loss get worse? How much noise can we add before everything breaks down--that is, what is the fault tolerance limit of our scheme?
2. What symmetries can be found with a full entanglement quantum circuit family that cannot be found by a linear entanglement family?
2. Run this on a real quantum computer. 

### Dependencies
1. Qiskit.
2. nlopt -- package for global optimizers. Required if using any of the optimizers with a prefix `g-` in the HQNet.
3. Cirq.
4. Pytorch.

Branch: Jon
