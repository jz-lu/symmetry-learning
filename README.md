# Hybrid Quantum Symmetry Learning Network (HQSLN) -- v1.0.0
This is HQSLN (suggested pronounciation: HOCK-SLOAN), a hybrid quantum scheme to learn the symmetries of an unknown quantum state, i.e. the unitary matrices for which the state is an eigenvector. This repository serves as a minimal working product for the techniques described in the corresponding paper, available [on the arXiv](https://arxiv.org/abs/2206.11970). In this documentation, we will briefly describe the program structure of the learning algorithm, and direct the reader to the appropriate files for playing with the algorithm. The full technical details are available in the paper. 

Without knowing any further details, a working minimal example of learning symmetries of the 3-qubit GHZ state can be found by running `simplified.py`. It contains a combination of the code below, but all in one file.

## Program Structure
Key elements of the algorithm are implemented in the Python scripts prefixed by a number of underscores `__`. the files without underscores analyze the algorithm in some particular way, which we describe in the next section. Generally, these are quite well-commented. We hope that a reader who has already read the paper should be able to read through the code without significant trouble.
1. `___constants.py`: As the name suggests. Most hyperparameters, excluding learning rate, are included here.
2. `__class_BasisTransformer.py`: Contains the class `BasisTransformer` which handles the random basis measurements used by the algorithm.
3. `__class_CNet.py`: Contains the class `CNet` which implements the classical convolutional net that performs regularization interacitvely with the quantum net.
4. `__class_HQNet.py`: Contains the class `HQNet` which implements the entire learning algorithm. In particular, it instantiates all the other classes.
5. `__class_PQC.py`: Contains the class `PQC` which instantiates our ansatz parameterized variational quantum circuit.
6. `__helpers.py`: Contains a collection of small helper functions which perform miscellaneous simple tasks that are repeatedly used in the algorithm.
7. `__loss_funcs.py`: Contains a number of different (classical! that is, the input is a measurement outcome, not a state -- so it does not include losses like the SWAP test) loss functions we used to score the quantum net. We ultimately went with the KL divergence.

## Classification of High-level Python programs for Algorithms & Data Collection
The high-level Python programs, which we used to generate the data used in the paper, are separated into two categories. One-shot runs, which we use to quickly visualize what is going on and perform testing/debugging, are available in Jupyter notebooks. For generating figures which required a lot of data, we used Python scripts *without any leading underscores*. We recommend running such scripts on a supercomputing cluster or something similar, rather than a personal laptop, since they can require a large amount of memory and several hours/a few days to complete.

### Jupyter Notebooks for Examples and Unit Tests
Most of these have comments and describe briefly what is going on. These notebooks are meant to showcase a simple example of some aspect (or the entirety) of the algorithm, and are not meant to be super rigorous in any statistical sense.

1. A unit test of the classical deep network regularizer is shown in `CNet.ipynb`.
2. The HQNSL scheme is shown in `HQN.ipynb`. 
3. An analysis a scheme in which we estimate the distribution by sampling, rather than use `qiskit`'s e.g. `state.probabilities()` function, is in `Estimation.ipynb`.
4. Simulated noise is added to the quantum state in `NoisyCirc.ipynb` and `NoisyState.ipynb`.
5. We study the regularization procedure in `Regularizer.ipynb`.

Though it is not super related, if it can be of use, `Hamiltonians.ipynb` generates the Hamiltonian of a XY model and the toric code.

## The Python Scripts
The following scripts were used to generate data for analysis.
1. `CircComp.py`: Calculate the "circuit" complexity of the 4-qubit cluster state, 4-qubit GHZ state, and 4-qubit XY Hamiltonian GS.
2. `cluster_generator.py`: Generate cluster states in Google's `cirq`, and convert to `qiskit` type Statevectors. Adapted from code given by K. Najafi.
3. `DScaling.py`: Analyze the numerical query complexity as a function of ansatz circuit depth.
4. `LScaling.py`: Same as above, but as a function of number of qubits.
5. `GHZ_generator.py`: Generate L-qubit GHZ state preparation circuits.
6. `mGHZ_generator.py`: Similar to above, but also include a `pi/3` rotation at the end.
7. `Noise.py`: Simulate the effect of noise on the GHZ state.
8. `p2op.py`: Convert a qiskit circuit form into operator form.
9. `Ryd_generator.py`: Import ground states of the 7-qubit Rydberg Hamiltonian under different phases, disodered (`DO`), high-entropy (`HE` -- the boundary between lobes in the phase diagram which have large entanglement entropy), and the periodic ordered phases `Z2` and `Z3`. The ground states, in `data/Rydberg_phases`, were generated externally in `Julia`, using the libraries `Yao` and `bloqade`.
10. `Rydberg.py`: Finds symmetries of the 7-qubit Rydberg chain.
11. `XY_generator.py`: Generate ground states of a XY Hamiltonian.

## Data Analysis
In addition to scripts for implementing the algorithm in various aspects, we also include our scripts used to analyze data, and the raw data, for the figures. The raw data is contained in the `data` folder, and the analysis of the raw data is in the `Analyzers` folder.

1. `CC.ipynb`: "Circuit complexity"
2. `DLS.ipynb`: Numerical query complexity
3. `DS/LS.ipynb`: Similar to the above, but just as a function of depth or qubit size.
4. `GradCirc.ipynb`: Analysis of the noise curve.
5. `PCA.py / PCA.ipynb`: PCA (principle component analysis) visualization of GHZ state symmetries found.

*Remark*: Depending on how you run the scripts, you may need to change the paths near the beginning of the script.

## Dependencies
1. `Qiskit`.
2. `nlopt` -- package for global optimizers. Required if using any of the optimizers with a prefix `g-` in the HQNet.
3. `Cirq`.
4. `Pytorch`.
5. `Jupyter`.

Branch: `main`
Version: 1.0.0
