# Adapted from Sona's cirq method of cluster state generation
import cirq
import copy
import numpy as np
from cirq import Simulator
from cirq.ops import CZ, H, S, CNOT,I
from cirq.circuits import InsertStrategy
from random import choices
from random import sample
import itertools
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from collections import Counter
from scipy.linalg import fractional_matrix_power
from typing import Tuple
from qiskit import QuantumCircuit, execute, Aer


# Conversions of circuits between libraries
def cirq2qasm(circuit, qubits):
    return cirq.QasmOutput((circuit.all_operations()), qubits)

def qasm2qiskit(asmcirc):
    return QuantumCircuit().from_qasm_str(str(asmcirc))

def cirq2qiskit(circuit, qubits):
    asmcirc = cirq2qasm(circuit, qubits)
    return qasm2qiskit(asmcirc)

def q_measure(num_qubits, print_log=False):
    # Measurement operator setup
    l = range(num_qubits*num_qubits)
    l1 = []; l2 = []
    for j in range(0,num_qubits*num_qubits,num_qubits):
        l1.append(j)
        l2.append(j + num_qubits-1)
        
    qmeas=[]
    for i in range(len(l)):
        if i not in l1 and i not in l2:
            dummy=[]
            dummy.append(i)
            if (i-num_qubits in l):
                dummy.append(i-num_qubits)
            if (i-1 in l):
                dummy.append(i-1)
            if (i+1 in l):
                dummy.append(i+1)
            if (i+num_qubits in l):
                dummy.append(i+num_qubits)
            qmeas.append(dummy)

        elif i in l1:
            dummy=[]
            dummy.append(i)
            if (i-num_qubits in l):
                dummy.append(i-num_qubits)
            if (i+1 in l):
                dummy.append(i+1)
            if (i+num_qubits in l):
                dummy.append(i+num_qubits)
            qmeas.append(dummy)

        elif i in l2:
            dummy=[]
            dummy.append(i)
            if (i-num_qubits in l):
                dummy.append(i-num_qubits)
            if (i-1 in l):
                dummy.append(i-1)
            if (i+num_qubits in l):
                dummy.append(i+num_qubits)
            qmeas.append(dummy)
    if print_log:
        print(f"qmeas:\n{qmeas}\n")
    return qmeas

def cirq_cluster_state(qmeas, num_qubits, plot=False, print_log=False):
    cirqs = []
    cirqs_meas = []
    cirq_nomeas = None
    
    # Build a num_qubits * num_qubits grid
    q = [cirq.GridQubit(i, j) for i in range(num_qubits) for j in range(num_qubits)] 
    
    par_list = []
    for k in range(len(qmeas)):
        circuit_2DC= cirq.Circuit()
        for i in range(len(q)):
            circuit_2DC.append(H(q[i]))

        for i in range(0,len(q)-1,num_qubits):
            for j in range(0,num_qubits-1):
                circuit_2DC.append([CZ(q[i+j], q[i+j+1])], strategy=InsertStrategy.EARLIEST)

        for i in range(0,len(q)-num_qubits):
            circuit_2DC.append([CZ(q[i], q[i+num_qubits])], strategy=InsertStrategy.NEW)

        if cirq_nomeas is None:
            cirq_nomeas = circuit_2DC
            circuit_2DC = copy.deepcopy(circuit_2DC)

        for i in range(len(q)):
            if i == k:
                circuit_2DC.append(H(q[i]), strategy=InsertStrategy.NEW if i == 0 else InsertStrategy.INLINE) 
            else:
                circuit_2DC.append(I(q[i]), strategy=InsertStrategy.NEW if i == 0 else InsertStrategy.INLINE) 
        cirqs.append(circuit_2DC)
        
        circuit_meas = copy.deepcopy(circuit_2DC)
        for i in range(len(qmeas[k])):
            circuit_meas.append(cirq.measure(q[qmeas[k][i]]))

        simulator_meas_DP = cirq.Simulator()
        result_meas_DP = simulator_meas_DP.run(circuit_meas, repetitions=1000)
        result_meas = result_meas_DP
        if plot:
            cirq.vis.plot_state_histogram(result_meas) 

        key_list = list(result_meas.measurements.keys())
        result_array = np.array([result_meas.measurements.get(key) for key in key_list])
        output1 = result_array
        output2 = output1.T
        shape = output2.shape
        output3 = np.reshape(output2,(shape[1], shape[2]))
        if print_log:
            print("Measurement out shape:", output3.shape)

        ev_li = []; od_li = []
        for xx in range(len(output3)):
            s = np.sum(output3[xx])
            if (s % 2 == 0): 
                pp1 = 1
                ev_li.append(pp1)
            else: 
                pp2 = -1
                od_li.append(pp2)

        parity = (np.sum(ev_li) + np.sum(od_li)) / len(output3)
        if print_log:
            print("Parity:", parity)
        par_list.append(parity)
        cirqs_meas.append(circuit_meas)
    return q, cirqs, cirqs_meas, cirq_nomeas

"""Cluster state circuit on L qubits"""
def cluster_state_circuit(L):
    # Get the circuits and show them in Cirq view
    qubits,_,_, cirq_nomeas = cirq_cluster_state(q_measure(L), L)
    qc = cirq2qiskit(cirq_nomeas, qubits)
    return qc