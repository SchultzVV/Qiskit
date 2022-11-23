from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from sympy import conjugate
from torch.autograd import Variable
import torch
from rsvg import rsvg
from rdmg import rdm_ginibre
import pennylane as qml

import numpy as np
#def bloch_sphere(circuit):
#    state = Statevector(circuit)
#    return plot_bloch_multivector(state)

#def inner_product(v,w):
#    d = len(v); ip = 0
#    for j in range(0,d):
#        ip += conjugate(v[j])*w[j]
#    return ip

def gen_paulis():
    Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
    Paulis[0] = torch.tensor([[0, 1], [1, 0]])        
    Paulis[2] = torch.tensor([[1, 0], [0, -1]])
    Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
    return Paulis

def init_state_fidelity(d):
    target_state = rsvg(d)
    target_op = np.outer(target_state.conj(), target_state)
    target_op = torch.tensor(target_op)
    return target_op

def init_state_exp_val(d):
    rrho = rdm_ginibre(2)
    Paulis = gen_paulis()
    target_vector = [np.trace(np.real(np.dot(rrho,i))) for i in Paulis]
    Variable(torch.tensor(target_vector ))
    return target_vector


def device_and_random_params():
    device = qml.device('qiskit.aer', wires=3, backend='qasm_simulator')
    params = np.random.normal(0,np.pi/2, 3)
    params = Variable(torch.tensor(params), requires_grad=True)
    return device, params

device, params = device_and_random_params()
def circuit(n_qubits, params, M=None):
    for i in range(n_qubits):
        qml.Hadamard(wires=i)
        qml.RX(params[0], wires=i)
        qml.RY(params[1], wires=i)
        qml.RZ(params[2], wires=i)
    return qml.expval(qml.Hermitian(M, wires=[0]))
print(qml.draw(circuit)(params))