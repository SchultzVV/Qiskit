from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from sympy import conjugate
from torch.autograd import Variable
import torch

def bloch_sphere(circuit):
    state = Statevector(circuit)
    return plot_bloch_multivector(state)

def inner_product(v,w):
    d = len(v); ip = 0
    for j in range(0,d):
        ip += conjugate(v[j])*w[j]
    return ip

def gen_paulis():
    Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
    Paulis[0] = torch.tensor([[0, 1], [1, 0]])        
    Paulis[2] = torch.tensor([[1, 0], [0, -1]])
    Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
    return Paulis
