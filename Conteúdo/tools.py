from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from sympy import conjugate


def bloch_sphere(circuit):
    state = Statevector(circuit)
    return plot_bloch_multivector(state)

def inner_product(v,w):
    d = len(v); ip = 0
    for j in range(0,d):
        ip += conjugate(v[j])*w[j]
    return ip