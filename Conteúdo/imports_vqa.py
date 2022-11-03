import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from sympy import conjugate
from qiskit_textbook.widgets import plot_bloch_vector_spherical
from torch.autograd import Variable
import torch