#from sympy import conjugate
from torch.autograd import Variable
import torch
from rsvg import rsvg
from rdmg import rdm_ginibre
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np

def gen_paulis(d):
    Paulis = Variable(torch.zeros([3*d, 2, 2], dtype=torch.complex128), requires_grad=False)
    aux = 0
    for i in range(0,d,3):
        Paulis[i+aux] = torch.tensor([[0, 1], [1, 0]])        
        Paulis[i+1+aux] = torch.tensor([[0, -1j], [1j, 0]])
        Paulis[i+2+aux] = torch.tensor([[1, 0], [0, -1]])
        aux += 2
    return Paulis

def init_state_rsvg(n_qb):
    d = 2**n_qb
    target_vector = rsvg(d)
    target_op = np.outer(target_vector.conj(), target_vector)
    target_op = torch.tensor(target_op)
    return target_vector, target_op

def init_state_rdm_ginibre(n_qb):
    d = 2**n_qb
    rho = rdm_ginibre(d)
    print(np.trace(np.dot(rho,rho)))
    target_op = torch.tensor(rho)
    return target_op

def init_state_exp_val(d):
    rrho = rdm_ginibre(4)
    Paulis = gen_paulis(d)
    target_vector = [np.trace(np.real(np.dot(rrho,i))) for i in Paulis]
    target_vector = Variable(torch.tensor(target_vector ))
    return target_vector

def get_device(n_qubit):
    device = qml.device('qiskit.aer', wires=n_qubit, backend='qasm_simulator')
    return device

def random_params(n):
    params = np.random.normal(0,np.pi/2, n)
    params = Variable(torch.tensor(params), requires_grad=True)
    return params

def fidelidade(circuit, params, target_op):
    return circuit(params, M=target_op).item()

def cost(circuit, params, alpha):
    L = (1-(circuit(params, M=alpha)))**2
    return L

def calc_mean(x_list):
    x_med = sum(x_list)/len(x_list)
    return x_med

def variancia(x_list, x1):
    x_med = calc_mean(x_list)
    var = (abs(x_med)-abs(x1))**2
    return var, x_med

def train(epocas, circuit, params, alpha):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, alpha)
    best_params = 1*params
    f=[]
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, alpha)
        #print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        f.append(fidelidade(circuit, best_params, alpha))
    print(epoch, loss.item())
    return best_params, f

def train2(epocas, circuit, params, alpha):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, alpha)
    best_params = 1*params
    f=[]
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, alpha)
        #print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        f.append(fidelidade(circuit, best_params, alpha))
    print(epoch, loss.item())
    return best_params, f
