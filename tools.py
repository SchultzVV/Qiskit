#from sympy import conjugate
from torch.autograd import Variable
import torch
from rsvg import rsvg
from rdmg import rdm_ginibre
import pennylane as qml
import matplotlib.pyplot as plt
import numpy as np
import cmath

#class TestApp(unittest.TestCase):
#    def __init__(self, *args, **kwargs):
#        super(TestApp, self).__init__(*args, **kwargs)
#        self.gen_stufs()
#
#    def gen_stufs(self):
#        self.path_input = "data/violencia_sexual_recorte_5k_registros.csv"
#        self.base = load_dataset(self.path_input)
#        self.base_vitima, self.base_autor = split_dataset_by_city(
#            self.base, 'S.PAULO')
#        self.mapa_cluster = populate_maps(self.base_vitima, self.base_autor)

def bpf(theta, phi, p):
    state = np.zeros(4,dtype=complex)
    state[0] = np.sqrt(1-p)*np.cos(theta/2)
    state[1] = cmath.exp(1j*phi)*np.sin(theta/2)
    state[2] = 1j*np.sqrt(p)*(-1)*cmath.exp(1j*phi)*np.sin(theta/2)
    state[3] = 1j*np.sqrt(p)*np.cos(theta/2)
    return state

def init_state_bpf(n_qb, p):
    d = 2**n_qb
    target_vector = bpf(np.pi/4, 0, p)
    target_op = np.outer(target_vector.conj(), target_vector)
    target_op = torch.tensor(target_op)
    return target_vector, target_op

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

def cost(circuit, params, target_op):
    L = (1-(circuit(params, M=target_op)))**2
    return L

def calc_mean(x_list):
    x_med = sum(x_list)/len(x_list)
    return x_med

def variancia(x_list, x1):
    x_med = calc_mean(x_list)
    var = (abs(x_med)-abs(x1))**2
    return var, x_med

def train(epocas, circuit, params, target_op):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, target_op)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    best_params = 1*params
    f=[]
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, target_op)
        #print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        f.append(fidelidade(circuit, best_params, target_op))
    print(epoch, loss.item())
    return best_params, f


def train(epocas, circuit, params, target_op):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, target_op)
    best_params = 1*params
    f=[]
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, target_op)
        print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        f.append(fidelidade(circuit, best_params, target_op))
    print(epoch, loss.item())
    return best_params, f

def train2(epocas, circuit, params, target_op):
    opt = torch.optim.Adam([params], lr=0.1)
    best_loss = 1*cost(circuit, params, target_op)
    best_params = 1*params
    f = []
    erros = []
    for epoch in range(epocas):
        opt.zero_grad()
        loss = cost(circuit, params, target_op)
        #print(epoch, loss.item())
        loss.backward()
        opt.step()
        if loss < best_loss:
            best_loss = 1*loss
            best_params = 1*params
        
        f.append(fidelidade(circuit, best_params, target_op))
        erros.append(loss.item())
    #print(epoch, loss.item())
    return best_params, f, erros

def vqa(n_qubits, depht=None):
    #n_qubits = 1
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        w = []
        aux = 0
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            w.append(j)
            aux+=2
        if n_qubits == 1:
            for z in range(1,depht):
                qml.RX(params[j+aux], wires=j)
                qml.RY(params[j+1+aux], wires=j)
                qml.RZ(params[j+2+aux], wires=j)
                aux+=2
            return qml.expval(qml.Hermitian(M, wires=w))
        for z in range(depht):
            for i in range(n_qubits-1):
                qml.CNOT(wires=[i,i+1])
            for j in range(n_qubits):
                qml.RX(params[j+aux], wires=j)
                qml.RY(params[j+1+aux], wires=j)
                qml.RZ(params[j+2+aux], wires=j)
                aux+=2
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def vqa_extra_cnot(n_qubits):
    #n_qubits = 1
    depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        qml.CNOT(wires=[0,1])
        aux = 0
        w = []
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
            w.append(j)
        qml.CNOT(wires=[0,2])
        qml.CNOT(wires=[1,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        #    w.append(j)
        #for j in range(n_qubits):
        #    qml.RX(params[j+aux], wires=j)
        #    qml.RY(params[j+1+aux], wires=j)
        #    qml.RZ(params[j+2+aux], wires=j)
        #    w.append(j)
        #    aux+=2
        #if n_qubits == 1:
        #    for z in range(1,depht):
        #        qml.RX(params[j+aux], wires=j)
        #        qml.RY(params[j+1+aux], wires=j)
        #        qml.RZ(params[j+2+aux], wires=j)
        #        aux+=2
        #    return qml.expval(qml.Hermitian(M, wires=w))
        #for z in range(depht):
        #    for i in range(n_qubits-1):
        #        qml.CNOT(wires=[i,i+1])
        #    for j in range(n_qubits):
        #        qml.RX(params[j+aux], wires=j)
        #        qml.RY(params[j+1+aux], wires=j)
        #        qml.RZ(params[j+2+aux], wires=j)
        #        aux+=2
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def vqa_extra_cnot_depth5(n_qubits):
    #n_qubits = 1
    depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        aux = 0
        w = []
        qml.CNOT(wires=[0,1])
        for j in range(n_qubits-1):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
            
        qml.CNOT(wires=[0,2])
        for j in range(1, n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            w.append(j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[0,1])
        qml.CNOT(wires=[2,3])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        qml.CNOT(wires=[1,0])
        qml.CNOT(wires=[2,3])
        #for j in range(n_qubits):
        #    qml.RX(params[j+aux], wires=j)
        #    qml.RY(params[j+1+aux], wires=j)
        #    qml.RZ(params[j+2+aux], wires=j)
        #    aux+=2
        #    w.append(j)
        #for j in range(n_qubits):
        #    qml.RX(params[j+aux], wires=j)
        #    qml.RY(params[j+1+aux], wires=j)
        #    qml.RZ(params[j+2+aux], wires=j)
        #    w.append(j)
        #    aux+=2
        #if n_qubits == 1:
        #    for z in range(1,depht):
        #        qml.RX(params[j+aux], wires=j)
        #        qml.RY(params[j+1+aux], wires=j)
        #        qml.RZ(params[j+2+aux], wires=j)
        #        aux+=2
        #    return qml.expval(qml.Hermitian(M, wires=w))
        #for z in range(depht):
        #    for i in range(n_qubits-1):
        #        qml.CNOT(wires=[i,i+1])
        for j in range(n_qubits):
            qml.RX(params[j+aux], wires=j)
            qml.RY(params[j+1+aux], wires=j)
            qml.RZ(params[j+2+aux], wires=j)
            aux+=2
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

def vqa_bpf(n_qubits, depht=None):
    #n_qubits = 1
    if depht == None:
        depht = n_qubits+1
    n = 3*n_qubits*(1+depht)
    params = random_params(n)
    device = get_device(n_qubits)
    @qml.qnode(device, interface="torch")
    def circuit(params, M=None):
        w = [0,1]

        for j in range(0,24,6):
            print(j)
            qml.RX(params[j], wires=0)
            qml.RY(params[j+1], wires=0)
            qml.RZ(params[j+2], wires=0)
            qml.RX(params[j+3], wires=1)
            qml.RY(params[j+4], wires=1)
            qml.RZ(params[j+5], wires=1)
            if j <18:
                qml.CNOT(wires=[0, 1])
        return qml.expval(qml.Hermitian(M, wires=w))
    return circuit, params

#n_qubits = 4
#fidelidades = []
#
#circuit, params = vqa_extra_cnot_depth5(n_qubits)
#target_vector, target_op = init_state_rsvg(n_qubits)
#fig, ax = qml.draw_mpl(circuit, decimals=1)(params, target_op)
#plt.show()

# 
# class VQA(object):
#     def __init__(self, *args, **kwargs):
#         super(VQA, self).__init__(*args, **kwargs)
#         self.gen_stufs()
# 
#     def gen_stufs(self, n_qubits):
#         self.path_input = "data/violencia_sexual_recorte_5k_registros.csv"
#         self.base = load_dataset(self.path_input)
# 
#     def gen_paulis(d):
#     Paulis = Variable(torch.zeros([3*d, 2, 2], dtype=torch.complex128), requires_grad=False)
#     aux = 0
#     for i in range(0,d,3):
#         Paulis[i+aux] = torch.tensor([[0, 1], [1, 0]])        
#         Paulis[i+1+aux] = torch.tensor([[0, -1j], [1j, 0]])
#         Paulis[i+2+aux] = torch.tensor([[1, 0], [0, -1]])
#         aux += 2
#     return Paulis
# 
# def init_state_rsvg(n_qb):
#     d = 2**n_qb
#     target_vector = rsvg(d)
#     target_op = np.outer(target_vector.conj(), target_vector)
#     target_op = torch.tensor(target_op)
#     return target_vector, target_op
# 
# def init_state_rdm_ginibre(n_qb):
#     d = 2**n_qb
#     rho = rdm_ginibre(d)
#     print(np.trace(np.dot(rho,rho)))
#     target_op = torch.tensor(rho)
#     return target_op
# 
# def init_state_exp_val(d):
#     rrho = rdm_ginibre(4)
#     Paulis = gen_paulis(d)
#     target_vector = [np.trace(np.real(np.dot(rrho,i))) for i in Paulis]
#     target_vector = Variable(torch.tensor(target_vector ))
#     return target_vector
# 
# def get_device(n_qubit):
#     device = qml.device('qiskit.aer', wires=n_qubit, backend='qasm_simulator')
#     return device
# 
# def random_params(n):
#     params = np.random.normal(0,np.pi/2, n)
#     params = Variable(torch.tensor(params), requires_grad=True)
#     return params
# 
# def fidelidade(circuit, params, target_op):
#     return circuit(params, M=target_op).item()
# 
# def cost(circuit, params, target_op):
#     L = (1-(circuit(params, M=target_op)))**2
#     return L
# 
# def calc_mean(x_list):
#     x_med = sum(x_list)/len(x_list)
#     return x_med
# 
# def variancia(x_list, x1):
#     x_med = calc_mean(x_list)
#     var = (abs(x_med)-abs(x1))**2
#     return var, x_med
# 
# def train(epocas, circuit, params, target_op):
#     opt = torch.optim.Adam([params], lr=0.1)
#     best_loss = 1*cost(circuit, params, target_op)
#     best_params = 1*params
#     f=[]
#     for epoch in range(epocas):
#         opt.zero_grad()
#         loss = cost(circuit, params, target_op)
#         #print(epoch, loss.item())
#         loss.backward()
#         opt.step()
#         if loss < best_loss:
#             best_loss = 1*loss
#             best_params = 1*params
#         f.append(fidelidade(circuit, best_params, target_op))
#     print(epoch, loss.item())
#     return best_params, f
# 
# def train2(epocas, circuit, params, target_op):
#     opt = torch.optim.Adam([params], lr=0.1)
#     best_loss = 1*cost(circuit, params, target_op)
#     best_params = 1*params
#     f=[]
#     for epoch in range(epocas):
#         opt.zero_grad()
#         loss = cost(circuit, params, target_op)
#         #print(epoch, loss.item())
#         loss.backward()
#         opt.step()
#         if loss < best_loss:
#             best_loss = 1*loss
#             best_params = 1*params
#         
#         f.append(fidelidade(circuit, best_params, target_op))
#     print(epoch, loss.item())
#     return best_params, f
# 
# def vqa(n_qubits):
#     #n_qubits = 1
#     depht = n_qubits+1
#     n = 3*n_qubits*(1+depht)
#     params = random_params(n)
#     device = get_device(n_qubits)
#     @qml.qnode(device, interface="torch")
#     def circuit(params, M=None):
#         w = []
#         aux = 0
#         for j in range(n_qubits):
#             qml.RX(params[j+aux], wires=j)
#             qml.RY(params[j+1+aux], wires=j)
#             qml.RZ(params[j+2+aux], wires=j)
#             w.append(j)
#             aux+=2
#         if n_qubits == 1:
#             for z in range(1,depht):
#                 qml.RX(params[j+aux], wires=j)
#                 qml.RY(params[j+1+aux], wires=j)
#                 qml.RZ(params[j+2+aux], wires=j)
#                 aux+=2
#             return qml.expval(qml.Hermitian(M, wires=w))
#         for z in range(depht):
#             for i in range(n_qubits-1):
#                 qml.CNOT(wires=[i,i+1])
#             for j in range(n_qubits):
#                 qml.RX(params[j+aux], wires=j)
#                 qml.RY(params[j+1+aux], wires=j)
#                 qml.RZ(params[j+2+aux], wires=j)
#                 aux+=2
#         return qml.expval(qml.Hermitian(M, wires=w))
#     return circuit, params
# 