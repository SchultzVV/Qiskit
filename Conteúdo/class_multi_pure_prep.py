import pennylane as qml
import numpy as np
#import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
#from qiskit.visualization import plot_state_city
from rsvg import rsvg
from rdmg import rdm_ginibre
class VQA():
    def setUp(self):
        self.n_qubit = 2
        self.observable = 'exp_val'# = 'target_op'
        self.ship = 'simulator'# = 'IBM'


    def gen_paulis():
        Paulis = Variable(torch.zeros([3, 2, 2], dtype=torch.complex128), requires_grad=False)
        Paulis[0] = torch.tensor([[0, 1], [1, 0]])
        Paulis[1] = torch.tensor([[0, -1j], [1j, 0]])
        Paulis[2] = torch.tensor([[1, 0], [0, -1]])
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


    target_vector = init_state_exp_val(2)

    #print('Traço(target_op) = ',np.trace(target_op))
    #plot_state_city(target_op.detach().numpy(), title='Matriz Densidade')

    def device_and_random_params(self):
        device = qml.device('qiskit.aer', wires=3, backend='qasm_simulator')
        params = np.random.normal(0,np.pi/2, self.n_qubit*3)
        params = Variable(torch.tensor(params), requires_grad=True)
        return device, params

    device, params = device_and_random_params()

    def circuit(n_qubits, params, M=None):
        for i in range(n_qubits):

            qml.Hadamard(wires=0)
            qml.RX(params[0], wires=0)
            qml.RY(params[1], wires=0)
            qml.RZ(params[2], wires=0)

        return qml.expval(qml.Hermitian(M, wires=[0]))
    
    def circuit(params, M=None):
        qml.Hadamard(wires=0)
        qml.RX(params[0], wires=0)
        qml.RY(params[1], wires=0)
        qml.RZ(params[2], wires=0)

        return qml.expval(qml.Hermitian(M, wires=[0]))

    qnode = qml.QNode(func=circuit, device=device, interface="torch")




    def cost(params, target_vector):
        Paulis = gen_paulis()
        L = 0
        for k in range(3):
            L += torch.abs(circuit(params, Paulis[k]) - target_vector[k])
        return L

    def train(epocas, params, target_vector):

        opt = torch.optim.Adam([params], lr=0.1)
        best_loss = 1*cost(params, target_vector)
        best_params = 1*params

        for epoch in range(epocas):
            opt.zero_grad()
            loss = cost(params, target_vector)
            print(epoch, loss.item())
            loss.backward()
            opt.step()
            if loss < best_loss:
                best_loss = 1*loss
                best_params = 1*params

        return best_params
    best_params = train(30, params, target_vector)


'''
def device_and_random_params():
    device = qml.device('qiskit.aer', wires=3, backend='qasm_simulator')
    params = np.random.normal(0,np.pi/2, 3)
    params = Variable(torch.tensor(params), requires_grad=True)
    return device,params

device, params = device_and_random_params()

def circuit_M(params, M=None):
    qml.Hadamard(wires=0)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RZ(params[2], wires=0)
    
    return qml.expval(qml.Hermitian(M, wires=[0]))
qnode = qml.QNode(circuit_M, device, interface="torch")

#fig, ax = qml.draw_mpl(circuit_M, decimals=2)(params)
#plt.show()


def cost(params):
    L = (1-(circuit_M(params, M=target_op)))**2
    return L

opt = torch.optim.Adam([params], lr=0.1)
best_loss = 1*cost(params)
best_params = 1*params

fidelity=[]
epochs=[]
for epoch in range(25):
    opt.zero_grad()
    loss = cost(params)
    print(epoch, loss.item())
    loss.backward()
    opt.step()
    if loss < best_loss:
        best_loss = 1*loss
        best_params = 1*params
    f = circuit_M(best_params, M=target_op).item()
    epochs.append(epoch)
    fidelity.append(f)

print(circuit_M(best_params, M=target_op).item())'''