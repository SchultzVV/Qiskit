import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from qiskit.visualization import plot_state_city
from rsvg import rsvg

def init_state(d):
    target_state = rsvg(d)
    target_op = np.outer(target_state.conj(), target_state)
    target_op = torch.tensor(target_op)
    return target_op


target_op = init_state(2)
#print('Traço(target_op) = ',np.trace(target_op))
#plot_state_city(target_op.detach().numpy(), title='Matriz Densidade')

def device_and_random_params():
    device = qml.device('qiskit.aer', wires=3, backend='qasm_simulator')
    params = np.random.normal(0,np.pi/2, 3)
    params = Variable(torch.tensor(params), requires_grad=True)
    return device,params

#device = qml.device('qiskit.aer', wires=3, backend='qasm_simulator')
#params = np.random.normal(0,np.pi/2, 3)
#params = Variable(torch.tensor(params), requires_grad=True)
device, params = device_and_random_params()

@qml.qnode(device, interface="torch")
def circuit(params, M=None):
    qml.Hadamard(wires=0)
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RZ(params[2], wires=0)
    
    return qml.expval(qml.Hermitian(M, wires=[0]))

def cost(params):
    L = (1-(circuit(params, M=target_op)))**2
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
    f = circuit(best_params, M=target_op).item()
    epochs.append(epoch)
    fidelity.append(f)

print(circuit(best_params, M=target_op).item())

plt.plot(epochs, fidelity)
plt.xlabel('epochs')
plt.ylabel('fidelidade')
plt.show()

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