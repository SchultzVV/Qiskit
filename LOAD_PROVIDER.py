from qiskit import IBMQ
from qiskit import *
import qiskit
#IBMQ.save_account('7cc830e0cb005ce6de9caf9c402a1992c5d38d25af4237c19e72a4c58ce204481eb7a0d08b9929e5b5d3028f3146c38d2da8f2eb8db8f6f8b02a97eeb8fbc6de')
#IBMQ.load_account()
#provider = qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')
def load():
    IBMQ.save_account('7cc830e0cb005ce6de9caf9c402a1992c5d38d25af4237c19e72a4c58ce204481eb7a0d08b9929e5b5d3028f3146c38d2da8f2eb8db8f6f8b02a97eeb8fbc6de')
    IBMQ.load_account()
def provider():
    load()
    return qiskit.IBMQ.get_provider(hub='ibm-q', group='open', project='main')

