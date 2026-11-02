from qiskit import QuantumCircuit
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy, Statevector, Kraus, entanglement_of_formation,Operator 
import matplotlib.pyplot as plt
# to run:
# & "c:\Users\lynam\anacondacustom\envs\qiskit\python.exe" functions.py


def conditional_entropy(params, rho):
    theta, phi = params
    #basis of spin operators along unit vector n
    basis = np.array([
        [np.cos(theta/2), np.exp(1j * phi)*np.sin(theta/2)],
        [np.sin(theta/2), -np.exp(1j * phi)*np.cos(theta/2)]])

    total =0 
    for b in basis:
        P = np.outer(b,b.conj()) #projection measurement
        M = np.kron(P,np.eye(2)) #hermitian, acts on qubit B (q1)
        post = DensityMatrix(M@rho.data@M)
        p = np.real(np.trace(post.data)) #prob of outcome i 
        if p>1e-12:
            #state of subsystem A:
            rho_A_i = partial_trace(DensityMatrix(post.data/p),[1])
            total += p * entropy(rho_A_i, base=2)
        
    return total   

def QD(rho):

    rho = DensityMatrix(rho)
    rho_A = partial_trace(rho, [1]) #Tr_B[]
    rho_B = partial_trace(rho,[0]) # Tr_A[]

    #entropy
    S_A = entropy(rho_A, base=2)
    S_B = entropy(rho_B, base=2)
    S_AB = entropy(rho, base=2)
    mutual_info = S_A + S_B - S_AB

    #thetas = np.linspace(0,np.pi,100)
    #phis = np.linspace(0,2*np.pi,200)

    #best = min(conditional_entropy([t,p], rho) for t in thetas for p in phis)
    thetas = np.linspace(0, np.pi, 6)
    phis = np.linspace(0, 2 * np.pi, 6, endpoint=False)

    best = min(
    minimize(conditional_entropy, x0=[t, p], args=(rho,),
              method="Nelder-Mead").fun
    for t in thetas
    for p in phis
    )
    J = S_A -best 

    return mutual_info-J


def make_channel(U,p):
    U1,U2 = U
    p0 = p
    p1 = 1-p0
    kruas_ops = [np.sqrt(p0)*U1.data, np.sqrt(p1)*U2.data]
    channel = Kraus(kruas_ops)
    return channel


#create a random unitary matrix following the method in "Random unitary matrices" K Zyczkowski and M Kus 1994 J.Phys.A:Math.Gen. 27 4235
def random_unitary(n_qubits: int):
    N = 2**n_qubits #dimension of Hilbert space

    alpha = np.random.uniform(0,2*np.pi)
    U = np.eye(N, dtype = complex)*np.exp(1j *alpha)

    #multiply composite transformations E_1,.., E_2

    for s in range(1,N):
        for r in range(s-1,-1,-1):
            r_param = r+1
            xi = np.random.uniform(0,1)
            phi = np.arcsin(xi**(1/(2**r_param)))
            psi = np.random.uniform(0,2*np.pi)

            #chi non zero only when r=0
            chi = np.random.uniform(0,2*np.pi) if r==0 else 0 

            #construct elementary transformation E_s,r
            E = np.eye(N, dtype=complex)
            E[r,r] = np.cos(phi)*np.exp(1j * psi)
            E[r, s] = np.sin(phi) * np.exp(1j * chi)
            E[s, r] = -np.sin(phi) * np.exp(-1j * chi)
            E[s, s] = np.cos(phi) * np.exp(-1j * psi)

            U = U@E

    return U
    











