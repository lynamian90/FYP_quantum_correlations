from qiskit import QuantumCircuit
import numpy as np
from scipy.optimize import minimize
from qiskit.quantum_info import DensityMatrix, partial_trace, entropy, Statevector, Kraus, entanglement_of_formation,Operator 
import matplotlib.pyplot as plt



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
    thetas = np.linspace(0, np.pi/2, 6)  # n and -n give the same measurement
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
def elementary_rotation_matrix(N:int,i:int,j:int,phi:float, psi:float,chi:float)-> np.ndarray:
    #Construct NxN elementary unitary rotation matrix E_{i,j}(\phi,psi,chi) as in Zyczkowski and Kus 1994
    E = np.eye(N, dtype=complex)
    cos,sin = np.cos(phi), np.sin(phi)

    E[i,i] = cos*np.exp(1j*psi)
    E[i,j] = sin*np.exp(1j*chi)
    E[j,i] = -sin*np.exp(-1j*chi)
    E[j,j] = cos*np.exp(-1j*psi)
    return E

def random_unitary_hurwitz(n_qubits: int) -> np.ndarray:
    N = 2**n_qubits

    #overall phase
    alpha = np.random.uniform(0, 2 * np.pi)
    U = np.eye(N, dtype=complex) * np.exp(1j * alpha)

    #Multiply composite transformations E_1, ..., E_{N-1}
    for r in range(1,N):
        #E_r acts on indices up to r+1
        E_r = np.eye(N, dtype=complex)
        for s in range(r,0,-1):
            #sample angles according to paper
            xi = np.random.uniform(0,1)
            phi = np.arcsin(xi**(1/(2*s)))
            psi = np.random.uniform(0, 2 * np.pi)

            #chi is non-zero only when s=1
            chi = np.random.uniform(0, 2 * np.pi) if s == 1 else 0

            E_elem = elementary_rotation_matrix(N, s-1, r, phi, psi, chi)
            E_r = E_r @ E_elem
        U = U @ E_r

    return U

def random_unitary_QR(n_qubits: int) -> np.ndarray:
    N = 2**n_qubits
    #sample a random complex matrix with entries from standard normal distribution
    Z = np.random.normal(size=(N,N)) + 1j * np.random.normal(size=(N,N))
    #QR decomposition
    Q,R = np.linalg.qr(Z)
    #make Q unitary by adjusting the phases of the diagonal of R
    D = np.diag(np.diag(R)/np.abs(np.diag(R)))
    U = Q @ D
    return U

def random_unitary(n_qubits: int, method = "hurwitz"):
    
    if method == "hurwitz":
        U = random_unitary_hurwitz(n_qubits)
    elif method == "QR":
        U = random_unitary_QR(n_qubits)
    
    else:
        raise ValueError(f"Unknown method: {method}")


    return U


#claude version of quantum discord calcultation (faster):
"""Quantum discord for a qubit (A, measured) + d-level system (B).
Follows Vinjanampathy & Rau, arXiv:1106.4488."""

def entropy(rho, tol=1e-12):
    """Von Neumann entropy in bits."""
    w = np.linalg.eigvalsh(rho)
    w = w[w > tol]
    return float(-np.sum(w * np.log2(w)))


def _cond_entropy(rho4, theta, phi):
    """sum_i p_i S(rho_i) after projective measurement on qubit A along (theta, phi).
    rho4 is rho reshaped to (2, d, 2, d).  Equivalent to the paper's
    'multiply the four blocks by cos^2(t/2), sin(t)e^{-i phi}/2, ... and sum' rule,
    but done for both outcomes n_+ and n_-."""
    n_plus = np.array([np.cos(theta / 2), np.exp(1j * phi) * np.sin(theta / 2)])
    n_minus = np.array([np.sin(theta / 2), -np.exp(1j * phi) * np.cos(theta / 2)])
    total = 0.0
    for n in (n_plus, n_minus):
        unnorm = np.einsum('a,aibj,b->ij', n.conj(), rho4, n)   # <n|rho|n>, d x d
        p = np.real(np.trace(unnorm))
        if p > 1e-12:
            total += p * entropy(unnorm / p)
    return total


def quantum_discord(rho, d=None, n_starts=8, seed=0):
    """Discord Q(rho^{AB}) in bits, measuring qubit A (first factor).
    rho : (2d x 2d) density matrix.  d inferred if omitted."""
    rho = np.asarray(rho, dtype=complex)
    d = d or rho.shape[0] // 2
    rho4 = rho.reshape(2, d, 2, d)
    rho_A = np.einsum('aibi->ab', rho4)
    rho_B = np.einsum('aiaj->ij', rho4)

    mutual_info = entropy(rho_A) + entropy(rho_B) - entropy(rho)

    f = lambda x: _cond_entropy(rho4, x[0], x[1])
    # candidates: the 'usual suspects' from the paper (theta = 0, pi/2; phi = 0, pi/2)
    cands = [(0, 0), (np.pi / 2, 0), (np.pi / 2, np.pi / 2)]
    best = min(f(c) for c in cands)
    # then a genuine optimization from random starts (theta in [0,pi/2], phi in [0,pi))
    rng = np.random.default_rng(seed)
    for _ in range(n_starts):
        x0 = [rng.uniform(0, np.pi / 2), rng.uniform(0, np.pi)]
        res = minimize(f, x0, method='Nelder-Mead',
                       options={'xatol': 1e-8, 'fatol': 1e-10})
        best = min(best, res.fun)

    classical_corr = entropy(rho_B) - best
    return max(mutual_info - classical_corr, 0.0)


#makes random unitary channel with n unitaries and random p values

def make_random_channel(n_unitaries:int =2, n_qubits:int = 2, method:str = "QR"):
    #Sample N unbiased probabilities from a symmetric dirichlet distribution 
    p_values = np.random.dirichlet(np.ones(n_unitaries))

    #generate N random unitaries
    kraus_ops = []
    for i in range(n_unitaries):
        U = random_unitary(n_qubits, method=method)
        kraus_ops.append(np.sqrt(p_values[i]) * U)

    return Kraus(kraus_ops)

