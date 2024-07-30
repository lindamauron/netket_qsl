import numpy as np 

from tqdm.auto import tqdm
from fractions import Fraction

import os
from jax import vmap

import qsl as qsl

folder = f'Noise/'
if not os.path.exists(folder):
    os.makedirs(os.path.dirname(folder+'/'), exist_ok=True)
os.makedirs(os.path.dirname(folder+'/logs/'), exist_ok=True)

# One can simply put any of these to 0 and it works (even sigmas)
p0 = 0
p1 = 2*np.pi/150    #|r><g|
p2 = 2*np.pi/80     #|g><r|
sigmaN = 3e-2
sigmaX = 2e-2

############################ Define system ############################
lattice = qsl.lattice.Torus(1.0, 2, 4)
N = lattice.N

hi = qsl.hilbert.TriangleHilbertSpace(lattice)
print('Hilbert space has ', N, 'particles.' )

############################ Now our state ############################
psi0 = np.zeros(hi.n_states, dtype=complex)
psi0[0] = 1
psi0 = psi0/np.linalg.norm(psi0)


sweep_time = 2.5
freq = qsl.frequencies.Cubic(sweep_time)

############################ All operators for measuring ############################
# Hamiltonian
# load the operators => H(t) = -freq.Ω(t)/2*X_op - freq.Δ(t)*N_op + freq.Ωf*V_op
H = qsl.operators.Rydberg_Hamiltionian(hi,lattice,freq)
X_op, N_op, V_op = H.for_sparse()
print(H)

############################ Time evolution ############################
# operators for noisy Hamiltonian
# In case of noise, we define ab[i] = |a><b|_i
gg = []
gr = []
rg = []
rr = []
rggr = []
grrg = []
LtLs = 0
Xs = []
for i in range(N):
    gg.append( qsl.operators.g_occ(hi,i).to_sparse() )
    rr.append( qsl.operators.r_occ(hi,i).to_sparse() )
    rg.append( qsl.operators.sigma_minus(hi,i).to_sparse() )
    gr.append( qsl.operators.sigma_plus(hi,i).to_sparse() )
    rggr.append( rg[-1]@gr[-1] )
    grrg.append( gr[-1]@rg[-1] )
    LtLs += p0*gg[-1] + p0*rr[-1] + p1*grrg[-1] + p2*rggr[-1]
    Xs.append( rg[-1]+gr[-1] )
    
# ode : d psi(t) / dt = -i H(t) psi(t) - 1/2 sum_j L_j^t L_j psi(t)
def psi_dot(t,psi):
    psi_dot = (1j*freq.Ω(t)/2)*(noisy_X@psi) + (1j*freq.Δ(t))*(noisy_N@psi) + (-1j*freq.Ωf)*(V_op@psi) 
    return psi_dot + (- 1/2) * (LtLs@psi)


def integrate(t,y,h):
    k1 = psi_dot(t, y)
    k2 = psi_dot(t+h/2, y+k1*h/2)
    k3 = psi_dot(t+h/2, y+k2*h/2)
    k4 = psi_dot(t+h, y+k3*h)
    return t+h, y+h*(k1+2*k2+2*k3+k4)/6

############################ Measurements ############################
# Measurements
def expect(psi, Op):
    if np.issubdtype(type(Op), np.number):
        return Op*(psi.conj().T @ psi)
    return psi.conj().T @ (Op @ psi)

P_op = qsl.operators.P(hi, sites=lattice.hexagons.bulk[0]).to_sparse()
Q_op = qsl.operators.Q(hi, sites=lattice.hexagons.bulk[0]).to_sparse()

from qsl.operators import dimer_probs as _local_probs
restr_basis = hi.all_states()
def dimer_probs(lattice, psi):
    def pi(s,u):
        ps = _local_probs(lattice, s)
        return ps * u.conj()*u

    ps = np.mean( vmap(pi, in_axes=(0,0), out_axes=0)(restr_basis, psi), axis=0 )
    return ps/ps.sum()


def measures(psi,log,t):
    norm = (psi.conj().T @ psi).real
    X_value = expect(psi, X_op)/norm
    N_value = expect(psi, N_op)/norm
    V_value = expect(psi, V_op)/norm

    log['Generator'].append( ( -freq.Ω(t)/2*X_value - freq.Δ(t)*N_value  + freq.Ωf*V_value ).real )
    log['n'].append( ( N_value/N ).real )
    log['time'].append( ( t ).real )
    log['Delta'].append( ( freq.Δ(t)/freq.Ωf ).real )
    log['Omega'].append( ( freq.Ω(t)/freq.Ωf ).real )

    log['P'].append( expect(psi, P_op).real/norm )
    log['Q'].append( expect(psi, Q_op).real/norm )

    probs = dimer_probs(lattice,psi).tolist()
    for i,op in enumerate(['monomer', 'dimer', 'double dimer', 'triple dimer', 'quadruple dimer']):
        log[op].append( probs[i].real )

    return

############################ FM ############################
Ps = []
Qs = []
PFMs = []
QFMs = []
for i in range(4):
    Ps.append(qsl.operators.P(hi,sites=lattice.hexagons.bulk[i]).to_sparse())
    Qs.append(qsl.operators.Q(hi,sites=lattice.hexagons.bulk[i]).to_sparse())

for i in range(4):
    for k in range(6):
        sites = np.arange(k,k+3)%6
        PFMs.append(qsl.operators.P(hi,sites=lattice.hexagons.bulk[i][sites]).to_sparse())
        QFMs.append(qsl.operators.Q(hi,sites=lattice.hexagons.bulk[i][sites]).to_sparse())

def FM_measures(psi,j):
    norm = (psi.conj().T @ psi).real
    P = 0
    Q = 0
    PFM = 0
    QFM = 0
    for i in range(4):
        P += expect(psi, Ps[i])/4
        Q += expect(psi, Qs[i])/4
        for k in range(6):
            PFM += expect(psi, PFMs[6*i+k])/6/4
            QFM += expect(psi, QFMs[6*i+k])/6/4

    P_meas[simul,j] = P/norm
    Q_meas[simul,j] = Q/norm
    PFM_meas[simul,j] = PFM/norm
    QFM_meas[simul,j] = QFM/norm

    return 

#################################### Jump operators ########################################
def apply_L0(psi,p0_t,i):
    return np.sqrt(p0)* (gg[i]@psi - (rr[i]@psi) ) /np.sqrt(p0_t)

def apply_L1(psi,p1_t,i):
    return np.sqrt(p1)* (rg[i]@psi) /np.sqrt(p1_t)

def apply_L2(psi,p2_t,i):
    return np.sqrt(p2)* (gr[i]@psi) /np.sqrt(p2_t)

apply_L = [apply_L0, apply_L1, apply_L2]

def jump(psi,p0,p1,p2):
    norm2 = psi.conj().T @ psi
    rrs = np.array([expect(psi, rggr[i]) for i in range(N)])
    ggs = np.array([expect(psi, grrg[i]) for i in range(N)])

    probs = np.zeros((N,3), dtype=float)
    probs[:,0] = (p0*norm2).real
    probs[:,1] = (p1*ggs).real
    probs[:,2] = (p2*rrs).real


    n = np.random.choice(np.arange(N*3), p=probs.reshape(-1)/probs.sum() )
    i = n//3 # which site
    j = n%3 # which jump
    
    psi = apply_L[j](psi,probs[i,j], i)


    return psi

#################################### Evolve ####################################
nsimuls = 100

P_meas = np.zeros((nsimuls,251), dtype=complex)
Q_meas = np.zeros((nsimuls,251), dtype=complex)
PFM_meas = np.zeros((nsimuls,251), dtype=complex)
QFM_meas = np.zeros((nsimuls,251), dtype=complex)

with open(folder+'/infos.md', 'w') as file:
    file.write('# Noisy time evolution \n')
    file.write('## Noise parameters \n')
    file.write(f'$p_0=2pi x {Fraction(p0/2/np.pi).limit_denominator(1000)}$, ')
    file.write(f'$p_1=2pi x {Fraction(p1/2/np.pi).limit_denominator(1000)}$, ')
    file.write(f'$p_2=2pi x {Fraction(p2/2/np.pi).limit_denominator(1000)}$ \n')
    file.write('variance of $\Delta$ : $\mathcal{N}(1,'+f'{sigmaN}'+')$ \n')
    file.write('variance of $\Omega$ : $\mathcal{N}(1,'+f'{sigmaX}'+')$ \n')

    file.write('## System\n')
    print(H, file=file)
    
for simul in range(nsimuls):
    deltas = np.random.normal(1.0,sigmaN,N)
    noisy_N = sum([deltas[j]*rr[j] for j in range(N)])

    omegas = np.random.normal(1.0,sigmaX,N)
    noisy_X = sum([omegas[j]*Xs[j] for j in range(N)])

    log = {}
    for k in ['Generator', 'n', 'time', 'Delta', 'Omega', 'P', 'Q', 'monomer', 'dimer', 'double dimer', 'triple dimer', 'quadruple dimer']:
        log[k] = []

    ############################ Numerical options for evolution ############################
    psi = psi0
    t = 0.0
    dt = 1e-3
    tf = sweep_time

    dt_stop=1e-2
    tstops = np.linspace(t,tf,int(tf/dt_stop+1))
    save_psi = False
    save_every = 100

    measuring = measures #exact measures

    norm = 1.0
    step_count = 0
    k = 0
    pjump = np.random.uniform(0,1)
    njump=0
    ############################ Time evolution ############################
    print(f'\n Let\'s start the evolution number {simul}')

    pbar = tqdm(total=tf,
            unit_scale=True,
            dynamic_ncols=True,
            )

    while t < tf:
        if tstops.size!=0 and (np.isclose(t, tstops[0]) or t > tstops[0]):
            k+=1
            tstops = tstops[1:]
            measuring(psi,log,t)
            FM_measures(psi,step_count//10)
            
            if save_psi and k%save_every==0:
                k=0
                np.save(folder+f'/states_{simul}/t={t:.4f}.npy', psi)    
        
        pbar.n = t
        pbar.set_postfix({'norm':f"{norm.real:.2f}", 'Generator':f"{log['Generator'][-1]:.2f}", 'n':step_count, 'jumps':njump})
        pbar.refresh()
        
        t, psi = integrate(t, psi, dt)
        step_count += 1

        norm = psi.conj().T@psi
        if norm <= 1 - pjump:
            psi = jump(psi,p0,p1,p2)
            pjump = np.random.uniform(0,1)
            njump +=1


    if tstops.size!=0 and (np.isclose(t, tstops[0]) or t > tstops[0]):
        k+=1
        tstops = tstops[1:]
        measuring(psi,log,t)
        FM_measures(psi,step_count//10)
        
        if save_psi and k%save_every==0:
            k=0
            np.save(folder+f'/states_{simul}/t={t:.4f}.npy', psi)
            
    for k in log.keys():
        log[k] = list(log[k])
    qsl.utils.save_log(folder+f'/logs/log_{simul}', log)
    np.save(folder+f'/P.npy', P_meas)
    np.save(folder+f'/Q.npy', Q_meas)
    np.save(folder+f'/PFM.npy', PFM_meas)
    np.save(folder+f'/QFM.npy', QFM_meas)

    pbar.close()