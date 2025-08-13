import matplotlib.pyplot as plt
import scipy.linalg
import scipy as sp
import scipy.sparse.linalg as sp_linalg
import numpy as np
import HamBuilder as hb
import ExactDiagScripts as ED
import copy

Sp = np.asarray([[0,1.0],[0.0,0]])                                                                                                 
Sm = np.asarray([[0,0.0],[1.0,0]])                                                                                                 
Sx = np.asarray([[0,1.0],[1.0,0]])                                                                                                 
Sy = np.asarray([[0,-1.0j],[1.0j,0]])                                                                                                       
Sz = np.asarray([[1.0,0],[0,-1.0]])   

stem = '/Users/samuelbegg/'
stem_save = stem + 'Documents/Sensing/matrix_results/' 
save = 'yes'
plot = 'yes'

#####################
#Inputs

tsteps = 200000
step = 0.0001
tvec = step*np.arange(0,tsteps)
N = 4 #system size
subsystem = 2 #Evaluate Fisher information for subsystem of this size

#####################
#Build Hamiltonian

Jz = 0.0
gamma = 0.5
Jx = -1.0*(1 + gamma)
Jy = -1.0*(1 - gamma)
hx = 0.0
hy = 0.0
hz_amp = 1.0
hz_period = 2*np.pi/1.0
h0_amp = 1.0
hz = -h0_amp - hz_amp*np.sin(2*np.pi/hz_period*step*np.arange(0,tsteps))
shift = 10**(-6)
tol = 10**(-12)
hz_shift = hz - shift
PbC = 1

#####################
#Initial State

initial_state = 'ground_state' #'ground_state' or 'manual'. If 'manual' alter the code below.

#####################

if initial_state == 'manual':

	down = np.asarray([0,1])
	up = np.asarray([1,0])

	psi_0 = copy.copy(up)
	psi_0 = np.kron(psi_0,up)
	psi_0 = np.kron(psi_0,up)
	psi_0 = np.kron(psi_0,up)
    
if initial_state == 'ground_state':                                        

	H = 0.25*Jx*hb.tens(Sx,Sx,N,0,PbC) +  0.25*Jy*hb.tens(Sy,Sy,N,0,PbC) +  0.5*hz[0]*hb.tens(Sz,np.identity(2),N,1,0) 

	w,v = scipy.linalg.eigh(H) 

	psi_0 = copy.copy(v[:,0])


psi_initial = copy.deepcopy(psi_0)
                                                                                              
# Initialize operators
sz = ED.Sz()
sy = ED.Sy()
sx = ED.Sx()

# Subsystem and bath dimensions
NA = int(subsystem)
dimsA = 2**NA
NB = N - NA
dimsB = 2**NB


# Initialize
Fisher = np.zeros(int(tsteps/100)+1)
rho = np.outer(psi_initial,np.conjugate(psi_initial))
rho_init = copy.copy(rho)
reduced_density = []
reduced_densityB = []

reduced_density = reduced_density + [ED.red_den(dimsA,dimsB,rho/np.trace(rho))] 

# Perform the integration
for ii in range(0,np.size(tvec)-1):

	H = 0.25*Jx*hb.tens(Sx,Sx,N,0,PbC) +  0.25*Jy*hb.tens(Sy,Sy,N,0,PbC) +  0.5*hz[ii]*hb.tens(Sz,np.identity(2),N,1,0)

	drho =    - 1j*(np.dot(H,rho)-np.dot(rho,H))

	rho = rho + drho*step

	rho = rho/np.real(np.trace(rho))

	if ii%100 == 0:

		reduced_density = reduced_density + [ED.red_den(dimsA,dimsB,rho/np.trace(rho))] 
		
		print(ii) 

# Now do the shifted evaluation h -> h + shift
		
if initial_state == 'ground_state':                                        

	H = 0.25*Jx*hb.tens(Sx,Sx,N,0,PbC) +  0.25*Jy*hb.tens(Sy,Sy,N,0,PbC) +  0.5*hz_shift[0]*hb.tens(Sz,np.identity(2),N,1,0) 

	w,v = scipy.linalg.eigh(H) 

	psi_0 = copy.copy(v[:,0])

	rho = np.outer(psi_0,np.conjugate(psi_0))
	
else:

	rho = copy.deepcopy(rho_init)

reduced_densityB = reduced_densityB + [ED.red_den(dimsA,dimsB,rho/np.trace(rho))] 

for ii in range(0,np.size(tvec)-1):
	
	H = 0.25*Jx*hb.tens(Sx,Sx,N,0,PbC) +  0.25*Jy*hb.tens(Sy,Sy,N,0,PbC) +  0.5*hz_shift[ii]*hb.tens(Sz,np.identity(2),N,1,0)
	
	drho =    - 1j*(np.dot(H,rho)-np.dot(rho,H))

	rho = rho + drho*step

	rho = rho/np.real(np.trace(rho))

	if ii%100 == 0:

		reduced_densityB = reduced_densityB + [ED.red_den(dimsA,dimsB,rho/np.trace(rho))] 
		
		print(ii)

# Calculate the Fisher information
		

for ii in range(0,np.size(reduced_density,0)):
	
	ignore = 0
	
	Fisheradd = 0

	Dred = (reduced_densityB[ii] - reduced_density[ii])/shift

	w, v = sp.linalg.eig(reduced_density[ii])

	for nn in range(0,np.size(w)):

		for mm in range(0,np.size(w)):

			if np.abs(w[mm] + w[nn]) > tol:

				Fisheradd += 2 * np.real(( np.conj(v[:,nn])  @  Dred  @ v[:,mm]  ) * ( np.conj(v[:,mm])  @  Dred  @ v[:,nn]  ) ) / ( w[mm] + w[nn] )

			else:

				ignore += 1

	Fisher[ii] = copy.copy(Fisheradd)
	
	print(ignore)


if save == 'yes':

	np.save(stem_save + 'Fisher', Fisher)
	np.save(stem_save + 'time2',tvec)

	
if plot == 'yes':

	plt.plot(step/100*np.arange(0,np.size(Fisher)),Fisher,'rx',label = 'X')
	plt.legend()
	plt.show()

print(reduced_density[0])
#print(reduced_densityB)
print(Fisher)



