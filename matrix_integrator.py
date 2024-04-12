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
stem_save = stem + 'Documents/Projects/Sensing/matrix_results/' 
save = 'yes'
plot = 'no'

##############
#Inputs
tsteps = 5000
step = 0.001
tvec = step*np.arange(0,tsteps)
N = 4

##############
#Hamiltonian Parameters

gamma = 0.4
Jx = -1.0*(1 + gamma)
Jy = -1.0*(1 - gamma)
hz_amp = 1.5
hz_period = 6.0/4.0 
h0_amp = 0.3
hz = -h0_amp - hz_amp*np.sin(2*np.pi/hz_period*step*np.arange(0,tsteps))
PbC = 1 #periodic boundary conditions

##############
#Initial State

initial_state = 'ground_state'

#Manually set the initial state
if initial_state == 'manual':

	down = np.asarray([0,1])

	up = np.asarray([1,0])

	plus = 1.0/np.sqrt(2.0)*np.asarray(down + up)

	psi_0 = copy.copy(up)

	psi_0 = np.kron(psi_0,up)

	psi_0 = np.kron(psi_0,up)

	psi_0 = np.kron(psi_0,up)

	psi_initial = copy.deepcopy(psi_0)

if initial_state == 'ground_state':         

	H = 0.25*Jx*hb.tens(Sx,Sx,N,0,PbC) +  0.25*Jy*hb.tens(Sy,Sy,N,0,PbC) +  0.5*hz[0]*hb.tens(Sz,np.identity(2),N,1,0) 

	w,v = np.linalg.eigh(H) 

	psi_0 = copy.copy(v[:,0])

	psi_initial = copy.copy(psi_0)
                                             
# Initial Observables Operators
sz = ED.Sz()
sy = ED.Sy()
sx = ED.Sx()
magMatrixZ = hb.tens(sz,np.identity(2),N,1,0)
magMatrixX = hb.tens(sx,np.identity(2),N,1,0)
magMatrixY = hb.tens(sy,np.identity(2),N,1,0)
corrMat = hb.tens(sz,sz,N,0,PbC)
corrMatX = hb.tens(sx,sx,N,0,PbC)
corrMatY = hb.tens(sy,sy,N,0,PbC)

NA = int(N/2)
dimsA = 2**NA
NB = N - NA
dimsB = 2**NB

# Initial Observables Measurement `Outputs'
logNegL = np.zeros([tsteps,1],dtype = complex)
logNegR = np.zeros([tsteps,1],dtype = complex)
magsites = np.zeros([N,tsteps],dtype = complex)
magnetisationZ = np.zeros([tsteps,1],dtype = complex)
magnetisationX = np.zeros([tsteps,1],dtype = complex)
magnetisationY = np.zeros([tsteps,1],dtype = complex)
normalisation = np.zeros([tsteps,1],dtype = complex)
entropyL = np.zeros([tsteps,1],dtype = complex)
entropyR = np.zeros([tsteps,1],dtype = complex)
correlation = np.zeros([tsteps,1],dtype = complex)
correlationY = np.zeros([tsteps,1],dtype = complex)
correlationX = np.zeros([tsteps,1],dtype = complex)
returnprob = np.zeros([tsteps,1],dtype = complex)
magtimeav = np.zeros([tsteps,1])
rho = np.outer(psi_initial,np.conjugate(np.transpose(psi_initial)))
rho_init = copy.deepcopy(rho)
returnprob[0] = 1
magnetisationZ[0] = (1.0/float(N))*np.trace(np.dot(magMatrixZ,rho)/np.trace(rho)) 
magnetisationY[0] = (1.0/float(N))*np.trace(np.dot(magMatrixY,rho)/np.trace(rho)) 
magnetisationX[0] = (1.0/float(N))*np.trace(np.dot(magMatrixX,rho)/np.trace(rho)) 
correlation[0] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMat,rho)/np.trace(rho)) 
correlationX[0] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatX,rho)/np.trace(rho)) 
correlationY[0] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatX,rho)/np.trace(rho)) 
reduced_density = ED.red_denL(dimsA,dimsB,rho) #trace out the right 
entropyL[0] = -np.trace(np.dot(reduced_density,scipy.linalg.logm(reduced_density))) #left / right + bath  
reduced_density = ED.red_denR(dimsA,dimsB,rho) #trace out the left 
entropyR[0] = -np.trace(np.dot(reduced_density,scipy.linalg.logm(reduced_density))) #left + bath / right 
normalisation[0] = np.trace(rho[:,:])
magMatrix = []

for ee in range(0,N):

	magMatrix = magMatrix + [hb.tens_single(Sz,np.identity(2),ee,(ee+1)%N,N)]

	magsites[ee,0] = np.trace(np.dot(magMatrix[ee],rho))/np.trace(rho)

#calculate observables at every time
for ii in range(0,np.size(tvec)-1):
	
	print(ii)

	H = 0.25*Jx*hb.tens(Sx,Sx,N,0,PbC) +  0.25*Jy*hb.tens(Sy,Sy,N,0,PbC) +  0.5*hz[ii]*hb.tens(Sz,np.identity(2),N,1,0) 
	
	drho =    - 1j*(np.dot(H,rho)-np.dot(rho,H))
	
	rho = rho + drho*step

	rho = rho/np.real(np.trace(rho))

	for ee in range(0,N):

		magsites[ee,ii+1] = np.trace(np.dot(magMatrix[ee],rho))/np.trace(rho)

	magnetisationZ[ii+1] = (1.0/float(N))*np.trace(np.dot(magMatrixZ,rho))/np.trace(rho) 
	magnetisationY[ii+1] = (1.0/float(N))*np.trace(np.dot(magMatrixY,rho))/np.trace(rho) 
	magnetisationX[ii+1] = (1.0/float(N))*np.trace(np.dot(magMatrixX,rho))/np.trace(rho) 
	correlation[ii+1] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMat,rho))/np.trace(rho) 
	correlationX[ii+1] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatX,rho))/np.trace(rho) 
	correlationY[ii+1] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatY,rho))/np.trace(rho) 
	normalisation[ii+1] = np.trace(rho[:,:])
	returnprob[ii+1] = np.trace(np.dot(rho,rho_init))
	reduced_density = ED.red_denL(dimsA,dimsB,rho/np.trace(rho)) #trace out the right 
	entropyL[ii+1] = -np.trace(np.dot(reduced_density,scipy.linalg.logm(reduced_density))) #left / right + bath  
	reduced_density = ED.red_denR(dimsA,dimsB,rho/np.trace(rho)) #trace out the left 
	entropyR[ii+1] = -np.trace(np.dot(reduced_density,scipy.linalg.logm(reduced_density))) #left + bath / right 

if save == 'yes':

	np.save(stem_save + 'magsites',magsites)
	np.save(stem_save + 'sigmaX',magnetisationX)
	np.save(stem_save + 'normalisation',normalisation)
	np.save(stem_save + 'sigmaY',magnetisationY)
	np.save(stem_save + 'sigmaZ',magnetisationZ)
	np.save(stem_save + 'sigmaZsigmaZ',correlation)
	np.save(stem_save + 'sigmaXsigmaX',correlationX)
	np.save(stem_save + 'sigmaYsigmaY',correlationY)
	
if plot == 'yes':

	marker = ['x','o','s','+','^']

	for kk in range(0,N):

		plt.plot(magsites[kk,:],label = str(kk),marker = marker[kk])

	print(magsites[:,0])

	plt.legend()
	plt.show()


	plt.plot(tvec,magnetisationX,'rx',label = 'X')
	plt.plot(tvec,magnetisationY,'og',label = 'Y')
	plt.plot(tvec,magnetisationZ,'b',label = 'Z')
	plt.plot(tvec,correlation,'m',label = 'zz+1')
	plt.plot(tvec,correlationX,'m',label = 'xx+1')
	plt.plot(tvec,correlationY,'m',label = 'yy+1')
	plt.plot(tvec,entropyL,'g',label = 'entropyL')
	plt.plot(tvec,entropyR,'g',label = 'entropyR')
	plt.plot(tvec,normalisation,'k',label = 'Normalisation')
	plt.legend()
	plt.ylim(-2,2)
	plt.show()



