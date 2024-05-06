import matplotlib.pyplot as plt
import scipy.linalg
import scipy as sp
import scipy.sparse.linalg as sp_linalg
import numpy as np
import HamBuilder as hb
import ExactDiagScripts as ED
import copy

stem = '/Users/samuelbegg/Documents/Projects/Sensing/ed_results/'
save = 'yes'  #'yes' or 'no'
plot = 'yes'  #'yes' or 'no'


##############################################################################
#Inputs

tsteps = 500
step = 0.01
tvec = step*np.arange(0,tsteps)
real = 1 #1 for real time 0 for imaginary time
#Fisher information parameters
shift = 0.0001
tol = 10**(-5)

#############################################################################
#Build Hamiltonian

N = 4
Jz = 0.0
gamma = 1.0
Jx = -1.0*(1 + gamma)
Jy = -1.0*(1 - gamma)
hx = 0.0 #-0.5
hy = 0.0
hz = -2.0 
PbC = 1 #periodic boundary conditions [PBC = 1 or 0 (yes or no)]

#############################################################################
#Initialize state

initial = 'manual' #'manual' or 'ground_state'

if initial == 'manual':
	down = [0,1]
	up = [1,0]
	psi_0 = copy.copy(up)
	psi_0 = np.kron(psi_0,up)
	psi_0 = np.kron(psi_0,up)
	psi_0 = np.kron(psi_0,up)
	#a = 0.0
	#psi_0 = hb.statebuild_bloch_transinvariant(1.0/np.sqrt(1 + a*np.conj(a)),a/np.sqrt(1 + a*np.conj(a)),N)      
	psi_initial = copy.deepcopy(psi_0)
	print(np.inner(np.conj(psi_0),psi_0),'checknorm')

psi_down = hb.statebuild_bloch_transinvariant(1,0,N)

#######################################
#Variables to Initialise

k = 2**N 
nummode = k
Sx = 0.5*np.asarray([[0,1.0],[1.0,0]])                                                                                                                                                                             
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,0]])                                                                                                                                                                          
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])   
st_v_ov = np.zeros([2**N,1],complex)                                                                                                                                                     
Overlapst = np.zeros([tsteps],dtype = complex)                                                                                                  

###############################################################################
#one-dimensional case

if N == 2:
		H = Jz*np.kron(Sz,Sz) + Jx*np.kron(Sx,Sx) + Jy*np.kron(Sy,Sy) + hx*np.kron(np.identity(2),Sx) + hx*np.kron(Sx,np.identity(2)) + hy*np.kron(np.identity(2),Sy) + hy*np.kron(Sy,np.identity(2))  + hz*np.kron(np.identity(2),Sz) + hz*np.kron(Sz,np.identity(2))

else:
		H = Jx*hb.tens(Sx,Sx,N,0,PbC) +  Jy*hb.tens(Sy,Sy,N,0,PbC) + hx*hb.tens(Sx,np.identity(2),N,1,0) + hy*hb.tens(Sy,np.identity(2),N,1,0) +  hz*hb.tens(Sz,np.identity(2),N,1,0)+ Jz*hb.tens(Sz,Sz,N,0,PbC) 

																													
###############################################################################
#perform the diagonlization

w,v = np.linalg.eigh(H) 

################################################################################
#Collect overlaps, normalize eigenvectors, for calculating observables

if initial=='ground_state':
	psi_0 = copy.copy(v[:,0])
	psi_initial = copy.copy(psi_0)

if real == 1:
	r = 1.0j
else:
	r = 1.0
for i in range(0,k):
	v[:,i] = v[:,i]/sum(v[:,i]*np.conj(v[:,i]))
	st_v_ov[i] = np.inner(np.conj(v[:,i]),psi_0)
	
st_v_ovNORM = np.conj(st_v_ov)*st_v_ov

#Check that squared overlaps add up to 1
print(sum(st_v_ovNORM),'Should be 1, overlap when t = 0')

sz = ED.Sz()
sy = ED.Sy()
sx = ED.Sx()

# Initial Observables Operators
magMatrixZ = hb.tens(sz,np.identity(2),N,1,0)
magMatrixX = hb.tens(sx,np.identity(2),N,1,0)
magMatrixY = hb.tens(sy,np.identity(2),N,1,0)
corrMat = hb.tens(sz,sz,N,0,PbC)
corrMatY = hb.tens(sy,sy,N,0,PbC)
corrMatX = hb.tens(sx,sx,N,0,PbC)

# Initial Observables Measurement `Outputs'
energy = np.zeros([tsteps,1],dtype = complex)
magsites = np.zeros([N,tsteps],dtype = complex)
magsitesY = np.zeros([N,tsteps],dtype = complex)
magsitesX = np.zeros([N,tsteps],dtype = complex)
magnetisationZ = np.zeros([tsteps,1],dtype = complex)
magnetisationX = np.zeros([tsteps,1],dtype = complex)
magnetisationY = np.zeros([tsteps,1],dtype = complex)
normalisation = np.zeros([tsteps,1],dtype = complex)
entropy = np.zeros([N,tsteps],dtype = complex)
logNegL = np.zeros([tsteps,1],dtype = complex)
Overlap = np.zeros([tsteps,1],dtype = complex)
OverlapFLIP = np.zeros([tsteps,1],dtype = complex)
correlation = np.zeros([tsteps,1],dtype = complex)
correlationY = np.zeros([tsteps,1],dtype = complex)
correlationX = np.zeros([tsteps,1],dtype = complex)
magtimeav = np.zeros([tsteps,1])
Fisher = np.zeros(tsteps)

wavefunction = []


#calculate observables at every time
for ii in range(0,np.size(tvec)):
	print('Obs',ii)

	wavefunction = ED.wavefun_gen(tvec[ii],v,w,st_v_ov,nummode,real)
	densityMatrix = ED.densitymatrix(tvec[ii],v,wavefunction)

	Overlap[ii] = np.inner(np.conj(np.transpose(psi_initial)),wavefunction)
	OverlapFLIP[ii] = np.inner(np.conj(np.transpose(psi_down)),wavefunction)
	normalisation[ii] = np.trace(densityMatrix[0,:,:])
	magnetisationZ[ii] = (1.0/float(N))*np.trace(np.dot(magMatrixZ,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
	magnetisationY[ii] = (1.0/float(N))*np.trace(np.dot(magMatrixY,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
	magnetisationX[ii] = (1.0/float(N))*np.trace(np.dot(magMatrixX,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
	correlation[ii] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMat,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
	correlationX[ii] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatX,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
	correlationY[ii] = (1.0/float(N-1+PbC))*np.trace(np.dot(corrMatY,densityMatrix[0,:,:])/np.trace(densityMatrix[0,:,:]), axis1 = 0, axis2 = 1) #first axis is time 
	magMatrix = []	
		
	for ee in range(0,N):
		magMatrix = magMatrix + [hb.tens_single(Sz,np.identity(2),ee,(ee+1)%N,N)]
		magsites[ee,ii] = np.trace(np.dot(magMatrix[ee],densityMatrix[0,:,:]))/np.trace(densityMatrix[0,:,:])

	magMatrix = []	
	for ee in range(0,N):
		magMatrix = magMatrix + [hb.tens_single(Sy,np.identity(2),ee,(ee+1)%N,N)]
		magsitesY[ee,ii] = np.trace(np.dot(magMatrix[ee],densityMatrix[0,:,:]))/np.trace(densityMatrix[0,:,:])
	magMatrix = []	

	for ee in range(0,N):
		magMatrix = magMatrix + [hb.tens_single(Sx,np.identity(2),ee,(ee+1)%N,N)]
		magsitesX[ee,ii] = np.trace(np.dot(magMatrix[ee],densityMatrix[0,:,:]))/np.trace(densityMatrix[0,:,:])

	for NA in range(1,N):
		dimsA = 2**NA
		NB = N - NA
		dimsB = 2**NB
		reduced_density = ED.red_den(dimsA,dimsB,densityMatrix[0,:,:]/np.trace(densityMatrix[0,:,:])) 
		#entropy[NA-1,ii] = -np.trace(np.dot(reduced_density,scipy.linalg.logm(reduced_density)))


wavefunction = []
red_den = []
#reduced density matrix for Fisher information
for ii in range(0,np.size(tvec)):
	print('RedA',ii)
	wavefunction = ED.wavefun_gen(tvec[ii],v,w,st_v_ov,nummode,real)
	densityMatrix = ED.densitymatrix(tvec[ii],v,wavefunction)
	NA = 2
	dimsA = 2**(NA)	
	NB = N - NA
	dimsB = 2**NB
	reduced_density = ED.red_den(dimsA,dimsB,densityMatrix[0,:,:]/np.trace(densityMatrix[0,:,:])) 
	red_den = red_den + [reduced_density]

# Consider the shifted result
hz = hz - shift
if N == 2:
		H = Jz*np.kron(Sz,Sz) + Jx*np.kron(Sx,Sx) + Jy*np.kron(Sy,Sy) + hx*np.kron(np.identity(2),Sx) + hx*np.kron(Sx,np.identity(2)) + hy*np.kron(np.identity(2),Sy) + hy*np.kron(Sy,np.identity(2))  + hz*np.kron(np.identity(2),Sz) + hz*np.kron(Sz,np.identity(2))

else:
		H = Jx*hb.tens(Sx,Sx,N,0,PbC) +  Jy*hb.tens(Sy,Sy,N,0,PbC) + hx*hb.tens(Sx,np.identity(2),N,1,0) + hy*hb.tens(Sy,np.identity(2),N,1,0) +  hz*hb.tens(Sz,np.identity(2),N,1,0)+ Jz*hb.tens(Sz,Sz,N,0,PbC) 
																											
###############################################################################
#perform the diagonlization

w,v = np.linalg.eigh(H) 

################################################################################
#Collect overlaps, normalize eigenvectors, for calculating observables

if real == 1:
	r = 1.0j
else:
	r = 1.0
for i in range(0,k):
	v[:,i] = v[:,i]/sum(v[:,i]*np.conj(v[:,i]))
	st_v_ov[i] = np.inner(np.conj(v[:,i]),psi_0)
	
st_v_ovNORM = np.conj(st_v_ov)*st_v_ov

wavefunction = []
#reduced density matrix for Fisher information
red_denB = []
for ii in range(0,np.size(tvec)):
	print('RedB',ii)
	wavefunction = ED.wavefun_gen(tvec[ii],v,w,st_v_ov,nummode,real)
	densityMatrix = ED.densitymatrix(tvec[ii],v,wavefunction)
	NA = 2
	dimsA = 2**(NA)	
	NB = N - NA
	dimsB = 2**NB
	reduced_density = ED.red_den(dimsA,dimsB,densityMatrix[0,:,:]/np.trace(densityMatrix[0,:,:])) 
	red_denB = red_denB + [reduced_density]


for ii in range(0,np.size(Fisher)):
	
    Fisheradd = 0
	
    Dred = (red_denB[ii] - red_den[ii])/shift

    w, v = sp_linalg.eigsh(red_den[ii])
	
    for nn in range(0,np.size(w)):
		
        for mm in range(0,np.size(w)):
           
            if np.abs(w[mm] + w[nn]) > tol:
				
                Fisheradd += 2*np.real(( np.conj(v[:,nn])  @  Dred  @ v[:,mm]  ) * ( np.conj(v[:,mm])  @  Dred  @ v[:,nn]  ) ) / ( w[mm] + w[nn] )

    Fisher[ii] = copy.copy(Fisheradd)

#save outputs to desktop and plot
if save == 'yes':
	np.save(stem + '/magsites',magsites)
	np.save(stem + '/magsitesY',magsitesY)
	np.save(stem + '/magsitesX',magsitesX)
	np.save(stem + '/sigmaX',magnetisationX)
	np.save(stem + '/overlap',Overlap)
	np.save(stem + '/normalisation',normalisation)
	np.save(stem + '/sigmaY',magnetisationY)
	np.save(stem + '/sigmaZ',magnetisationZ)
	np.save(stem + '/sigmaY',magnetisationY)
	np.save(stem + '/sigmaZsigmaZ',correlation)
	np.save(stem + '/sigmaXsigmaX',correlationX)
	np.save(stem + '/sigmaYsigmaY',correlationY)
	np.save(stem + '/entropy',entropy)
	np.save(stem + '/lognegativity',logNegL)
	np.save(stem + '/energy',energy)
	np.save(stem + '/Fisher',Fisher)

if plot == 'yes':
	plt.plot(tvec,logNegL,'y',label= 'neg')
	plt.plot(tvec,magnetisationX,label = 'X')
	plt.plot(tvec,magnetisationY,label = 'Y')
	plt.plot(tvec,magnetisationZ,label = 'Z')
	plt.plot(tvec,correlation,label = 'zz+1')
	plt.plot(tvec,correlationX,label = 'xx+1')
	plt.plot(tvec,correlationY,label = 'yy+1')
	plt.plot(tvec[1::],energy[1::],label = 'energy')
	plt.plot(tvec,entropy[0,:],label = 'entropy')
	plt.plot(tvec,Overlap*np.conj(Overlap),'x',label = 'Probability Overlap')
	plt.plot(tvec,normalisation,label = 'Normalisation')
	plt.legend()
	plt.ylim(-2,2)
	plt.show()

for mm in range(0,N):
	plt.plot(tvec,magsites[mm,:])
plt.show()
  
plt.plot(tvec,Fisher)
plt.show()



