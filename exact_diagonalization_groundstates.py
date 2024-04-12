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

Sx = 0.5*np.asarray([[0,1.0],[1.0,0]])                                                                                                                                                                             
Sy = 0.5*np.asarray([[0,-1.0j],[1.0j,0]])                                                                                                                                                                          
Sz = 0.5*np.asarray([[1.0,0],[0,-1.0]])         


##############################################################################
#Fisher information parameters
shift = 0.00001
tol = 10**(-7)
subsystem = 2

#############################################################################
#Build Hamiltonian
N = 4
Jz = 0.0
gamma = 0.4
Jx = -1.0*(1 + gamma)
Jy = -1.0*(1 - gamma)
hx = 0.0 
hy = 0.0
phasepoints = 401 # points to sampling h0
hzmat = - np.linspace(-2.5,2.5,phasepoints) #leading minus sign to make equivalent to free fermion code 
PbC = 1 #periodic boundary conditions [PBC = 1 or 0 (yes or no)]
                                                                            
###############################################################################

Fisher = np.zeros(phasepoints)
spinZ= np.zeros(phasepoints)

for ii in range(0, phasepoints):

    hz = copy.copy(hzmat[ii])

    print(hz)

    if N == 2:
            
            H = Jz*np.kron(Sz,Sz) + Jx*np.kron(Sx,Sx) + Jy*np.kron(Sy,Sy) + hx*np.kron(np.identity(2),Sx) + hx*np.kron(Sx,np.identity(2)) + hy*np.kron(np.identity(2),Sy) + hy*np.kron(Sy,np.identity(2))  + hz*np.kron(np.identity(2),Sz) + hz*np.kron(Sz,np.identity(2))

    else:
            H = Jx*hb.tens(Sx,Sx,N,0,PbC) +  Jy*hb.tens(Sy,Sy,N,0,PbC) + hx*hb.tens(Sx,np.identity(2),N,1,0) + hy*hb.tens(Sy,np.identity(2),N,1,0) +  hz*hb.tens(Sz,np.identity(2),N,1,0)+ Jz*hb.tens(Sz,Sz,N,0,PbC) 

                                                                                                                        
    ###############################################################################
    #perform the diagonlization

    w,v = np.linalg.eigh(H) 

    ################################################################################

    #reduced density matrix for Fisher information
    densityMatrix = np.outer(v[:,0],np.conj(v[:,0]))
    NA = subsystem
    dimsA = 2**(NA)	
    NB = N - NA
    dimsB = 2**NB
    reduced_density = ED.red_den(dimsA,dimsB,densityMatrix/np.trace(densityMatrix)) 
    red_den = reduced_density

    spinZ[ii] = np.trace(red_den @ np.kron(Sz,np.identity(2)))/np.trace(red_den)

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

    #reduced density matrix for Fisher information
    densityMatrix = np.outer(v[:,0],np.conj(v[:,0]))
    NA = subsystem
    dimsA = 2**(NA)	
    NB = N - NA
    dimsB = 2**NB
    reduced_density = ED.red_den(dimsA,dimsB,densityMatrix/np.trace(densityMatrix)) 
    red_denB = reduced_density

    Fisheradd = 0
    
    Dred = (red_denB- red_den)/shift

    w, v = sp_linalg.eigsh(red_den)
    
    ignore = 0
    index = 0

    for nn in range(0,np.size(w)):
        
        for mm in range(0,np.size(w)):
        
            index += 1

            if np.abs(w[mm] + w[nn]) > tol:
                
                Fisheradd += 2*np.real(( np.conj(v[:,nn])  @  Dred  @ v[:,mm]  ) * ( np.conj(v[:,mm])  @  Dred  @ v[:,nn]  ) ) / ( w[mm] + w[nn] )

            else:

                ignore += 1
          
    print('Ignored fraction ', ignore/index)

    Fisher[ii] = copy.copy(Fisheradd)

#save outputs to desktop and plot
if save == 'yes':
    np.save(stem + '/Fisher',Fisher)
    np.save(stem + '/hzmat',-hzmat)
    np.save(stem + '/mag',spinZ)
  
print(Fisher)



