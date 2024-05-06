import numpy as np
from datetime import datetime
import copy
import scipy.linalg as lin


def initialize_state(sites, gamma, J, h0_in, h1_in, boundary_conditions, initial_state):
	#Build nearest neighbourn interaction Matrix
	Jmat = np.zeros([sites,sites])

	for mm in range(0,sites):

		Jmat[mm,(mm+1)%sites]  = copy.copy(J) 

	for mm in range(0,sites):

		Jmat[(mm+1)%sites,mm]  = copy.copy(J) 
		
	#Build Initial State

	#Initialize
	vec = np.zeros(sites)

	obs = np.zeros([sites,sites],dtype = complex) #correlation matrix <c^{dag}_i c_j>

	Dag_obs = np.zeros([sites,sites],dtype = complex) #correlation matrix <c^{dag}_i c^{dag}_j>

	if initial_state == 'flips':

		vec[0] =  1.0 #spin 1 flipped up. Note that an odd number is required to ensure correct parity with JW transformation.
		
		vec[1] =  0.0

		for m in range(0,sites):

			for n in range(0,sites):
				
				if m == n:

					obs[m,n] = vec[m]*vec[n]  

					Dag_obs[m,n] = 0.0 

	if initial_state == 'momentum':
		
		for m in range(0,sites):

			for n in range(0,sites):

				if m == n:

					obs[m,n] = 1.0/sites

					Dag_obs[m,n] = 0.0       


	#If use this option must have N odd
	if initial_state == 'spin_up':
		
		for m in range(0,sites):

			for n in range(0,sites):

				if m == n:

					obs[m,n] = 1.0

					Dag_obs[m,n] = 0.0
				
				else:

					obs[m,n] = 0.0
					
					Dag_obs[m,n] = 0.0


	if initial_state == 'ground_state':
		
		#note that this is also implemented in loop for the different points of the phase diagram.
		obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in, h1_in, boundary_conditions)

	return obs, Dag_obs


def correlation_groundstate(sites, gamma, J , h0_in, h1_in, boundary_conditions):
    
	obs = np.zeros([sites,sites],dtype = complex) 

	Dag_obs = np.zeros([sites,sites],dtype = complex) 

	thetaK = np.zeros(sites)

	if boundary_conditions == 'ABC':

		fourier_def = 1

	# elif boundary_conditions == 'ABC' and (sites%2)==1:

	# 	fourier_def = 0

	# elif boundary_conditions == 'PBC' and (sites%2)==0:

	# 	fourier_def = 0

	elif boundary_conditions == 'PBC':

		fourier_def = 0

	
	if fourier_def == 0:

		#for kk in range(0,sites):
				
		kval = -np.pi + 2*np.arange(0,sites)*np.pi/sites
				
		thetaK = np.arctan2( J * gamma * np.sin(kval), J * np.cos(kval) + h0_in + h1_in)

		for m in range(0,sites):
			
			for n in range(0,sites):

				#for kk in range(0,sites):

				#kval = -np.pi + 2*kk*np.pi/sites

				Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK))

				obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK/2.0))**2)
					
	if fourier_def == 1:

		#for kk in range(0,int(sites)):
				
		kval = -np.pi + (2*np.arange(0,sites)+1)*np.pi/sites #(2*(kk) + 1)*np.pi/sites

		thetaK = np.arctan2( J * gamma * np.sin(kval), J * np.cos(kval) + h0_in + h1_in)

		for m in range(0,sites):
			
			for n in range(0,sites):

				#for kk in range(0,int(sites)):

				#kval = -np.pi + (2*kk+1)*np.pi/sites #(2*(kk) + 1)*np.pi/sites

				Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK))

				obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK/2.0))**2)
	
	return obs, Dag_obs

#def correlation_groundstate_loops(sites, gamma, J , h0_in, h1_in, boundary_conditions):
    
	obs = np.zeros([sites,sites],dtype = complex) 

	Dag_obs = np.zeros([sites,sites],dtype = complex) 

	thetaK = np.zeros(sites)

	if boundary_conditions == 'ABC' and (sites%2)==0:

		fourier_def = 1

	# elif boundary_conditions == 'ABC' and (sites%2)==1:

	# 	fourier_def = 0

	# elif boundary_conditions == 'PBC' and (sites%2)==0:

	# 	fourier_def = 0

	elif boundary_conditions == 'PBC' and (sites%2)==1:

		fourier_def = 0

	
	if fourier_def == 0:

		for kk in range(0,sites):
				
				kval = -np.pi + 2*kk*np.pi/sites
				
				thetaK[kk] = np.arctan2( J * gamma * np.sin(kval), J * np.cos(kval) + h0_in + h1_in)

		for m in range(0,sites):
			
			for n in range(0,sites):

				for kk in range(0,sites):

					kval = -np.pi + 2*kk*np.pi/sites

					Dag_obs[m,n] += 1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK[kk])

					obs[m,n] += 1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK[kk]/2.0))**2
					
	if fourier_def == 1:

		for kk in range(0,int(sites)):
				
				kval = -np.pi + (2*kk+1)*np.pi/sites #(2*(kk) + 1)*np.pi/sites

				thetaK[kk] = np.arctan2( J * gamma * np.sin(kval), J * np.cos(kval) + h0_in + h1_in)

		for m in range(0,sites):
			
			for n in range(0,sites):

				for kk in range(0,int(sites)):

					kval = -np.pi + (2*kk+1)*np.pi/sites #(2*(kk) + 1)*np.pi/sites

					Dag_obs[m,n] += 1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK[kk])

					obs[m,n] += 1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK[kk]/2.0))**2
	
	return obs, Dag_obs


def Mmatrix(sites, boundary_conditions):
	
	M = np.zeros([sites,sites])

	for ii in range(0,sites):
		
		for jj in range(0,sites):
			
			if (ii - 1)%sites  == jj:

				M[ii,jj] = 1.0

			if (ii + 1)%sites == jj:

				M[ii,jj] = 1.0

			#Flip the couplings if we have ABC
			if boundary_conditions == 'ABC':

				if ii == 0 and jj == sites - 1:

					M[ii,jj] = - M[ii,jj]

				if jj == 0 and ii == sites - 1:

					M[ii,jj] = - M[ii,jj]

	return M


def Nmatrix(sites, boundary_conditions):
	
	N = np.zeros([sites,sites])

	for ii in range(0,sites):
		
		for jj in range(0,sites):
			
			if (ii - 1)%sites == jj:

				N[ii,jj] = -1.0

			if (ii + 1)%sites == jj:

				N[ii,jj] = 1.0

			#Flip the couplings if we have ABC
			if boundary_conditions == 'ABC':
				
				if ii == 0 and jj == sites - 1:

					N[ii,jj] = - N[ii,jj]

				if jj == 0 and ii == sites - 1:

					N[ii,jj] = - N[ii,jj]

	return N


def integrator_matrices(obs, Dag_obs, J, gamma, h0, h1, times, dt, measure_interval, sites, boundary_conditions, method):

	startTime = datetime.now()
	 
	M = Mmatrix(sites, boundary_conditions)

	N = Nmatrix(sites, boundary_conditions)
	
	#Initialize integration variables
	dobs_predmat = np.zeros([sites,sites],dtype = complex)

	Dag_dobs_predmat = np.zeros([sites,sites],dtype = complex)

	if method == 'heun' or 'RK4':

		dobs_mat = np.zeros([sites,sites],dtype = complex)

		Dag_dobs_mat = np.zeros([sites,sites],dtype = complex)
		
	#Initialize observables
	corr_diag = np.zeros([sites,times],dtype = complex)

	#Initialize output correlation matrices
	Corr_mat_list = []

	Dag_mat_list = []

	kk = 0 # variable updated when measurements made (for storing outputs)

	dobs = np.zeros([sites,sites])

	Dag_dobs = np.zeros([sites,sites])

	
	for ii in range(0,times):

		if (ii%1000) == 0:

			print(ii)

		Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs))) #Refer to EOM in notes for reasoning here. Have terms c^{dag}_m c^{dag}_n and c_m c_n, whereas conjugate transpose of first term is c_n cm, expect minus sign but gives wrong answer.
		
		#retain for Heun and RK4 schemes
		orig_obs = copy.deepcopy(obs)
		
		Dag_orig_obs = copy.deepcopy(Dag_obs)

		dobs = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

		Dag_dobs = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0 + h1[ii])*Dag_obs
		
		#save matrix elements for Heun and RK4 schemes
		dobs_predmat = copy.copy(dobs)

		Dag_dobs_predmat  = copy.copy(Dag_dobs)

		obs = obs + 1j*dobs_predmat*dt	

		Dag_obs = Dag_obs + 1j*Dag_dobs_predmat*dt		

		
		if method == 'heun':
			#Correction Step

			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			dobs = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0 + h1[ii])*Dag_obs

			dobs_mat  = copy.copy(dobs)
					
			Dag_dobs_mat  = copy.copy(Dag_dobs)		

			obs = orig_obs + 1j*dt*0.5*(dobs_mat + dobs_predmat)

			Dag_obs = Dag_orig_obs + 1j*dt*0.5*(Dag_dobs_mat + Dag_dobs_predmat)

		if method == 'RK4':
			#Runge Kutta 4th order
			
			#Define k1 and input for k2

			k1 =  copy.copy(1j*dobs_predmat)

			Dag_k1 =  copy.copy(1j*Dag_dobs_predmat)

			obs = orig_obs + 0.5*k1*dt #input for k2 (like a midpoint)

			Dag_obs = Dag_orig_obs + 0.5*Dag_k1*dt #input for Dag_k2 (like a midpoint)

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))

			dobs_mat = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs_mat = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0 + h1[ii])*Dag_obs
			
			######### 
			#Define k2 and input for k3
							
			k2 =  copy.copy(1j*dobs_mat)

			Dag_k2 =  copy.copy(1j*Dag_dobs_mat)

			obs = orig_obs + 0.5*k2*dt #input for k3 (like a midpoint)

			Dag_obs = Dag_orig_obs + 0.5*Dag_k2*dt #input for Dag_k3 (like a midpoint)

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))

			dobs_mat = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs_mat = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0 + h1[ii])*Dag_obs

			######### 
			#Define k3 and input for k4	
					
			k3 =  copy.copy(1j*dobs_mat)

			Dag_k3 =  copy.copy(1j*Dag_dobs_mat)

			obs = orig_obs + k3*dt #input for k4

			Dag_obs = Dag_orig_obs + Dag_k3*dt #input for Dag_k4

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			dobs_mat = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs_mat = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0 + h1[ii])*Dag_obs	

			######### 
			#Define k4
			
			k4 = 1j*dobs_mat

			Dag_k4 = 1j*Dag_dobs_mat

			#Evaluate the final RK4 expression

			obs = orig_obs + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

			Dag_obs = Dag_orig_obs + dt/6.0*(Dag_k1 + 2*Dag_k2 + 2*Dag_k3 + Dag_k4)


		if (ii+1) < int(times)+1:

			if (ii+1)%measure_interval == 0:
	
				Corr_mat_list = Corr_mat_list + [obs]

				Dag_mat_list = Dag_mat_list + [Dag_obs]
	
				corr_diag[:,kk] = copy.copy(np.diag(obs))

				kk = kk + 1

	print(datetime.now() - startTime,'End Simulation')

	return Corr_mat_list, Dag_mat_list, corr_diag


#def integrator(obs, Dag_obs, J, gamma, h0, h1, times, dt, measure_interval, sites, boundary_conditions, method):

	print("Only works for PBC fermions. Replaced by Integrator Matrices")

	startTime = datetime.now()

	#Initialize integration variables
	dobs_predmat = np.zeros([sites,sites],dtype = complex)
	Dag_dobs_predmat = np.zeros([sites,sites],dtype = complex)

	if method == 'heun' or 'RK4':
		dobs_mat = np.zeros([sites,sites],dtype = complex)
		Dag_dobs_mat = np.zeros([sites,sites],dtype = complex)
		
	idmat = 1.0*np.identity(sites,dtype = complex)

	#Initialize observables
	corr_diag = np.zeros([sites,times],dtype = complex)

	#Initialize output correlation matrices
	Corr_mat_list = []
	Dag_mat_list = []

	kk = 0
	for ii in range(0,times):

		if (ii%1000) == 0:

			print(ii)

		Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs))) #Refer to EOM in notes for reasoning here. Have terms c^{dag}_m c^{dag}_n and c_m c_n, whereas conjugate transpose of first term is c_n cm, expect minus sign but gives wrong answer.
		
		#Retain0.5* current obs value for Heun scheme
		orig_obs = copy.deepcopy(obs)
		Dag_orig_obs = copy.deepcopy(Dag_obs)

		#Prediction_step
		for m in range(0,sites):

			for n in range(0,sites):
				###c^dag_m c_n equation

				#J contribution
				dobs = J/2*(obs[m%sites,(n-1)%sites] + obs[m%sites,(n+1)%sites] - obs[(m+1)%sites,n%sites] - obs[(m-1)%sites,n%sites])

				#gamma contribution
				dobs += gamma*J/2*(Dag_obs[m%sites,(n+1)%sites] - Dag_obs[m%sites,(n-1)%sites] - Ann_obs[(m+1)%sites,n%sites] +  Ann_obs[(m-1)%sites,n%sites])
			
				###c^dag_m c^dag_n equation
				
				#J contribution
				Dag_dobs = J/2*(-Dag_obs[(m-1)%sites,(n)%sites] + Dag_obs[(n-1)%sites,m%sites]   - Dag_obs[(m+1)%sites,n%sites] + Dag_obs[(n+1)%sites,m%sites])

				#Gamma contribution
				Dag_dobs += -gamma*J/2*(-idmat[(m-1)%sites,n%sites] + idmat[(m+1)%sites,n%sites] +  obs[(n)%sites,(m-1)%sites] - obs[n%sites,(m+1)%sites] - obs[(m)%sites,(n-1)%sites] + obs[(m)%sites,(n+1)%sites])
				
				#h0 and h_1 contribution
				Dag_dobs += (-2*(h0 + h1[ii])*Dag_obs[m%sites,n%sites])

				###save matrix elements
				dobs_predmat[m,n]  = copy.copy(dobs)

				Dag_dobs_predmat[m,n]  = copy.copy(Dag_dobs)


		obs = obs + 1j*dobs_predmat*dt	

		Dag_obs = Dag_obs + 1j*Dag_dobs_predmat*dt		

		#Correction Step

		if method == 'heun':

			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			for m in range(0,sites):

				for n in range(0,sites):

					dobs = ( (J/2)*obs[m%sites,(n-1)%sites] + (J/2)*obs[m%sites,(n+1)%sites] + (-J/2)*obs[(m+1)%sites,n%sites] +  (-J/2)*obs[(m-1)%sites,n%sites] )
					
					dobs += gamma*J/2*(Dag_obs[m%sites,(n+1)%sites] - Dag_obs[m%sites,(n-1)%sites] - Ann_obs[(m+1)%sites,n%sites] +  Ann_obs[(m-1)%sites,n%sites])

					Dag_dobs = (-J/2*Dag_obs[(m-1)%sites,(n)%sites] + J/2*Dag_obs[(n-1)%sites,m%sites]   - J/2*Dag_obs[(m+1)%sites,n%sites] + J/2*Dag_obs[(n+1)%sites,m%sites])
					
					Dag_dobs += (-gamma*(-J/2*idmat[(m-1)%sites,n%sites] + J/2*idmat[(m+1)%sites,n%sites] +  J/2*obs[(n)%sites,(m-1)%sites] - J/2*obs[n%sites,(m+1)%sites]   - J/2*obs[(m)%sites,(n-1)%sites] + J/2*obs[(m)%sites,(n+1)%sites]))
					
					Dag_dobs += (-2*(h0 + h1[ii])*Dag_obs[m%sites,n%sites])

					dobs_mat[m,n]  = copy.copy(dobs)
					
					Dag_dobs_mat[m,n]  = copy.copy(Dag_dobs)		

			obs = orig_obs + 1j*dt*0.5*(dobs_mat + dobs_predmat)

			Dag_obs = Dag_orig_obs + 1j*dt*0.5*(Dag_dobs_mat + Dag_dobs_predmat)

		if method == 'RK4':
			######### 
			#Define k1 and input for k2

			k1 =  copy.copy(1j*dobs_predmat)

			Dag_k1 =  copy.copy(1j*Dag_dobs_predmat)

			obs = orig_obs + 0.5*k1*dt #input for k2 (like a midpoint)

			Dag_obs = Dag_orig_obs + 0.5*Dag_k1*dt #input for Dag_k2 (like a midpoint)

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			for m in range(0,sites):

				for n in range(0,sites):

					dobs = ( (J/2)*obs[m%sites,(n-1)%sites] + (J/2)*obs[m%sites,(n+1)%sites] + (-J/2)*obs[(m+1)%sites,n%sites] +  (-J/2)*obs[(m-1)%sites,n%sites] )

					dobs += gamma*J/2*(Dag_obs[m%sites,(n+1)%sites] - Dag_obs[m%sites,(n-1)%sites] - Ann_obs[(m+1)%sites,n%sites] +  Ann_obs[(m-1)%sites,n%sites])

					Dag_dobs = (-J/2*Dag_obs[(m-1)%sites,(n)%sites] + J/2*Dag_obs[(n-1)%sites,m%sites]   - J/2*Dag_obs[(m+1)%sites,n%sites] + J/2*Dag_obs[(n+1)%sites,m%sites])
					
					Dag_dobs += (-gamma*(-J/2*idmat[(m-1)%sites,n%sites] + J/2*idmat[(m+1)%sites,n%sites] +  J/2*obs[(n)%sites,(m-1)%sites] - J/2*obs[n%sites,(m+1)%sites]   - J/2*obs[(m)%sites,(n-1)%sites] + J/2*obs[(m)%sites,(n+1)%sites]))
					
					Dag_dobs += (-2*(h0 + h1[ii])*Dag_obs[m%sites,n%sites])

					dobs_mat[m,n]  = copy.copy(dobs)
					
					Dag_dobs_mat[m,n]  = copy.copy(Dag_dobs)		
			
			######### 
			#Define k2 and input for k3
							
			k2 =  copy.copy(1j*dobs_mat)

			Dag_k2 =  copy.copy(1j*Dag_dobs_mat)

			obs = orig_obs + 0.5*k2*dt #input for k3 (like a midpoint)

			Dag_obs = Dag_orig_obs + 0.5*Dag_k2*dt #input for Dag_k3 (like a midpoint)

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			for m in range(0,sites):

				for n in range(0,sites):

					dobs = ( (J/2)*obs[m%sites,(n-1)%sites] + (J/2)*obs[m%sites,(n+1)%sites] + (-J/2)*obs[(m+1)%sites,n%sites] +  (-J/2)*obs[(m-1)%sites,n%sites] )
					
					dobs += gamma*J/2*(Dag_obs[m%sites,(n+1)%sites] - Dag_obs[m%sites,(n-1)%sites] - Ann_obs[(m+1)%sites,n%sites] +  Ann_obs[(m-1)%sites,n%sites])

					Dag_dobs = (-J/2*Dag_obs[(m-1)%sites,(n)%sites] + J/2*Dag_obs[(n-1)%sites,m%sites]   - J/2*Dag_obs[(m+1)%sites,n%sites] + J/2*Dag_obs[(n+1)%sites,m%sites])
					
					Dag_dobs += (-gamma*(-J/2*idmat[(m-1)%sites,n%sites] + J/2*idmat[(m+1)%sites,n%sites] +  J/2*obs[(n)%sites,(m-1)%sites] - J/2*obs[n%sites,(m+1)%sites]   - J/2*obs[(m)%sites,(n-1)%sites] + J/2*obs[(m)%sites,(n+1)%sites]))
					
					Dag_dobs += (-2*(h0 + h1[ii])*Dag_obs[m%sites,n%sites])

					dobs_mat[m,n]  = copy.copy(dobs)
					
					Dag_dobs_mat[m,n]  = copy.copy(Dag_dobs)		

			######### 
			#Define k3 and input for k4	
					
			k3 =  copy.copy(1j*dobs_mat)

			Dag_k3 =  copy.copy(1j*Dag_dobs_mat)

			obs = orig_obs + k3*dt #input for k4

			Dag_obs = Dag_orig_obs + Dag_k3*dt #input for Dag_k4

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			for m in range(0,sites):

				for n in range(0,sites):

					dobs = ( (J/2)*obs[m%sites,(n-1)%sites] + (J/2)*obs[m%sites,(n+1)%sites] + (-J/2)*obs[(m+1)%sites,n%sites] +  (-J/2)*obs[(m-1)%sites,n%sites] )
					
					dobs += gamma*J/2*(Dag_obs[m%sites,(n+1)%sites] - Dag_obs[m%sites,(n-1)%sites] - Ann_obs[(m+1)%sites,n%sites] +  Ann_obs[(m-1)%sites,n%sites])

					Dag_dobs = (-J/2*Dag_obs[(m-1)%sites,(n)%sites] + J/2*Dag_obs[(n-1)%sites,m%sites]   - J/2*Dag_obs[(m+1)%sites,n%sites] + J/2*Dag_obs[(n+1)%sites,m%sites])
					
					Dag_dobs += (-gamma*(-J/2*idmat[(m-1)%sites,n%sites] + J/2*idmat[(m+1)%sites,n%sites] +  J/2*obs[(n)%sites,(m-1)%sites] - J/2*obs[n%sites,(m+1)%sites]   - J/2*obs[(m)%sites,(n-1)%sites] + J/2*obs[(m)%sites,(n+1)%sites]))
					
					Dag_dobs += (-2*(h0 + h1[ii])*Dag_obs[m%sites,n%sites])

					dobs_mat[m,n]  = copy.copy(dobs)
					
					Dag_dobs_mat[m,n]  = copy.copy(Dag_dobs)	

			######### 
			#Define k4
			
			k4 = 1j*dobs_mat

			Dag_k4 = 1j*Dag_dobs_mat

			#Evaluate the final RK4 expression

			obs = orig_obs + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

			Dag_obs = Dag_orig_obs + dt/6.0*(Dag_k1 + 2*Dag_k2 + 2*Dag_k3 + Dag_k4)


		if (ii+1) < int(times)+1:

			if (ii+1)%measure_interval == 0:
	
				Corr_mat_list = Corr_mat_list + [obs]

				Dag_mat_list = Dag_mat_list + [Dag_obs]

				for mmm in range(0,sites):		
					
					corr_diag[mmm,kk] = copy.copy(obs[mmm,mmm])

				kk = kk + 1

	print(datetime.now() - startTime,'End Simulation')

	return Corr_mat_list, Dag_mat_list, corr_diag


def Fisher_Calc(phase, obs, Dag_obs, J, gamma, h0_in, shift, h1, times, dt, measure_interval, sites, sub_system_range, tol, derivative_estimator, integration_type, method, initial_state, boundary_conditions):

	idmat = 1.0*np.identity(sites,dtype = complex)

	print(phase, 'h0 value = ', h0_in)

	if initial_state == 'ground_state':
		
		obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in, h1[0], boundary_conditions)

	if derivative_estimator == 'order4':
		loop = 5
		Gamma1 = [[] for i in range(0,np.size(sub_system_range))]
		Gamma2 = [[] for i in range(0,np.size(sub_system_range))]
		Gamma3 = [[] for i in range(0,np.size(sub_system_range))]
		Gamma4 = [[] for i in range(0,np.size(sub_system_range))]
		Gamma5 = [[] for i in range(0,np.size(sub_system_range))]

	elif derivative_estimator == 'order2':

		loop = 2
		Gamma1 = [[] for i in range(0,np.size(sub_system_range))]
		Gamma2 = [[] for i in range(0,np.size(sub_system_range))]
		
	for kk in range(0,loop):	

		if derivative_estimator == 'order2':

			if integration_type == 'loops':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, times, dt, measure_interval, sites, boundary_conditions, method)

			if integration_type == 'matrices':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator_matrices(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, times, dt, measure_interval, sites, boundary_conditions, method)

			if kk == 0:

				particle_numL = copy.copy(corr_diag)

		if derivative_estimator == 'order4':
				
			if integration_type == 'loops':	
			
				Corr_mat_list, Dag_mat_list, corr_diag = integrator(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, times, dt, measure_interval, sites, boundary_conditions, method)

			if integration_type == 'matrices':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator_matrices(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, times, dt, measure_interval, sites, boundary_conditions, method)
		
			if kk == 2:

				particle_numL = copy.copy(corr_diag)

		measure_times = int(times/measure_interval)

		for ss in range(0,np.size(sub_system_range)):

			sub_system = sub_system_range[ss]

			for ttt in range(0,measure_times):
				
				Gamma = np.zeros([2*sub_system,2*sub_system],dtype = complex)

				Corr_mat = copy.copy(Corr_mat_list[ttt])

				Dag_mat = copy.copy(Dag_mat_list[ttt])
			
				for ii in range(0,sub_system):

					for jj in range(0,sub_system):

						# #Note: python convention of first index being 0 means that fields all shifted by one compared to as written in notes.
						# Gamma[(2*ii),(2*jj)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						# Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii,jj] - 2*1j*np.real(Corr_mat[ii,jj] - Dag_mat[ii,jj])
			
						# Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii,jj] + 2*1j*np.real(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						# Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] - Dag_mat[ii,jj])

						Gamma[(2*ii),(2*jj)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii,jj] - 2*1j*np.real(Corr_mat[ii,jj] - Dag_mat[ii,jj])
			
						Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii,jj] + 2*1j*np.real(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] - Dag_mat[ii,jj])


				#Gamma = 0.5*copy.copy(Gamma) #I am using this since the above formula do not have 1.0/sqrt(2) in majorana definition.
				
				#Gamma = Gamma - np.diag(np.diag(Gamma))

				Gamma = 0.5*(Gamma - np.transpose(np.conj(Gamma)))

				if kk == 0:
					
					Gamma1[ss] = Gamma1[ss] + [Gamma]

				if kk == 1:

					Gamma2[ss] = Gamma2[ss] + [Gamma]

				if kk == 2:

					Gamma3[ss] = Gamma3[ss] + [Gamma]

				if kk == 3:

					Gamma4[ss] = Gamma4[ss] + [Gamma]
				
				if kk == 4:

					Gamma5[ss] = Gamma5[ss] + [Gamma]


	#Initialize matrix
	Fisher = np.zeros([np.size(sub_system_range),measure_times])

	print('Simulations finished, reduced density matrices extracted. Now calculate Fisher information.')

	startTime = datetime.now()

	for zzz in range(0,np.size(sub_system_range)):

		avoid_index = 0

		sub_system = sub_system_range[zzz]

		print('Analyse sub_system L = ' + str(sub_system))
		
		for ttt in range(0,measure_times):
			
			if derivative_estimator == 'order2':
				
				Gamma = copy.copy(Gamma1[zzz][ttt])

			elif derivative_estimator == 'order4':

				Gamma = copy.copy(Gamma3[zzz][ttt])
			
			#calculate eigenvectors of Gamma

			w , v = lin.eigh(Gamma) #eigenvalues w[aa] and eigenvectors v[:,aa]

			if -np.max(-w) < -1.0000000001:
			
				print('Error test: minimum eigenvalue',-np.max(-w)) #if negative this indicates error
			
			if np.max(w) > 1.00000000001:

				print('max eigenvalue',np.max(w)) #if > 1 this indicates error
			
			Hermitian_check = np.round(Gamma,3) - np.conj(np.transpose(np.round(Gamma,3)))

			Hermitian_check[np.abs(Hermitian_check)<10**(-15)] =0
			
			if np.allclose(Hermitian_check,np.zeros([2*sub_system,2*sub_system],dtype=complex)) == False:
				
				print('Hermitian Fail')

			if derivative_estimator == 'order2':
				
				Gamma1_t = copy.copy(Gamma1[zzz][ttt])

				Gamma2_t = copy.copy(Gamma2[zzz][ttt])

				GammaD = (Gamma2_t - Gamma1_t)/shift 

			elif derivative_estimator == 'order4':
				
				Gamma1_t = copy.copy(Gamma1[zzz][ttt])

				Gamma2_t = copy.copy(Gamma2[zzz][ttt])

				Gamma4_t = copy.copy(Gamma4[zzz][ttt])
				
				Gamma5_t = copy.copy(Gamma5[zzz][ttt])

				GammaD = (Gamma1_t  - 8*Gamma2_t + 8*Gamma4_t - Gamma5_t)/(12*shift)

			Fisher_time = 0

			for rr in range(0,2*sub_system):
				
				for ss in range(0,2*sub_system):

					if np.abs(1 - w[rr]*w[ss]) > tol:

						Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,ss]) * (np.conj(v[:,ss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[ss]) #(np.conj(v[:,rr]) @ GammaD @ v[:,ss]) * (np.conj(v[:,ss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[ss])

					else:

						avoid_index = avoid_index + 1

			Fisher[zzz,ttt] = Fisher_time  

		print('Subsystem analaysis complete. ' + 'Number of avoided divergences (average per time) = ' + str(int(avoid_index/measure_times)))

	print(datetime.now() - startTime,'End Fisher Calculation')

	return [Fisher, particle_numL ] 


def Fisher_Groundstate(J, gamma, h0_in, h1, sites, sub_system_range, sub_system_edge, tol, shift, derivative_estimator, boundary_conditions):

	idmat = 1.0*np.identity(sites,dtype = complex)

	if derivative_estimator == 'order2':

		loop = 2
		Gamma1 = []
		Gamma2 = []
	
	elif derivative_estimator == 'order4':

		loop = 5
		Gamma1 = []
		Gamma2 = []
		Gamma3 = []
		Gamma4 = []
		Gamma5 = []

	for kk in range(0,loop):	

		startTime = datetime.now()

		if derivative_estimator == 'order2':

			Corr_mat, Dag_mat = correlation_groundstate(sites, gamma, J, h0_in + shift*kk, h1, boundary_conditions)

			if kk == 0:

				particle_numL = np.mean(np.diag(Corr_mat))

		elif derivative_estimator == 'order4':
			
			Corr_mat, Dag_mat = correlation_groundstate(sites, gamma, J, h0_in -2*shift + shift*kk, h1, boundary_conditions)

			if kk == 2:

				particle_numL = np.mean(np.diag(Corr_mat))

		print(datetime.now() - startTime,'Time to calculate the initial correlation matrices')

		startTime = datetime.now()

		for ss in range(0,np.size(sub_system_range)):

			sub_system = sub_system_range[ss]

			Gamma = np.zeros([2*sub_system,2*sub_system],dtype = complex)
		
			for ii in range(0,sub_system):

				for jj in range(0,sub_system):

					#Note: python convention of first index being 0 means that fields all shifted by one compared to as written in notes.
					Gamma[(2*ii),(2*jj)] = idmat[ii + sub_system_edge,jj + sub_system_edge] + 2*1j*np.imag(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] + Dag_mat[ii + sub_system_edge, jj + sub_system_edge])
		
					Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii + sub_system_edge,jj + sub_system_edge] - 2*1j*np.real(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] + Dag_mat[ii + sub_system_edge, jj + sub_system_edge])

					Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii + sub_system_edge,jj + sub_system_edge] + 2*1j*np.real(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] + Dag_mat[ii + sub_system_edge, jj + sub_system_edge])

					Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii + sub_system_edge,jj + sub_system_edge] + 2*1j*np.imag(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] - Dag_mat[ii + sub_system_edge, jj + sub_system_edge])


			Gamma = 0.5*(Gamma - np.transpose(Gamma))
			
			if kk == 0:

				Gamma1 = Gamma1 + [Gamma]

			if kk == 1:

				Gamma2 = Gamma2 + [Gamma]

			if kk == 2:

				Gamma3 = Gamma3 + [Gamma]

			if kk == 3:

				Gamma4 = Gamma4 + [Gamma]
			
			if kk == 4:

				Gamma5 = Gamma5 + [Gamma]

		print(datetime.now() - startTime,'Time to evaluate Gamma matrix for single h0')

	#Initialize matrix

	startTime = datetime.now()

	Fisher = []

	avoid_index = np.zeros(np.size(sub_system_range))
	
	for ss in range(0,np.size(sub_system_range)):

		avoid_var = 0

		sub_system = sub_system_range[ss]
			
		if derivative_estimator == 'order2':
			
			Gamma = copy.copy(Gamma1[ss])

		elif derivative_estimator == 'order4':

			Gamma = copy.copy(Gamma3[ss])
		
		#calculate eigenvectors of Gamma

		w,v = lin.eigh(Gamma) #eigenvalues w[aa] and eigenvectors v[:,aa]

		if -np.max(-w) < -1.000000001:
		
			print('Error test: minimum eigenvalue',-np.max(-w)) #if negative this indicates error
		
		if np.max(w) > 1.000000001:

			print('max eigenvalue',np.max(w)) #if > 1 this indicates error
		
		Hermitian_check = np.round(Gamma,3) - np.conj(np.transpose(np.round(Gamma,3)))

		Hermitian_check[np.abs(Hermitian_check)<10**(-15)] =0

		
		if np.allclose(Hermitian_check,np.zeros([2*sub_system,2*sub_system],dtype=complex)) == False:
			
			print('Hermitian Fail')

		if derivative_estimator == 'order2':
			
			Gamma1_t = copy.copy(Gamma1[ss])

			Gamma2_t = copy.copy(Gamma2[ss])

			GammaD = (Gamma2_t - Gamma1_t)/shift 

		elif derivative_estimator == 'order4':
			
			Gamma1_t = copy.copy(Gamma1[ss])

			Gamma2_t = copy.copy(Gamma2[ss])

			Gamma4_t = copy.copy(Gamma4[ss])
			
			Gamma5_t = copy.copy(Gamma5[ss])

			GammaD = (Gamma1_t  - 8*Gamma2_t + 8*Gamma4_t - Gamma5_t)/(12*shift)

		Fisher_time = 0

		for rr in range(0,2*sub_system):
			
			for sss in range(0,2*sub_system):

				if np.abs(1 - w[rr]*w[sss]) > tol:

					Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,sss]) * (np.conj(v[:,sss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[sss])

				else:

					avoid_var = avoid_var + 1

		Fisher = Fisher +  [Fisher_time] 

		avoid_index[ss] = avoid_var
		
	print('Number of avoided divergences',avoid_index)

	#print('subsystem',sub_system_range)

	print(datetime.now() - startTime,'End Fisher Calculation')

	return [Fisher, particle_numL]