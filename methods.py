import numpy as np
from datetime import datetime
import copy
import scipy.linalg as lin
from scipy.interpolate import interp1d
from scipy.integrate import solve_ivp
import numpy.random as random


def BdG_vacuum(sites):
	
	vac_state = []	

	for cc in range(0,sites):

		vac_state = vac_state + [np.asarray([0,1])] #basis state is down (unoccupied state), initial up-state used in method

	return vac_state

def random_state(sites):
	
	rand_state = []	

	for cc in range(0,sites):

		if random.rand() > 0.5:

			rand_state = rand_state + [np.asarray([1,0])] #basis state is up (occupied state), initial down-state used in method
		
		else:
			rand_state = rand_state + [np.asarray([0,1])] #basis state is down (unoccupied state), initial up-state used in method

	return rand_state

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

	elif boundary_conditions == 'PBC':

		fourier_def = 0

	if fourier_def == 0:
				
		kval = - np.pi + 2*np.arange(0,sites)*np.pi/sites
				
		thetaK = np.arctan2( J * gamma * np.sin(kval), J * np.cos(kval) + h0_in + h1_in)


		for m in range(0,sites):
			
			for n in range(0,sites):

				Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK))

				obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK/2.0))**2)
					
	if fourier_def == 1:
				
		#kval = - np.pi + (2*np.arange(0,sites)+1)*np.pi/sites 
		kval = - np.pi + (2*np.arange(0,sites)+1)*np.pi/sites 

		thetaK = np.arctan2( J * gamma * np.sin(kval), J * np.cos(kval) + h0_in + h1_in)

		for m in range(0,sites):
			
			for n in range(0,sites):

				#Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK))

				#obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK/2.0))**2)
				Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (1j) * 0.5 * np.sin(thetaK))

				obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * (np.cos(thetaK/2.0))**2)
		

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


def initial_BdG(J, gamma, h0_in, h1_in, sites, boundary_conditions):

	#Calculate the initial condition for the ground-state case

	if boundary_conditions == 'PBC':

		kvec = -np.pi + 2*np.arange(0,sites)*np.pi/sites

	if boundary_conditions == 'ABC':

		kvec = -np.pi + (2*np.arange(0,sites)+1)*np.pi/sites #(2*(kk) + 1)*np.pi/sites

	state = []

	for kk in range(0,sites):

		thetaK = np.arctan2( J * gamma * np.sin(kvec[kk]), J * np.cos(kvec[kk]) + h0_in + h1_in[0])

		u = np.cos(thetaK / 2.0)

		v = 1j * np.sin(thetaK / 2.0)

		state = state + [np.asarray([u , v])]

	initial_state = copy.copy(state)

	return initial_state



def integrator_BdG(J, gamma, h0_in, h1_in, h1_in_midpoint, oneperiod_steps, dt, sites, boundary_conditions, initial_state, method):

	if boundary_conditions == 'PBC':

		kvec = -np.pi + 2*np.arange(0,sites)*np.pi/sites

	if boundary_conditions == 'ABC':

		kvec = -np.pi + (2*np.arange(0,sites)+1)*np.pi/sites 

	state = copy.copy(initial_state)

	for tt in range(0,oneperiod_steps):

		for kk in range(0,sites):
		
			Hmat = np.asarray([[-(J*np.cos(kvec[kk]) + h0_in + h1_in[tt]), 1j * J * gamma * np.sin(kvec[kk])] , [ -1j * J * gamma * np.sin(kvec[kk]),  (J * np.cos(kvec[kk]) + h0_in + h1_in[tt])]])

			Hmat_midpoint = np.asarray([[-(J*np.cos(kvec[kk]) + h0_in + h1_in_midpoint[tt]), 1j * J * gamma * np.sin(kvec[kk])] , [ -1j * J * gamma * np.sin(kvec[kk]),  (J * np.cos(kvec[kk]) + h0_in + h1_in_midpoint[tt])]])

			Hmat_endpoint = np.asarray([[-(J*np.cos(kvec[kk]) + h0_in + h1_in[(tt+1)%oneperiod_steps]), 1j * J * gamma * np.sin(kvec[kk])] , [ -1j * J * gamma * np.sin(kvec[kk]),  (J * np.cos(kvec[kk]) + h0_in + h1_in[(tt+1)%oneperiod_steps])]])

			if method == 'heun':

				temp_dstate =  Hmat @ state[kk]

				temp_state = state[kk] + -1j * dt * temp_dstate

				dstate =  Hmat_endpoint @ temp_state
			
				state[kk] = state[kk] + -1j * dt * 0.5 * (temp_dstate + dstate)

			if method == 'RK4':

				k1 =  Hmat @ state[kk]

				step_state = state[kk] + -1j * dt * k1 * 0.5

				k2 = Hmat_midpoint @ step_state

				step_state = state[kk] + -1j * dt * k2 * 0.5

				k3 = Hmat_midpoint @ step_state

				step_state = state[kk] + -1j * dt * k3

				k4 = Hmat_endpoint @ step_state

				state[kk] = state[kk] + -1j *  dt * 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)

	return state


def floquet_eigsystem(state):
	
	sites = len(state)

	valmat = []

	vecmat = []

	for kk in range(0,sites):
		
		Umatrix = [ [ state[kk][0] , - np.conj(state[kk][1]) ] , [ state[kk][1] , np.conj(state[kk][0]) ]]

		#print('Iden' , np.dot(Umatrix,np.conjugate(np.transpose(Umatrix))))

		val, vec = lin.eig( Umatrix )

		valmat = valmat + [val]

		vecmat = vecmat + [vec]

	return valmat , vecmat
		
def floquet_evolution(J, gamma, h0_in, h1_in, h1_in_midpoint, final_time, oneperiod_steps, dt, sites, boundary_conditions, method, eval, evec, initial_state):

	ov0 = np.zeros(sites, dtype = complex)

	ov1 = np.zeros(sites, dtype = complex)

	# Calculate overlaps of initial state

	for kk in range(0,sites):

		ov0[kk] = np.dot(np.conj(evec[kk][:,0]) , initial_state[kk])

		ov1[kk] = np.dot(np.conj(evec[kk][:,1]) , initial_state[kk])

	# Calculate the final state in [u_k, v_k] space

	final_state = np.zeros((sites,2), dtype = complex)

	for kk in range(0,sites):

		final_state[kk,:] = (eval[kk][0])**(final_time) * ov0[kk]  * evec[kk][:,0]  + (eval[kk][1])**(final_time) * ov1[kk] * evec[kk][:,1]

	# Return correlation matrices at this final time

	obs = np.zeros([sites,sites],dtype = complex) 

	Dag_obs = np.zeros([sites,sites],dtype = complex) 	
	
	if boundary_conditions == 'PBC':

		kval = - np.pi + 2*np.arange(0,sites)*np.pi/sites

	elif boundary_conditions == 'ABC':
	
		kval = - np.pi + (2*np.arange(0,sites)+1)*np.pi/sites

	obs_kspace = (final_state[:,0])*np.conj(final_state[:,0])

	Dag_obs_kspace = np.conj(final_state[:,0]) * final_state[:,1]

	for m in range(0,sites):
	
		for n in range(0,sites):

			Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * Dag_obs_kspace)

			obs[m,n] =  np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * obs_kspace)

	return  obs , Dag_obs, eval

		
def floquet_evolution_eff(final_time, sites, boundary_conditions, eval, evec, initial_state):

	ov0 = np.zeros(sites, dtype = complex)

	ov1 = np.zeros(sites, dtype = complex)

	# Calculate overlaps of initial state

	for kk in range(0,sites):

		ov0[kk] = np.dot(np.conj(evec[kk][:,0]) , initial_state[kk])

		ov1[kk] = np.dot(np.conj(evec[kk][:,1]) , initial_state[kk])

	# Calculate the final state in [u_k, v_k] space

	final_state = np.zeros((sites,2), dtype = complex)

	for kk in range(0,sites):

		final_state[kk,:] = (eval[kk][0])**(final_time) * ov0[kk]  * evec[kk][:,0]  + (eval[kk][1])**(final_time) * ov1[kk] * evec[kk][:,1]

	# Return correlation matrices at this final time

	obs = np.zeros([sites,sites],dtype = complex) 

	Dag_obs = np.zeros([sites,sites],dtype = complex) 	
	
	if boundary_conditions == 'PBC':

		kval = - np.pi + 2*np.arange(0,sites)*np.pi/sites

	elif boundary_conditions == 'ABC':
	
		kval = - np.pi + (2*np.arange(0,sites)+1)*np.pi/sites

	obs_kspace = (final_state[:,0])*np.conj(final_state[:,0])

	Dag_obs_kspace = np.conj(final_state[:,0]) * final_state[:,1]

	for m in range(0,sites):
	
		for n in range(0,sites):

			Dag_obs[m,n] = np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * Dag_obs_kspace)

			obs[m,n] =  np.sum(1.0/sites * np.exp(-1j*(n-m)*kval) * obs_kspace)

	return  obs , Dag_obs, eval

# convert states from position to momentum space
# def position_to_momentum(sites, boundary_conditions, state):

# 	kval = None
# 	if boundary_conditions == 'PBC':
# 		kval = - np.pi + 2*np.arange(0,sites)*np.pi/sites
# 	elif boundary_conditions == 'ABC':
# 		kval = - np.pi + (2*np.arange(0,sites)+1)*np.pi/sites

# 	output_state = np.zeros((sites,2), dtype = complex)

# 	for kk in range(0,sites):
# 		# convert each state from position to momentum space
			
# 		for jj in range(0,sites):

# 			output_state[kk] += state[jj][1] * np.exp(-1j * kval[jj] * jj) / np.sqrt(sites)

# 	return output_state


def integrator_matrices(obs, Dag_obs, J, gamma, h0_in, h1, h1_midpoint, times, dt, measure_interval, sites, boundary_conditions, method):

	startTime = datetime.now()
	 
	M = Mmatrix(sites, boundary_conditions)

	N = Nmatrix(sites, boundary_conditions)
	
	#Initialize integration variables
	dobs_predmat = np.zeros([sites,sites],dtype = complex)

	Dag_dobs_predmat = np.zeros([sites,sites],dtype = complex)

	if method in ['heun','RK4']:

		dobs_mat = np.zeros([sites,sites],dtype = complex)

		Dag_dobs_mat = np.zeros([sites,sites],dtype = complex)
		
	#Initialize observables
	corr_diag = np.zeros([sites,int(times/measure_interval)],dtype = complex)

	#Initialize output correlation matrices
	Corr_mat_list = []

	Dag_mat_list = []

	kk = 0 # variable updated when measurements made (for storing outputs)

	dobs = np.zeros([sites,sites])

	Dag_dobs = np.zeros([sites,sites])

	Corr_mat_list = Corr_mat_list + [obs]

	Dag_mat_list = Dag_mat_list + [Dag_obs]
	
	for ii in range(0,times):

		# if (ii%1000) == 0:

		# 	print(ii)

		Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs))) #Refer to EOM in notes for reasoning here. Have terms c^{dag}_m c^{dag}_n and c_m c_n, whereas conjugate transpose of first term is c_n cm, expect minus sign but gives wrong answer.
		
		#retain for Heun and RK4 schemes
		orig_obs = copy.copy(obs)
		
		Dag_orig_obs = copy.copy(Dag_obs)

		dobs = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

		Dag_dobs = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0_in + h1[ii])*Dag_obs
		
		#save matrix elements for Heun and RK4 schemes
		dobs_predmat = copy.copy(dobs)

		Dag_dobs_predmat  = copy.copy(Dag_dobs)

		obs = obs + 1j*dobs_predmat*dt	

		Dag_obs = Dag_obs + 1j*Dag_dobs_predmat*dt		

		
		if method == 'heun':
			#Correction Step

			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			dobs = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0_in + h1[ii+1])*Dag_obs

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

			Dag_dobs_mat = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0_in + h1_midpoint[ii])*Dag_obs
			
			######### 
			#Define k2 and input for k3
							
			k2 =  copy.copy(1j*dobs_mat)

			Dag_k2 =  copy.copy(1j*Dag_dobs_mat)

			obs = orig_obs + 0.5*k2*dt #input for k3 (like a midpoint)

			Dag_obs = Dag_orig_obs + 0.5*Dag_k2*dt #input for Dag_k3 (like a midpoint)

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))

			dobs_mat = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs_mat = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0_in + h1_midpoint[ii])*Dag_obs

			######### 
			#Define k3 and input for k4	
					
			k3 =  copy.copy(1j*dobs_mat)

			Dag_k3 =  copy.copy(1j*Dag_dobs_mat)

			obs = orig_obs + k3*dt #input for k4

			Dag_obs = Dag_orig_obs + Dag_k3*dt #input for Dag_k4

			######### 
			Ann_obs = copy.copy(np.conj(np.transpose(Dag_obs)))
			
			dobs_mat = -J/2*np.dot(M,obs) + J/2*np.dot(obs,np.transpose(M))  - J/2*gamma*np.dot(N,Ann_obs) +  J/2*gamma*np.dot(Dag_obs,np.transpose(N))

			Dag_dobs_mat = -J/2*np.dot(M,Dag_obs) + J/2*np.conj(np.dot(Ann_obs,np.transpose(M))) - J/2*gamma*np.dot(obs,np.transpose(N)) + J/2*np.conj(gamma*np.dot(N,obs)) - J/2*gamma*N - 2*(h0_in + h1[ii+1])*Dag_obs	

			######### 
			#Define k4
			
			k4 = copy.copy(1j*dobs_mat)

			Dag_k4 = copy.copy(1j*Dag_dobs_mat)

			#Evaluate the final RK4 expression

			obs = orig_obs + dt/6.0*(k1 + 2*k2 + 2*k3 + k4)

			Dag_obs = Dag_orig_obs + dt/6.0*(Dag_k1 + 2*Dag_k2 + 2*Dag_k3 + Dag_k4)


		if (ii+1) < int(times)+1:
	
				Corr_mat_list = Corr_mat_list + [obs]

				Dag_mat_list = Dag_mat_list + [Dag_obs]
	
				corr_diag[:,kk] = copy.copy(np.diag(obs))

				kk = kk + 1

	print(datetime.now() - startTime,'End Simulation')

	return Corr_mat_list, Dag_mat_list, corr_diag

def integrator_matrices_eff(obs, Dag_obs, J, gamma, h0_in, h1, times, dt, sites, boundary_conditions, method, atol, rtol):

	startTime = datetime.now()
	 
	M = Mmatrix(sites, boundary_conditions)

	N = Nmatrix(sites, boundary_conditions)

	def system_rhs(t, y, J, gamma, h0_in, h1_func, M, N, sites):
		obs = y[:sites*sites].reshape((sites, sites))
		Dag_obs = y[sites*sites:].reshape((sites, sites))

		Ann_obs = np.conj(Dag_obs.T)
		h = h0_in + h1_func(t)  # time-dependent field

		dobs = (
			-J/2 * M @ obs + J/2 * obs @ M.T
			- J/2 * gamma * N @ Ann_obs + J/2 * gamma * Dag_obs @ N.T
		)

		Dag_dobs = (
			-J/2 * M @ Dag_obs + J/2 * np.conj(Ann_obs @ M.T)
			- J/2 * gamma * obs @ N.T + J/2 * np.conj(gamma * N @ obs)
			- 2 * h * Dag_obs - J/2 * gamma * N 
		)
		

		# Combine into a single flattened vector
		dydt = 1j * np.concatenate([dobs.ravel(), Dag_dobs.ravel()])
		return dydt

	def make_h1_func(h1_array, dt):
		time_grid = np.arange(len(h1_array)) * dt
		return interp1d(time_grid, h1_array, kind='linear', fill_value='extrapolate')
	
	# Initialization
	obs0 = copy.copy(obs)          # shape: (L, L)
	Dag_obs0 = copy.copy(Dag_obs)  # shape: (L, L)
	y0 = np.concatenate([obs0.ravel(), Dag_obs0.ravel()])

	# Create interpolated h1 function
	h1_func = make_h1_func(h1, dt)

	# Solve
	sol = solve_ivp(
		fun=lambda t, y: system_rhs(t, y, J, gamma, h0_in, h1_func, M, N, sites),
		t_span=(0, dt*times),
		y0=y0,
		method=method,              #'RK45, DOP853', 'BDF'
		t_eval=np.arange(0, dt*(times), dt),
		vectorized=False,           # Set to True if your RHS supports vector inputs
		atol=atol,
		rtol=rtol
	)

	num_steps = len(sol.t)
	Corr_mat_list = []
	Dag_mat_list = []

	corr_diag = np.zeros([sites,int(num_steps)],dtype = complex)

	for k in range(num_steps):
		y = sol.y[:, k]
		obs = y[:sites*sites].reshape((sites, sites))
		Dag_obs = y[sites*sites:].reshape((sites, sites))

		Corr_mat_list.append(obs)
		Dag_mat_list.append(Dag_obs)
		corr_diag[:,k] = copy.copy(np.diag(obs))

	print(datetime.now() - startTime,'End Simulation')

	return Corr_mat_list, Dag_mat_list, corr_diag



def fermion_to_Majorana(Corr_mat, Dag_mat, sub_system):

	idmat = np.identity(sub_system, dtype = complex)

	Gamma = np.zeros([2*sub_system, 2*sub_system],dtype = complex)

	for ii in range(0,sub_system):

		for jj in range(0,sub_system):

			Gamma[(2*ii),(2*jj)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] + Dag_mat[ii,jj])

			Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii,jj] - 2*1j*np.real(Corr_mat[ii,jj] - Dag_mat[ii,jj])

			Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii,jj] + 2*1j*np.real(Corr_mat[ii,jj] + Dag_mat[ii,jj])

			Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] - Dag_mat[ii,jj])

	Gamma = 0.5*(Gamma - np.transpose(Gamma))

	return Gamma


def Fisher_from_Gamma(Gamma_list,  shift, tol, sub_system):

	#Fisher information from Gamma matrix using 2nd order derivative approximation

	avoid_index = 0
	
	Gamma1 = Gamma_list[0]

	Gamma2 = Gamma_list[1]

	Gamma3 = Gamma_list[2]

	Gamma4 = Gamma_list[3]

	Gamma5 = Gamma_list[4]

	#calculate eigenvectors of Gamma

	w , v = lin.eigh(Gamma3) #eigenvalues w[aa] and eigenvectors v[:,aa]

	if -np.max(-w) < -1.0000000001:
	
		print('Error test: minimum eigenvalue',-np.max(-w)) #if negative this indicates error
	
	if np.max(w) > 1.00000000001:

		print('max eigenvalue',np.max(w)) #if > 1 this indicates error
	
	Hermitian_check = np.round(Gamma3,3) - np.conj(np.transpose(np.round(Gamma3,3)))

	Hermitian_check[np.abs(Hermitian_check)<10**(-15)] =0
	
	if np.allclose(Hermitian_check,np.zeros([2*sub_system,2*sub_system],dtype=complex)) == False:
		
		print('Hermitian Fail')

	GammaD = (Gamma1  - 8*Gamma2 + 8*Gamma4 - Gamma5) / (12*shift)

	Fisher_time = 0

	for rr in range(0,2*sub_system):
		
		for ss in range(0,2*sub_system):

			if np.abs(1 - w[rr]*w[ss]) > tol:
				
				Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,ss]) * (np.conj(v[:,ss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[ss]) 

			else:

				avoid_index = avoid_index + 1

	print('avoid singularity index', avoid_index)

	Fisher = Fisher_time  
	
	return Fisher


def Fisher_from_Gamma_2ndOrder(Gamma, Gamma_pert, shift, tol, sub_system):

	#Fisher information from Gamma matrix using 2nd order derivative approximation

	avoid_index = 0
	
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

	GammaD = (Gamma_pert  - Gamma) / shift

	Fisher_time = 0

	for rr in range(0,2*sub_system):
		
		for ss in range(0,2*sub_system):

			if np.abs(1 - w[rr]*w[ss]) > tol:
				
				Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,ss]) * (np.conj(v[:,ss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[ss]) 

			else:

				avoid_index = avoid_index + 1

	print('avoid singularity index', avoid_index)

	Fisher = Fisher_time  
	
	return Fisher


def Fisher_Calc(phase, obs, Dag_obs, J, gamma, h0_in, shift, h1, h1_midpoint, times, dt, measure_interval, sites, sub_system_range, tol, derivative_estimator, integration_type, method, initial_state, boundary_conditions):

	idmat = 1.0*np.identity(sites,dtype = complex)

	print(phase, 'h0 value = ', h0_in)

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

			if initial_state == 'ground_state':
				
				obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in + shift*kk, h1[0], boundary_conditions)

			if integration_type == 'matrices':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator_matrices(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, h1_midpoint, times, dt, measure_interval, sites, boundary_conditions, method)

			if kk == 0:

				particle_numL = copy.copy(corr_diag)
				

		if derivative_estimator == 'order4':

			if initial_state == 'ground_state':
				
				obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in + shift*kk, h1[0], boundary_conditions)

			if integration_type == 'matrices':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator_matrices(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, h1_midpoint, times, dt, measure_interval, sites, boundary_conditions, method)
		
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

						Gamma[(2*ii),(2*jj)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii,jj] - 2*1j*np.real(Corr_mat[ii,jj] - Dag_mat[ii,jj])
			
						Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii,jj] + 2*1j*np.real(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] - Dag_mat[ii,jj])


				Gamma = 0.5*(Gamma - np.transpose(Gamma))

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

				GammaD = (Gamma2_t - Gamma1_t) / shift 
			
			elif derivative_estimator == 'order4':
				
				Gamma1_t = copy.copy(Gamma1[zzz][ttt])

				Gamma2_t = copy.copy(Gamma2[zzz][ttt])

				Gamma4_t = copy.copy(Gamma4[zzz][ttt])
				
				Gamma5_t = copy.copy(Gamma5[zzz][ttt])

				GammaD = (Gamma1_t  - 8*Gamma2_t + 8*Gamma4_t - Gamma5_t) / (12*shift)

			Fisher_time = 0

			for rr in range(0,2*sub_system):
				
				for ss in range(0,2*sub_system):

					if np.abs(1 - w[rr]*w[ss]) > tol:
						
						Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,ss]) * (np.conj(v[:,ss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[ss]) 

					else:

						avoid_index = avoid_index + 1

			Fisher[zzz,ttt] = Fisher_time  

		print('Subsystem analaysis complete. ' + 'Number of avoided divergences (average per time) = ' + str(int(avoid_index/measure_times)))

	print(datetime.now() - startTime,'End Fisher Calculation')
	
	return [Fisher, particle_numL ] 

def Fisher_Calc_eff(phase, obs, Dag_obs, J, gamma, h0_in, shift, h1, times, dt, sites, sub_system_range, tol, derivative_estimator, method, initial_state, boundary_conditions, atol, rtol):

	integration_type = 'matrices'

	idmat = np.identity(sites,dtype = complex)

	measure_times = int(times)
	num_subsystems = len(sub_system_range)

	print(phase, 'h0 value = ', h0_in)

	if derivative_estimator == 'order4':
		
		loop = 5
		Gamma1 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]
		Gamma2 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]
		Gamma3 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]
		Gamma4 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]
		Gamma5 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]

	elif derivative_estimator == 'order2':

		loop = 2
		Gamma1 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]
		Gamma2 = [[None for _ in range(measure_times)] for _ in range(num_subsystems)]
		
	for kk in range(0,loop):	

		if derivative_estimator == 'order2':

			if initial_state == 'ground_state':
				
				obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in + shift*kk, h1[0], boundary_conditions)

			if integration_type == 'matrices':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator_matrices_eff(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1, times, dt, sites, boundary_conditions, method, atol, rtol)

			if kk == 0:

				particle_numL = copy.copy(corr_diag)

				pair_creation = copy.copy(Dag_mat_list)	
								
				corr_mat = copy.copy(Corr_mat_list)	
			

		if derivative_estimator == 'order4':

			if initial_state == 'ground_state':
				
				obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in + shift*kk, h1[0], boundary_conditions)

			if integration_type == 'matrices':

				Corr_mat_list, Dag_mat_list, corr_diag = integrator_matrices_eff(obs, Dag_obs, J, gamma, h0_in + shift*kk, h1,  times, dt, sites, boundary_conditions, method, atol, rtol)
		
			if kk == 2:

				particle_numL = copy.copy(corr_diag)

				pair_creation = copy.copy(Dag_mat_list)	

				corr_mat = copy.copy(Corr_mat_list)	

		for ss in range(0,num_subsystems):

			sub_system = sub_system_range[ss]

			for ttt in range(0,measure_times):
				
				Gamma = np.zeros([2*sub_system,2*sub_system],dtype = complex)

				Corr_mat = copy.copy(Corr_mat_list[ttt])

				Dag_mat = copy.copy(Dag_mat_list[ttt])
			
				for ii in range(0,sub_system):

					for jj in range(0,sub_system):

						# #Note: python convention of first index being 0 means that fields all shifted by one compared to as written in notes.

						Gamma[(2*ii),(2*jj)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii,jj] - 2*1j*np.real(Corr_mat[ii,jj] - Dag_mat[ii,jj])
			
						Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii,jj] + 2*1j*np.real(Corr_mat[ii,jj] + Dag_mat[ii,jj])

						Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii,jj] + 2*1j*np.imag(Corr_mat[ii,jj] - Dag_mat[ii,jj])


				Gamma = 0.5*(Gamma - np.transpose(Gamma))

				if kk == 0:
					
					Gamma1[ss][ttt] = Gamma

				if kk == 1:

					Gamma2[ss][ttt] = Gamma

				if kk == 2:

					Gamma3[ss][ttt] = Gamma

				if kk == 3:

					Gamma4[ss][ttt] = Gamma
				
				if kk == 4:

					Gamma5[ss][ttt] = Gamma


	
	#Initialize matrix
	Fisher = np.zeros([num_subsystems,measure_times])

	print('Simulations finished, reduced density matrices extracted. Now calculate Fisher information.')

	startTime = datetime.now()

	for zzz in range(0,num_subsystems):

		avoid_index = 0

		sub_system = sub_system_range[zzz]

		print('Analyse sub_system L = ' + str(sub_system))
		
		for ttt in range(0,measure_times):
			
			if derivative_estimator == 'order2':
				
				Gamma = Gamma1[zzz][ttt]

			elif derivative_estimator == 'order4':

				Gamma = Gamma3[zzz][ttt]
			
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
				
				Gamma1_t = Gamma1[zzz][ttt]

				Gamma2_t = Gamma2[zzz][ttt]

				GammaD = (Gamma2_t - Gamma1_t) / shift 
			
			elif derivative_estimator == 'order4':
				
				Gamma1_t = Gamma1[zzz][ttt]

				Gamma2_t = Gamma2[zzz][ttt]

				Gamma4_t = Gamma4[zzz][ttt]
				
				Gamma5_t = Gamma5[zzz][ttt]

				GammaD = (Gamma1_t  - 8*Gamma2_t + 8*Gamma4_t - Gamma5_t) / (12*shift)

			Fisher_time = 0

			for rr in range(0,2*sub_system):
				
				for ss in range(0,2*sub_system):

					if np.abs(1 - w[rr]*w[ss]) > tol:
						
						Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,ss]) * (np.conj(v[:,ss]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[ss]) 

					else:

						avoid_index = avoid_index + 1

			Fisher[zzz,ttt] = Fisher_time  

		print('Subsystem analaysis complete. ' + 'Number of avoided divergences (average per time) = ' + str(int(avoid_index/measure_times)))

	print(datetime.now() - startTime,'End Fisher Calculation')
	
	return [Fisher, particle_numL, corr_mat, pair_creation ] 


def Fisher_Groundstate(J, gamma, h0_in, h1, sites, sub_system_range, sub_system_edge, tol, shift, derivative_estimator, boundary_conditions):

	print('h0',h0_in)

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

			# np.save("Dag_pbc.npy",Dag_mat)
			# np.save("obs_pbc.npy",Corr_mat)
			# print("SAVED")

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
		
					Gamma[(2*ii),(2*jj+1)%(2*sub_system)] = 1j*idmat[ii + sub_system_edge,jj + sub_system_edge] - 2*1j*np.real(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] - Dag_mat[ii + sub_system_edge, jj + sub_system_edge])

					Gamma[(2*ii+1)%(2*sub_system),(2*jj)] = -1j*idmat[ii + sub_system_edge,jj + sub_system_edge] + 2*1j*np.real(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] + Dag_mat[ii + sub_system_edge, jj + sub_system_edge])

					Gamma[(2*ii+1)%(2*sub_system),(2*jj+1)%(2*sub_system)] = idmat[ii + sub_system_edge,jj + sub_system_edge] + 2*1j*np.imag(Corr_mat[ii + sub_system_edge, jj + sub_system_edge] - Dag_mat[ii + sub_system_edge, jj + sub_system_edge])

			#correlation matrix for Majoranas, can use to extract density, correlations
			corr_Gamma = copy.copy(Gamma)

			#Need covariance matrix for Fisher information
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

		#Run additional conistency check 
  
		if kk == 0:

			#particle number on first site, from Majorana matrix as consistency check.
			particle_numMaj = -1j*0.5*corr_Gamma[1,0] + 0.5

			if particle_numMaj - Corr_mat[0,0] > 10**(-8):

				print('WARNING: Majorana and regular Fermion expectation values do not agree')



	startTime = datetime.now()

	Fisher = []

	avoid_index = np.zeros(np.size(sub_system_range))

	avoid_percentage = np.zeros(np.size(sub_system_range))
	
	for ss in range(0,np.size(sub_system_range)):

		sub_system = sub_system_range[ss]
			
		if derivative_estimator == 'order2':
			
			Gamma = copy.copy(Gamma1[ss])

		elif derivative_estimator == 'order4':

			Gamma = copy.copy(Gamma3[ss])
		
		#calculate eigenvectors of Gamma

		w,v = lin.eigh(Gamma) #eigenvalues w[aa] and eigenvectors v[:,aa]
		
		#######################
		#DEBUGGING
		# Calculate the reduced density matrix explicitely for debugging, all eigenvalues of the reduced density matrix can be printed for small system sizes.
		red_calc = 'no'

		if red_calc == 'yes':
			red  =  np.asarray([(1 + np.abs(w[0]))/2,(1 - np.abs(w[0]))/2])
			for oo in range(1,int(np.size(w)/2)):
				hold = np.asarray([(1 + np.abs(w[oo]))/2,(1 - np.abs(w[oo]))/2])	
				red = np.kron(hold,red)
			print('1:sub system', sub_system, 'reduced density eigenvalues', -np.sort(-red))
			print('1:sum eig reduced density',np.sum(red))
			red1 = copy.copy(red)
			######## Testing 
			#calculate eigenvectors of Gamma2
			w,v = lin.eigh(Gamma2[ss]) #eigenvalues w[aa] and eigenvectors v[:,aa]
			# Calculate the reduced density matrix explicitely for debugging, all eigenvalues of the reduced density matrix can be printed for small system sizes.
			red  =  np.asarray([(1 + np.abs(w[0]))/2,(1 - np.abs(w[0]))/2])
			for oo in range(1,int(np.size(w)/2)):
				hold = np.asarray([(1 + np.abs(w[oo]))/2,(1 - np.abs(w[oo]))/2])
				red = np.kron(hold,red)
			print('2: sub system', sub_system, 'reduced density eigenvalues', -np.sort(-red))
			print('2: sum eig reduced density',np.sum(red))
			print('diff', - red + red1)
		##########################
			
		#print("eig_Gamma_list", w)

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


		avoid_var = 0
		total = 0

		for rr in range(0,2*sub_system):
			
			for yy in range(0,2*sub_system):

				total = total + 1
				
				if np.abs(1 - w[rr]*w[yy]) > tol:

					#if np.abs(rr - yy) == 0: #diagonal terms
					
					Fisher_time += 0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,yy]) * (np.conj(v[:,yy]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[yy])

					#print(np.round(np.real(0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,yy]) * (np.conj(v[:,yy]) @ GammaD @ v[:,rr]) / (1 - w[rr]*w[yy])),5), np.round(np.real(Fisher_time),8))


				else:
					
					Fisher_time += 0 #0.5*(np.conj(v[:,rr]) @ GammaD @ v[:,yy]) * (np.conj(v[:,yy]) @ GammaD @ v[:,rr]) / (1 - -1)
					
					avoid_var = avoid_var + 1

		Fisher = Fisher +  [Fisher_time] 

		avoid_index[ss] = avoid_var

		avoid_percentage[ss] = avoid_var/total
		
	print(datetime.now() - startTime,'End Fisher Calculation','Number of avoided divergences',avoid_index, 'percentage', avoid_percentage[ss])

	#print('subsystem',sub_system_range)

	return [Fisher, particle_numL, avoid_percentage]

# def floquet_spectrum(obs, Dag_obs, J, gamma, h0_in, h1, times, dt, sites, integration_type, method, initial_state, boundary_conditions):

# 	if initial_state == 'ground_state':
		
# 		obs, Dag_obs = correlation_groundstate(sites, gamma, J, h0_in, h1[0], boundary_conditions)

# 	if integration_type == 'matrices':

# 			Uk_matrix_list = integrator_BdG(obs, Dag_obs, J, gamma, h0_in, h1, times, dt, 1, sites, boundary_conditions, method)

# 	#Calculate the spectrum of the Uk matrices.




# 	return floq_eigval, floq_eigvec


def floquet_evolution_eff_vectorized(final_time, sites, boundary_conditions, evals, evecs, initial_states):
    """
	# pp can be 1, but for some specific problems in sensing may not be (i.e. Fisher information calc.).
    evals: shape (pp, sites, 2) complex eigenvalues for each pp, k, band
    evecs: shape (pp, sites, 2, 2) eigenvectors for each pp, k, band, component
    initial_states: shape (pp, sites, 2) initial BdG spinors
    """

    # Calculate overlaps ov0 and ov1 for all pp, sites at once:
    # ov0[pp, k] = <evecs[pp,k,:,0] | initial_states[pp,k,:]>
    ov0 = np.sum(np.conj(evecs[..., :, 0]) * initial_states, axis=-1)  # shape (pp, sites)
    ov1 = np.sum(np.conj(evecs[..., :, 1]) * initial_states, axis=-1)  # shape (pp, sites)

    # Raise eigenvalues to final_time power:
    evals_t0 = evals[..., 0] ** final_time  # shape (pp, sites)
    evals_t1 = evals[..., 1] ** final_time  # shape (pp, sites)

    # Construct final_state for all pp,k:
    # final_state[pp, k, :] = evals_t0 * ov0 * evec[:, :, 0] + evals_t1 * ov1 * evec[:, :, 1]
    final_state = (
        (evals_t0[..., np.newaxis] * ov0[..., np.newaxis]) * evecs[..., :, 0]
        + (evals_t1[..., np.newaxis] * ov1[..., np.newaxis]) * evecs[..., :, 1]
    )  # shape (pp, sites, 2)

    # Determine k vector:
    if boundary_conditions == 'PBC':
        kval = -np.pi + 2 * np.arange(sites) * np.pi / sites
    elif boundary_conditions == 'ABC':
        kval = -np.pi + (2 * np.arange(sites) + 1) * np.pi / sites
    else:
        raise ValueError("Unsupported boundary condition")

    # Compute k-space observables:
    obs_kspace = np.abs(final_state[..., 0]) ** 2   # shape (pp, sites)
    Dag_obs_kspace = np.conj(final_state[..., 0]) * final_state[..., 1]  # shape (pp, sites)

    # Fourier transform obs and Dag_obs back to real space for each pp:
    # Use broadcasting and einsum or matrix multiplication for speed.

    # Phase factor: shape (sites, sites)
    m = np.arange(sites)[:, None, None]
    n = np.arange(sites)[None, :, None]
    k = kval[None, None, :]
    phase = np.exp(-1j * (m - n) * k) / sites
    # To handle shape, do for each pp:
    # obs[pp,m,n] = sum_k phase[m,n,k] * obs_kspace[pp,k]
    # Similarly for Dag_obs 

    # But phase shape is (sites, sites, sites), which can be large. So use einsum smartly:

    # Reshape phase for einsum: (m,n,k)
    # obs_kspace: (pp, k)
    # output obs: (pp, m, n)
    obs = np.einsum('mnk,pk->pmn', phase, obs_kspace)
    Dag_obs = np.einsum('mnk,pk->pmn', phase, Dag_obs_kspace)

    # Return obs, Dag_obs, and evals
    return obs, Dag_obs, evals