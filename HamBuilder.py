import numpy as np
import copy
import scipy.sparse as sparse
#Initialisaton for Testing
Sx = 0.5*np.asarray([[0,1],[1,0]])                                                                                                                                                                                 
Sy = 0.5*np.asarray([[0,-1j],[1j,0]])                                                                                                                                                                              
Sz = 0.5*np.asarray([[1,0],[0,-1]])  
#

def tens(X,Y,sites,linear,PBC):
	
	hambuild = np.zeros([2,2])	
	for yy in range(0,sites-1):
		hambuild = np.kron(hambuild,np.zeros([2,2])) # zero matrix for storing final values
	#	if linear == 1 and yy == sites-2-1:
	#		hamPEN = hambuild
	#print(np.size(hambuild),'size hambuild')
	for LEFT in range(0,sites-1): #correct number of iterations, does not get to the exceptional end site	
		sitestore = np.identity(2)	
		for ss in range(0,LEFT-1): #only for n > 3 since LEFT max value 1 so does not go through the loop
			sitestore = np.kron(np.identity(2),sitestore) #correct in the case where 1x .. and 1 x 1 ...	
		#print(LEFT,'left')	
		if LEFT > 0:
			sitestore = np.kron(sitestore,np.kron(X,Y))
			#print(sitestore,'left=1')
		if LEFT == 0:
			sitestore = np.kron(X,Y)
			#print(sitestore,'left=0')
			
		for pp in range(0,sites-LEFT-2):  #inserting identities on the RHS
			sitestore = np.kron(sitestore,np.identity(2))
			#print(pp,'pp')
			#print(sitestore)	
		hambuild = hambuild + sitestore 


	if linear == 1:
		  #end term in the chain, ie. 1x1x1xSx
		sitestore2 = np.identity(2)
		for ss in range(0,sites-2):  #goes through all identities but not include first and last term so -2
			sitestore2 = np.kron(np.identity(2),sitestore2)
		endterm = np.kron(sitestore2,X)
		#print(endterm,'endterm') #works correctly for 2 and 3 site case
		hambuild = hambuild + endterm
	
	if PBC == 1 and sites > 2 and linear == 0:	
				
			sitestore3 = np.kron(X,np.identity(2))
			for ss in range(0,sites-3):
				sitestore3 = np.kron(sitestore3,np.identity(2))
			#endterm3 = np.kron(sitestore3,X)
			endterm3 = np.kron(sitestore3,X)
		
			hambuild = hambuild + endterm3

		
	return hambuild


def statebuild(y):

	up = np.asarray([1,0])	
	down = np.asarray([0,1])	
	if y[0] == 1:
		state = up
	if y[0] == 0:
		state = down

	for zz in range (1,np.size(y)):
		if y[zz] == 1:
			state = np.kron(state,up)  	 
		if y[zz] == 0:
			state = np.kron(state,down)

	return state


def statebuild_bloch_transinvariant(a,b,N):

	up = np.asarray([1,0])	
	down = np.asarray([0,1])	
	unit = a*up + b*down
	state = copy.deepcopy(unit)
	for zz in range (1,N):
		state = np.kron(state,unit)  	 

	return state

#print(statebuild_bloch_transinvariant(0.5,0.5,5))



def tens_single_redef(X1,X2,loc1,loc2,sites):  #for a given x,y at any locati 
	lister = [loc1,loc2]		
	smaller = min(lister)
	larger = max(lister)
	
	idx = lister.index(min(lister))
	if idx == 0:
		X = X1
		Y = X2
	else:
		X = X2
		Y = X1
	
	#print(smaller,'smaller',X)
	#print(larger,'larger',Y)
	
	hambuild = np.zeros([2,2])	

	if smaller > 0:
		#print('initialise')
		sitestore = np.identity(2)	
		for ss in range(0,smaller-1): #identity matrices on the left 
			sitestore = np.kron(np.identity(2),sitestore) 
			#print(ss,'ss')	
		sitestore = np.kron(sitestore,X)
		#print('used left')
	else:
		sitestore = X 
		#print('initialise used left')


	for tt in range(0,larger-smaller-1): 
		sitestore = np.kron(sitestore,np.identity(2))
		#print(tt,'tt')

	sitestore = np.kron(sitestore,Y)
	#print('used right')
	if larger < sites-1:
		for kk in range(0,sites-larger):
			sitestore = np.kron(sitestore,np.identity(2))

	hambuild = sitestore	
	return hambuild

#Location 2 is 1 site too far left? see tens_single_redef above
def tens_single(X1,X2,loc1,loc2,sites):  #for a given x,y at any locati 
	lister = [loc1,loc2]		
	smaller = min(lister)
	larger = max(lister)
	
	idx = lister.index(min(lister))
	if idx == 0:
		X = X1
		Y = X2
	else:
		X = X2
		Y = X1
	
	hambuild = np.zeros([2,2])	

	if smaller > 0:
		#print('initialise')
		sitestore = np.identity(2)	
		for ss in range(0,smaller-1): #identity matrices on the left 
			sitestore = np.kron(np.identity(2),sitestore) 
			#print(ss,'ss')	
		sitestore = np.kron(sitestore,X)
		#print('used left')
	else:
		sitestore = X 
		#print('initialise used left')
	
	for tt in range(0,larger-smaller-1): 
		sitestore = np.kron(sitestore,np.identity(2))
		#print(tt,'tt')
	sitestore = np.kron(sitestore,Y)
	#print('used right')
	if larger < sites:	
		for kk in range(0,sites-larger-1):
			sitestore = np.kron(sitestore,np.identity(2))

	hambuild = sitestore	
	return hambuild

def tens_twosite(X,loc,sites):  #for a given x,y at any locati 
	hambuild = np.zeros([2,2])
	if loc < (sites -1):
		if loc > 0:	
			sitestore = np.identity(2)
			for ss in range(0,loc-1): #identity matrices on the left 
				sitestore = np.kron(np.identity(2),sitestore) 
			sitestore = np.kron(sitestore,X)
		else:
			sitestore = X
		
		for tt in range(loc,sites-2): 
			sitestore = np.kron(sitestore,np.identity(2))

	else:	
		sitestore = np.identity(2)
		for ss in range(0,loc-2): #identity matrices on the left 
			sitestore = np.kron(np.identity(2),sitestore) 
		sitestore = np.kron(sitestore,X)

	hambuild = sitestore	
	return hambuild

	
def tens_twosite_arbd(X,d,loc,sites):  #X is defined on 2 sites i.e. Sx_i Sx_{i+1}
	hambuild = np.zeros([d,d])
	if loc < (sites -1):
		if loc > 0:	
			sitestore = np.identity(d)
			for ss in range(0,loc-1): #identity matrices on the left 
				sitestore = np.kron(np.identity(d),sitestore) 
			sitestore = np.kron(sitestore,X)
		else:
			sitestore = X
		
		for tt in range(loc,sites-2): 
			sitestore = np.kron(sitestore,np.identity(d))


	hambuild = sitestore	
	return hambuild

def tens_twosite_arbd_sparse(X,d,loc,sites):  #X is defined on 2 sites i.e. Sx_i Sx_{i+1}
	hambuild = np.zeros([d,d])
	if loc < (sites -1):
		if loc > 0:	
			sitestore = np.identity(d)
			for ss in range(0,loc-1): #identity matrices on the left 
				sitestore = sparse.kron(np.identity(d),sitestore) 
			sitestore = sparse.kron(sitestore,X)
		else:
			sitestore = X
		
		for tt in range(loc,sites-2): 
			sitestore = sparse.kron(sitestore,np.identity(d))


	hambuild = sitestore	
	return hambuild

def tens_arbphys(X,Y,d,sites,linear,PBC):

	
	hambuild = np.zeros([d,d])	
	for yy in range(0,sites-1):
		hambuild = np.kron(hambuild,np.zeros([d,d])) # zero matrix for storing final values
	#	if linear == 1 and yy == sites-2-1:
	#		hamPEN = hambuild
	#print(np.size(hambuild),'size hambuild')
	for LEFT in range(0,sites-1): #correct number of iterations, does not get to the exceptional end site	
		sitestore = np.identity(d)	
		for ss in range(0,LEFT-1): #only for n > 3 since LEFT max value 1 so does not go through the loop
			sitestore = np.kron(np.identity(d),sitestore) #correct in the case where 1x .. and 1 x 1 ...	
		#print(LEFT,'left')	
		if LEFT > 0:
			sitestore = np.kron(sitestore,np.kron(X,Y))
			#print(sitestore,'left=1')
		if LEFT == 0:
			sitestore = np.kron(X,Y)
			#print(sitestore,'left=0')
			
		for pp in range(0,sites-LEFT-2):  #inserting identities on the RHS
			sitestore = np.kron(sitestore,np.identity(d))
			#print(pp,'pp')
			#print(sitestore)	
		hambuild = hambuild + sitestore 


	if linear == 1:
		  #end term in the chain, ie. 1x1x1xSx
		sitestore2 = np.identity(d)
		for ss in range(0,sites-2):  #goes through all identities but not include first and last term so -2
			sitestore2 = np.kron(np.identity(d),sitestore2)
		endterm = np.kron(sitestore2,X)
		#print(endterm,'endterm') #works correctly for 2 and 3 site case
		hambuild = hambuild + endterm
	
	if PBC == 1 and sites > 2 and linear == 0:	
				
			sitestore3 = np.kron(X,np.identity(d))
			for ss in range(0,sites-3):
				sitestore3 = np.kron(sitestore3,np.identity(d))
			#endterm3 = np.kron(sitestore3,X)
			endterm3 = np.kron(sitestore3,X)
			print(np.size(endterm3),'end')
			hambuild = hambuild + endterm3

		
	return hambuild
