import numpy as np 
from utils.utility import *


def get_classnames(arr):
	return list(map(lambda t: classnames[t], arr))

def turn_off_node(idx,logits,tree,nodes_to_be_off,level):

	#if(logits[idx]!=0):
	nodes_to_be_off.append(idx)

	
	logits[idx]=0
		
	# Get all parents:
	parents=np.where(tree[:,idx]==1)[0]
	children=np.where(tree[idx,:]==1)[0]

	for p in parents:
		process_parent(p,logits,tree,nodes_to_be_off,level+1)

	for c in children:
		process_child(c,logits,tree,nodes_to_be_off,level+1)
		

	return nodes_to_be_off

def get_ON_OFF(nodes,logits):
	return list(map(lambda t: ('OFF','ON')[logits[t]==1],nodes))


def check_all_nodes_off(idx,logits,nodes):
	for n in nodes:
		if(logits[n]==1):
			return False

	return True


def process_parent(idx,logits,tree,nodes_to_be_off,level):
	# A parent can be ON or OFF. If its ON, then at least one children must be ON

	#print('Current node being processed as parent:',self.classnames[idx])
	children=np.where(tree[idx,:]==1)[0]
	#print('Processing children of',alphabets[idx],list(map(lambda t:alphabets[t],children)))
	
	if(logits[idx]==1):
		# See if all children are off
		if(check_all_nodes_off(idx,logits,children)):
			#$print('All children off:',self.classnames[idx])
			#print('Node:',alphabets[idx],'all children off')
			turn_off_node(idx,logits,tree,nodes_to_be_off,level+1)

	else:
		if(not check_all_nodes_off(idx,logits,children)):
			for c in children:
				if(logits[c]==1):
					turn_off_node(c,logits,tree,nodes_to_be_off,level+1)


	parents=np.where(tree[:,idx]==1)[0]
	for p in parents:
		process_parent(p,logits,tree,nodes_to_be_off,level+1)
		# For each child, if there are multiple parents ON, then leave the child ON. otherwise, turn the child OFF
	#else:
		#print('Not all children are off.',list(zip(self.get_classnames(children),self.get_ON_OFF(children,logits))))
	
	#else: It means the parent needs to remain ON
 
def process_child(idx,logits,tree,nodes_to_be_off,level):
	# Get its children
	# if(logits[idx]==0):
	# 	# If it is already off, we don't need to consider anything else
	# 	return

	# #print('Current node being processed as child:',self.classnames[idx])
	# #print('Processing parents of',alphabets[idx],list(map(lambda t:alphabets[t],parents)))
	# # See if all children are off
	# if(check_all_nodes_off(idx,logits,parents)):
	# 	#print('All parents off:',self.classnames[idx])
	# 	#print('Node:',alphabets[idx],'all parents off')
	# 	turn_off_node(idx,logits,tree,nodes_to_be_off,level+1)


	if(logits[idx]!=0):
		nodes_to_be_off.append(idx)

	
	logits[idx]=0

	children= np.where(tree[idx,:]==1)[0]
	for c in children:
		if(logits[c]!=0):
			nodes_to_be_off.append(c)
			logits[c]=0


	for c in children:
		process_child(c,logits,tree,nodes_to_be_off,level+1)

	#else:
		#print('Not all parents are off',list(zip(self.get_classnames(parents),self.get_ON_OFF(parents,logits))))

	#else: It means the parent needs to remain ON
def compute_tree_loss(logits,target_class,tree):
	return turn_off_node(target_class,logits,tree,[],level=0)

# def compute_tree_on_loss(logits,target_classes,tree):
	
	
# 	allnodes=[]
# 	logits[target_classes]=1.0
# 	for current_node in target_classes:
# 		while(True):
# 			parents=np.where(tree[:,current_node]==1.0)[0]
# 			# if there is no parent or at least one parent is on
# 			if len(parents)==0 or np.sum(logits[parents])>0:
# 				break 

# 			allnodes.append(parents[0])
# 			logits[parents[0]]=1.0
# 			current_node=parents[0]
	
# 	for current_node in target_classes:
		
# 		while True:
# 			child=np.where(tree[current_node,:]==1.0)[0]
# 			if len(child)==0 or np.sum(logits[child])>0:
# 				break 

# 	return list(set(allnodes))

def compute_tree_on_global_parent_loss(logits,target_class,tree):

	newnodes=[]
	current_node=target_class
	logits[current_node]=1.0
	newnodes.append(current_node)

	parents=np.where(tree[:,current_node]==1.0)[0]

	if len(parents)==0:
		return newnodes

	present_indices=parents[np.where(logits[parents]>0)[0]]
	
	
	if len(present_indices)>0:
		
		for p in present_indices:
			newnodes+=compute_tree_on_global_parent_loss(logits,p,tree)

	else:
		newnodes+=compute_tree_on_global_parent_loss(logits,parents[0],tree)

	return newnodes

def compute_tree_on_global_child_loss(logits,target_class,tree):

	newnodes=[]
	current_node=target_class
	
	logits[current_node]=1.0
	newnodes.append(current_node)

	child=np.where(tree[current_node,:]==1.0)[0]

	if len(child)==0:
		return newnodes

	present_indices=child[np.where(logits[child]>0)[0]]

	
	if len(present_indices)>0:
		for p in present_indices:
			newnodes+=compute_tree_on_global_child_loss(logits,p,tree)

	else:
		
		newnodes+=compute_tree_on_global_child_loss(logits,child[0],tree)


	return newnodes

def compute_tree_on_global_loss(logits,target_classes,tree):
	
	
	allnodes=[]
	logits[target_classes]=1.0
	for current_node in target_classes:
		allnodes+=compute_tree_on_global_parent_loss(logits,current_node,tree)
		allnodes+=compute_tree_on_global_child_loss(logits,current_node,tree)

		# parents=np.where(tree[:,current_node]==1.0)[0]
		# for p in parents:
		# 	allnodes+=compute_tree_on_global_parent_loss(logits,p,tree)

		# child=np.where(tree[current_node,:]==1.0)[0]

		# for p in child:
		# 	allnodes+=compute_tree_on_global_child_loss(logits,p,tree)
	return list(set(allnodes))

# def compute_tree_on_global_loss(logits,target_classes,tree):
	
	
# 	allnodes=[]
# 	logits[target_classes]=1.0
# 	for current_node in target_classes:
# 		while(True):
# 			parents=np.where(tree[:,current_node]==1.0)[0]
# 			# if there is no parent or at least one parent is on
# 			if len(parents)==0:
# 				break 

# 			present_indices=np.where(logits[parents]>0)[0]

# 			if len(present_indices)>0:
# 				current_node=present_indices[0]
# 				logits[current_node]=1.0
# 				allnodes.append(current_node)
# 			else:
# 				logits[parents[0]]=1.0
# 				allnodes.append(parents[0])
# 				current_node=parents[0]
			
# 	for current_node in target_classes:
		
# 		while True:
# 			child=np.where(tree[current_node,:]==1.0)[0]

# 			if len(child)==0:
# 				break 

# 			present_indices=np.where(logits[child]>0)[0]

# 			if len(present_indices)>0:
# 				current_node=present_indices[0]
# 				logits[current_node]=1.0
# 				allnodes.append(current_node)
# 			else:
# 				logits[child[0]]=1.0
# 				allnodes.append(child[0])
# 				current_node=child[0]

# 	return list(set(allnodes))

# def compute_tree_on_loss(logits,target_classes,tree):
	
	
# 	allnodes=[]
# 	logits[target_classes]=1.0
# 	for current_node in target_classes:
# 		while(True):
# 			parents=np.where(tree[:,current_node]==1.0)[0]
# 			# if there is no parent or at least one parent is on
# 			if len(parents)==0:
# 				break 

# 			present_indices=parents[np.where(logits[parents]>0)[0]]
# 			if len(present_indices)>0:
# 				current_node=present_indices[0]
# 				logits[current_node]=1.0
# 				allnodes.append(current_node)
# 			else:
# 				logits[parents[0]]=1.0
# 				allnodes.append(parents[0])
# 				current_node=parents[0]
			
# 	for current_node in target_classes:
		
# 		while True:
# 			child=np.where(tree[current_node,:]==1.0)[0]

# 			if len(child)==0:
# 				break 

# 			present_indices=child[np.where(logits[child]>0)[0]]

# 			if len(present_indices)>0:
# 				current_node=present_indices[0]
# 				logits[current_node]=1.0
# 				allnodes.append(current_node)
# 			else:
# 				logits[child[0]]=1.0
# 				allnodes.append(child[0])
# 				current_node=child[0]

# 	return list(set(allnodes))

def compute_tree_off_loss(logits,target_classes,tree):
	
	
	allnodes=[]
	logits[target_classes]=0.0

	for target_class in target_classes:
		allnodes+=turn_off_node(target_class,logits,tree,[],level=0)

	return list(set(allnodes))

def check_child_consistency(logits,idx,tree):
	# if current node is ON, at least one child should be ON.
	# If current node is OFF, all children must be off

	children=np.where(tree[idx,:]==1)[0]

	if(len(children)==0):
		return True

	if(logits[idx]==1):
		#print('logits ',idx,'is ON')
		alloff=True
		for c in children:
			if(not check_child_consistency(logits,c,tree)):
				return False 

			#print('Child of',c,'consistent')
			if(logits[c]==1):
				#print(c,'is ON')
				alloff=False

		if(alloff):
			return False


	if(logits[idx]==0):
		#print('logits ',idx,'is OFF')
		alloff=True
		for c in children:
			if(not check_child_consistency(logits,c,tree)):
				return False 

			if(logits[c]==1):
				# if at least one parent is on, then pass
				tempparents=np.where(tree[:,c]==1.0)[0]
				if len(np.where(logits[tempparents]>0)[0])>0:
					pass 
				else:
					alloff=False

		if(alloff==False):
			return False

	return True


def check_parent_consistency(logits,idx,tree):
	parents=np.where(tree[:,idx]==1)[0]	
	children=np.where(tree[idx,:]==1)[0]


	if(len(parents)==0):
		return True 


	#print('Prent;',parents)

	if(logits[idx]==1):
		# Then its parent should be ON
		alloff=True
		for p in parents:
			#
			if (not check_parent_consistency(logits,p,tree)):
				return False 

			if(logits[p]==1):
				#print('Parent is ON:')
				# if(not check_parent_consistency(logits,p,tree)):
				# 	print('Parent not consistent')
				# 	return False 

				alloff=False

		if(alloff):
			return False

	if(logits[idx]==0):
		# Then its parent should be OFF. If the parent is ON
		# Then at least one of its child must be ON
		alloff=True
		for p in parents:
			if (not check_parent_consistency(logits,p,tree)):
				return False 

			if(logits[p]==1):
				children=np.where(tree[p,:]==1)[0]
				onechildon=False
				for c in children:
					if(logits[c]==1):
						onechildon=True
				
				if(not onechildon):
					return False

	return True



def check_local_global_consistency(logits,target_class,num_classes,tree):
	localconsistent=True
	globalconsistent=True
	localidx=int(target_class)
	if(check_consistency(np.copy(logits),localidx,tree)==False):
		localconsistent=False
		globalconsistent=False
	else:
		for localidx in range(num_classes):
			if(check_consistency(np.copy(logits),localidx,tree)==False):
				globalconsistent=False
				break

	return localconsistent,globalconsistent

def check_consistency(logits,target_class,tree):
	# if current node is ON
	idx=int(target_class)
	#print('Tree:',tree.shape,logits.shape,idx,type(idx))

	children=np.where(tree[idx,:]==1)[0]
	parent=np.where(tree[:,idx]==1)[0]
	
	if(not check_child_consistency(logits,idx,tree)):
		return False 
	# print('Done childre')
	if(not check_parent_consistency(logits,idx,tree)):
		return False 

	return True 
	
	# if(logits[idx]==1):
	# 	# At least one children should be ON
	# 	if(len(children)!=0):
	# 		alloff=True
	# 		for c in children:
	# 			if(logits[c]==1):
	# 				alloff=False

	# 		if(alloff):
	# 			return False


	# else:
		#if logits[idx]==0


# class Tree_Loss:
# 	def __init__(self):
# 		self.classnames=get_pickle_data('KG_data/classnames_pickle')


# 	def get_classnames(self,arr):
# 		return list(map(lambda t: self.classnames[t], arr))

# 	def turn_off_node(self,idx,logits,tree,nodes_to_be_off):
# 		if(logits[idx]==0):
# 			return

# 		#print('Turning Off Node:',self.classnames[idx])
# 		nodes_to_be_off.append(idx)
# 		#print('Node',alphabets[idx],'appended')
# 		logits[idx]=0
# 		# Get all parents:
# 		parents=np.where(tree[:,idx]==1)[0]
# 		#print('Parents of',self.classnames[idx],':',self.get_classnames(parents))
# 		for p in parents:
# 			self.process_parent(p,logits,tree,nodes_to_be_off)


# 		children=np.where(tree[idx,:]==1)[0]
# 		#print('Children of',self.classnames[idx],':',self.get_classnames(children))
# 		for c in children:
# 			self.process_child(c,logits,tree,nodes_to_be_off)
# 			#turn_off_node(c,logits,tree,nodes_to_be_off)

# 		return nodes_to_be_off

# 	def get_ON_OFF(self,nodes,logits):
# 		return list(map(lambda t: ('OFF','ON')[logits[t]==1],nodes))


# 	def check_all_nodes_off(self,idx,logits,nodes):
# 		for n in nodes:
# 			if(logits[n]==1):
# 				return False

# 		return True


# 	def process_parent(self,idx,logits,tree,nodes_to_be_off):
# 		# Get its children
# 		if(logits[idx]==0):
# 			# If it is already off, we don't need to consider anything else
# 			return

# 		#print('Current node being processed as parent:',self.classnames[idx])
# 		children=np.where(tree[idx,:]==1)[0]
# 		#print('Processing children of',alphabets[idx],list(map(lambda t:alphabets[t],children)))
# 		# See if all children are off
# 		if(self.check_all_nodes_off(idx,logits,children)):
# 			#$print('All children off:',self.classnames[idx])
# 			#print('Node:',alphabets[idx],'all children off')
# 			self.turn_off_node(idx,logits,tree,nodes_to_be_off)
# 		#else:
# 			#print('Not all children are off.',list(zip(self.get_classnames(children),self.get_ON_OFF(children,logits))))
		
# 		#else: It means the parent needs to remain ON

# 	def process_child(self,idx,logits,tree,nodes_to_be_off):
# 		# Get its children
# 		if(logits[idx]==0):
# 			# If it is already off, we don't need to consider anything else
# 			return

# 		#print('Current node being processed as child:',self.classnames[idx])
# 		parents=np.where(tree[:,idx]==1)[0]
# 		#print('Processing parents of',alphabets[idx],list(map(lambda t:alphabets[t],parents)))
# 		# See if all children are off
# 		if(self.check_all_nodes_off(idx,logits,parents)):
# 			#print('All parents off:',self.classnames[idx])
# 			#print('Node:',alphabets[idx],'all parents off')
# 			self.turn_off_node(idx,logits,tree,nodes_to_be_off)
# 		#else:
# 			#print('Not all parents are off',list(zip(self.get_classnames(parents),self.get_ON_OFF(parents,logits))))

# 		#else: It means the parent needs to remain ON


# 	def compute(self,logits,target_class,tree):

# 		return self.turn_off_node(target_class,logits,tree,[])


# from utility import *
# import pandas as pd 
# from tqdm import tqdm 

# classnames=get_pickle_data('KG_data/classnames_pickle')
# [tree]=get_pickle_data('KG_data/final_tree')

# # logits=np.random.rand(5000,)
# # logits[544]=1
# # loss=Tree_Loss()
# # out=loss.compute(logits,544,tree)
# # print(len(out))
# #print(list(map(lambda t: classnames[t],np.where(tree[544,:]==1)[0])))
# #0/0






# # num=14
# # arr=np.zeros(shape=(num,num),dtype=np.int32)
# # arr[0,3:6]=1
# # arr[1,3]=1
# # arr[2,5]=1
# # arr[6,7]=1
# # arr[3,7]=1
# # arr[4,8]=1
# # arr[5,8]=1
# # arr[7,10:12]=1
# # arr[8,12]=1
# # arr[9,12:14]=1
# # #arr[3,10]=1

# # alphabets=['A','B','C','D','E','F','G','H','I','J','K','L','M','N']
# # logits_map={
# # 	'A':1,
# # 	'B':0,
# # 	'C':1,
# # 	'D':0,
# # 	'E':1,
# # 	'F':1,
# # 	'G':1,
# # 	'H':1,
# # 	'I':1,
# # 	'J':0,
# # 	'K':1,
# # 	'L':0,
# # 	'M':1,
# # 	'N':0,
# # }
# # #logits=np.array([1,0,1,1,1,1,0,1,1,0,1,0,1,0])
# # logits=np.array(list(logits_map.values()))
# # target_class=10
# # tree=np.transpose(arr)
# #logits=np.transpose(logits)


# def check_all_nodes_off(logits,nodes):
# 	if(len(nodes)==0):
# 		return False

# 	for n in nodes:
# 		if(logits[n]==1):
# 			return False

# 	return True

# def process_node(idx,logits,tree,processed_nodes):
# 	if(idx in processed_nodes):
# 		return False # No Attack if aleady been processed

# 	processed_nodes.append(idx)
# 	parents=np.where(tree[:,idx]==1)[0]
# 	print('parents:',list(map(lambda t: classnames[t],parents)))
# 	if(check_all_nodes_off(logits,parents)):
# 		print('Node',classnames[idx],'had all parents off!')
# 		return True
# 	else:
# 		for p in parents:
# 			if(logits[p]==1):
# 				if(process_node(p,logits,tree,processed_nodes)):
# 					return True

# 	children=np.where(tree[idx,:]==1)[0]
# 	print('children:',list(map(lambda t: classnames[t],children)))
# 	if(check_all_nodes_off(logits,children)):
# 		print('Node',classnames[idx],'had all children off!')
# 		return True

# 	for c in children:
# 		if(logits[c]==1):
# 			if(process_node(c,logits,tree,processed_nodes)):
# 				return True

# 	return False



# types=['type1','type2','type3']
# #types=['type4']
# epsilons=list(range(0,60,5))
# filepath='/mnt/xfs1/hassan2/data/baseline/test/'

# classes=[3559,1777]

# select_indices=[]
# not_attacked_count=0
# for c in classes:
# 	for t in types:
# 		print('Type:',t)
# 		for eps in epsilons:
# 			eps=0
# 			not_attacked_count=0
# 			hdfread=pd.read_hdf(os.path.join(filepath,t,str(c),str(eps)+'.h5'),key='df')
# 			preds=hdfread.iloc[:,:-1].to_numpy()
# 			preds=1/(1+np.exp(-1*preds))
# 			oldpreds=preds.copy()
# 			preds=np.where(preds>=0.4,1,0)

# 			# if(eps!=0):
# 			# 	preds=preds[np.array(select_indices),:]

# 			for i in tqdm(range(preds.shape[0])):
# 				logits=preds[i,:]
# 				ON_nodes=np.where(logits==1)[0]
# 				processed_nodes=[]
# 				attack=False
# 				for n in ON_nodes:
# 					# Get its parents
# 					if(process_node(n,logits,tree,processed_nodes)):
# 						attack=True
# 						break

# 				if(attack):
# 					pass
# 					#count_attacked+=1
# 					print('Was attacked!')

# 				else:
# 					if(eps==0):
# 						select_indices.append(i)
# 					not_attacked_count+=1
# 					#print('Not attacked!')
			
# 			print('eps:,',eps,', Not attacked:',not_attacked_count,', out of:',preds.shape)
# 			0/0
# # print('Children:')
# # for idx,a in enumerate(alphabets):
# # 	print(a,list(map(lambda t:alphabets[t], list(np.where(arr[idx,:]==1)[0]))))
# # loss=Tree_Loss()
# # out=loss.compute(logits,target_class,tree)

# # print(list(map(lambda t: alphabets[t],out)))
