import sys
sys.path.append('./')
from scripts.tree_loss import *
from tqdm import tqdm 
import pandas as pd 
import collections 

target_on_off=0
target_set_size=10

# data=pd.read_hdf('/mnt/raptor/hassan/data/lvis/valbalancedconsistentlabels.h5',key='df')
# labels2=data.iloc[:,:-1].to_numpy()
# labels2=np.where(labels2>0,1,0)

# data=pd.read_hdf('/mnt/raptor/hassan/data/lvis/valbalancedlabels.h5',key='df')
# labels=data.iloc[:,:-1].to_numpy()
# labels=np.where(labels>0,1,0)


# print(collections.Counter(np.sum(labels,axis=1)))
# print(collections.Counter(np.sum(labels2,axis=1)))
# 0/0

data=pd.read_hdf('/mnt/raptor/hassan/data/lvis/trainbalancedlabels.h5',key='df')
labels=data.iloc[:,:-1].to_numpy()



imgids=data.iloc[:,-1].to_numpy().tolist()


labels=np.where(labels>0,1,0)


tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/lvis/tree','rb'))

num_classes=tree.shape[0]


finalimgids=[]

#select_indices=torch.ones(size=(labels.shape[0],)).type(torch.ByteTensor)
select_indices=np.ones((labels.shape[0],),dtype=bool)
#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)

#print('Total indices:',np.count_nonzero(select_indices))

newlabels=np.zeros_like(labels,dtype=np.float32)

for k in tqdm(range(labels.shape[0])):
	current_label=np.copy(labels[k,:])

	_,g=check_local_global_consistency(np.copy(current_label),0,num_classes,tree)

	if g is False:
		presentindices=np.where(current_label==1.0)[0]
		
		treenodes=compute_tree_on_global_loss(np.copy(current_label),presentindices,tree)
		current_label[treenodes]=1.0

		_,g=check_local_global_consistency(np.copy(current_label),0,num_classes,tree)


	assert(g is True)
	newlabels[k,:]=np.copy(current_label)

	if k%1000==0:
		tempdata=pd.DataFrame(newlabels)
		tempdata['ids']=imgids
		tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/trainbalancedconsistentlabels.h5',key='df',mode='w')


tempdata=pd.DataFrame(newlabels)
tempdata['ids']=imgids
tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/trainbalancedconsistentlabels.h5',key='df',mode='w')





