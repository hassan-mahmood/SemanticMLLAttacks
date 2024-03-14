import sys
sys.path.append('./')
from scripts.tree_loss import *
from tqdm import tqdm 
import pandas as pd 
import collections 
from utils.utility import *

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

data=pd.read_hdf('/mnt/raptor/hassan/data/csloi/Labels/cleanoutputs.h5',key='df')
origlabels=data.iloc[:,:-1].to_numpy()


imgids=data.iloc[:,-1].to_numpy().tolist()


labels=np.where(origlabels>0,1,0)


tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))


mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))
mapping_inverse = {v: k for k, v in mapping.items()}
origindices=list(mapping.keys())
destindices=list(mapping.values())


num_classes=tree.shape[0]

#select_indices=torch.ones(size=(labels.shape[0],)).type(torch.ByteTensor)
select_indices=np.zeros((labels.shape[0],),dtype=bool)
#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)

#print('Total indices:',np.count_nonzero(select_indices))

newlabels=np.zeros_like(labels,dtype=np.float32)


for target_set_size in [2,3,4,5]:
	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/oiconsistentON/','new'+str(target_set_size)),'rb'))
	
	current_path=os.path.join('/mnt/raptor/hassan/data/csloi/Labels/targeted/',str(target_set_size))
	create_folder(current_path)

	for target_label in alltargetlabels:
		treetargetlabels=[mapping[c] for c in target_label if c in mapping.keys()]

		select_indices=np.ones_like(labels[:,0],dtype=np.float32)
		for t in target_label:
			select_indices=np.logical_and(select_indices==1.0,labels[:,t]==1.0).astype(np.float32)

		target_specific_labels=origlabels[select_indices==1.0,:]
		target_specific_imgids=np.array(imgids)[select_indices==1.0].tolist()
		top_l_indices=[]
		top_g_indices=[]

		if len(treetargetlabels)==0:
			tempdata=pd.DataFrame(target_specific_labels[:50,:])
			tempdata['ids']=np.array(target_specific_imgids)[:50].tolist()
			# print(tempdata)
			tempdata.to_hdf(os.path.join(current_path,str('_'.join([str(k) for k in target_label]))+'.h5'),key='df',mode='w')
			
			continue


		for k in tqdm(range(target_specific_labels.shape[0])):


			current_label=np.zeros((601,),dtype=np.int32)
			current_label[destindices]=np.copy(labels[k,:][origindices])

			# current_label=np.copy(labels[k,:])
			all_l=True 
			for t in treetargetlabels:
				
				l,g=check_local_global_consistency(np.copy(current_label),t,num_classes,tree)
				all_l=all_l and l


			if g is True:
				top_g_indices.append(k)
			elif all_l is True:
				top_l_indices.append(k)

			if len(top_g_indices)>50:
				break


		finalindices=top_g_indices+top_l_indices
		finalindices=finalindices[:50]

		print(target_label,'Global consistent:',len(top_g_indices),'Local consistent:',len(top_l_indices))
		tempdata=pd.DataFrame(target_specific_labels[finalindices,:])
		tempdata['ids']=np.array(target_specific_imgids)[finalindices].tolist()
		# print(tempdata)
		tempdata.to_hdf(os.path.join(current_path,str('_'.join([str(k) for k in target_label]))+'.h5'),key='df',mode='w')

	

# print(np.count_nonzero(select_indices==1.0))


# indices=np.where(select_indices==1)[0]
# tempdata=pd.DataFrame(origlabels[indices,:])
# tempdata['ids']=np.array(imgids)[indices].tolist()
# tempdata.to_hdf('/mnt/raptor/hassan/data/oi/Labels/preds/consistent5.h5',key='df',mode='w')
