import sys
sys.path.append('./')
from scripts.tree_loss import *
from tqdm import tqdm 
import pandas as pd 

target_on_off=0
target_set_size=10

labels=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabelsconsistent.h5',key='df').iloc[:,:-1].to_numpy()
labels=np.where(labels<=0,0,1)
alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/oi/',str(target_set_size)),'rb'))
tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))

mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))

origindices=list(mapping.keys())
destindices=list(mapping.values())


num_classes=tree.shape[0]

# consistent=0
# for c in tqdm(range(labels.shape[0])):
# 	current_label=np.zeros((601,),dtype=np.int32)
# 	current_label[destindices]=labels[c,:][origindices]
# 	_,g=check_local_global_consistency(current_label,0,num_classes,tree)
# 	if g is True:
# 		consistent+=1

# 	if consistent%100==0:
# 		print(consistent)

# print(labels.shape,consistent)
# 0/0


for mytargetlabels in alltargetlabels[:10]:
	consistent=0
	select_indices=np.ones((labels.shape[0],),dtype=bool)
	#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
	for k in mytargetlabels:
		#print('Select indices:',k,select_indices.shape)
		select_indices=np.logical_and(select_indices,labels[:,k]==target_on_off)

	print('Current select_indices',np.count_nonzero(select_indices))

	# current_labels=labels[select_indices,:]
	# current_labels=current_labels[:100]

	# current_labels[:,mytargetlabels]=1.0

	# for k in range(current_labels.shape[0]):
	# 	_,g=check_local_global_consistency(current_labels[k,:],0,num_classes,tree)
	# 	if g is True:
	# 		consistent+=1


	# print('Consistent:',consistent)

0/0
target_on_off=0
target_set_size=6

data=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabels3.h5',key='df')

labels=data.iloc[:,:-1].to_numpy()

labels=np.where(labels<=0,0,1)

imgids=data.iloc[:,-1].to_numpy().tolist()
tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))

num_classes=tree.shape[0]
alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/oi/',str(target_set_size)),'rb'))
mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))

origindices=list(mapping.keys())
destindices=list(mapping.values())

finalimgids=[]
for mytargetlabels in alltargetlabels[:10]:

	#select_indices=torch.ones(size=(labels.shape[0],)).type(torch.ByteTensor)
	select_indices=np.ones((labels.shape[0],),dtype=bool)
	#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
	for k in mytargetlabels:
		#print('Select indices:',k,select_indices.shape)
		select_indices=np.logical_and(select_indices,labels[:,k]==target_on_off)

	#print('Total indices:',np.count_nonzero(select_indices))
	final_indices=[]
	select_indices=np.where(select_indices==True)[0].tolist()

	for k in tqdm(range(labels.shape[0])):
		current_label=np.zeros((601,),dtype=np.int32)
		

		current_label[destindices]=labels[k,:][origindices]
		_,g=check_local_global_consistency(current_label,0,num_classes,tree)

		if g is True:
			final_indices.append(k)
			finalimgids.append(imgids[k])
	print('Total indices:',labels.shape[0],'Final indices:',len(final_indices))
	#0/0
	print('Image ids:',len(finalimgids))
	pickle.dump(finalimgids,open('/mnt/raptor/hassan/data/KG_data/oi/consistentimgids','wb'))
	0/0



