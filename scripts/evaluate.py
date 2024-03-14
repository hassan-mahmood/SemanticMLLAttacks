import numpy as np 
import os  
import pandas as pd 
import torch 
import collections 
path = '/mnt/raptor/hassan/UAPs/'

orig_preds_path=os.path.join(path,'orig_preds/')

all_orig_preds=[]
all_orig_preds_ids=[]
for f in os.listdir(orig_preds_path):
	data=pd.read_hdf(os.path.join(orig_preds_path,f),key='df',mode='r')
	print(data.shape)
	all_orig_preds.append(torch.tensor(data.iloc[:,:-1].values.astype(np.float32)).cpu().numpy())
	all_orig_preds_ids.append(data.iloc[:,-1].to_numpy())


all_orig_preds=np.concatenate(all_orig_preds)

bool_orig_preds=np.where(all_orig_preds>0,1,0)

all_orig_preds_ids=np.concatenate(all_orig_preds_ids)
print(all_orig_preds.shape,all_orig_preds_ids.shape)

mytargetlabels=[1,4,8,15]

target_path=os.path.join(path,'best_1_4_8_15')

all_target_preds=[]
all_target_preds_ids=[]


for f in os.listdir(target_path):
	data=pd.read_hdf(os.path.join(target_path,f),key='df',mode='r')
	#print(data.shape)
	all_target_preds.append(torch.tensor(data.iloc[:,:-1].values.astype(np.float32)).cpu().numpy())
	all_target_preds_ids.append(data.iloc[:,-1].to_numpy())

all_target_preds=np.concatenate(all_target_preds)
bool_target_preds=np.where(all_target_preds>0,1,0)

all_target_preds_ids=np.concatenate(all_target_preds_ids)

newlabels=np.copy(bool_orig_preds)

for k in mytargetlabels:
	newlabels[:,k]=1-newlabels[:,k]


#select_indices = np.where(np.in1d(all_orig_preds_ids, all_target_preds_ids))[0]

count_success=0
count_total=0

freq=[]
for idx,img_id in enumerate(all_target_preds_ids):

	orig_idx=np.where(all_orig_preds_ids==img_id)[0]
	if(len(orig_idx)==0):
		continue 
	orig_idx=orig_idx[0]
	orig_label=newlabels[orig_idx,:]
	freq.append((bool_target_preds[idx,:]==orig_label).astype(np.float32).sum())
	
	if (bool_target_preds[idx,mytargetlabels]==orig_label[mytargetlabels]).astype(np.float32).sum()==len(mytargetlabels):
		count_success+=1

	count_total+=1


print('Total',count_total)
print('Success:',count_success)
print(collections.Counter(freq))


0/0



for (img_id,pred) in zip(all_target_preds_ids,all_target_preds):
	
	orig_img_idx=np.where(all_orig_preds_ids==img_id)[0][0]

	print(bool_target_preds[:,mytargetlabels]==newlabels[:,mytargetlabels])
	0/0
	
	orig_pred=bool_orig_preds

	0/0



0/0
koutputs=torch.tensor(bool_target_preds)
newlabels=torch.tensor(newlabels)
print(koutputs.shape,newlabels.shape)
0/0

success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
print('Success:',len(success_indices)/koutputs.shape[0])
scores=(koutputs==newlabels).float().sum(1)
# print(scores)
# 0/0

select_indices = np.where(np.in1d(all_orig_preds_ids, np.array([all_target_preds_ids[0],'asdfasdf',all_target_preds_ids[0]])))[0]



success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
#print('Success:',len(success_indices),torch.min(self.e),torch.max(self.e))



#select_indices=np.where(all_target_preds_ids==all_orig_preds_ids)[0]
print(select_indices)
print(all_target_preds_ids.shape)
print(select_indices.shape)


