import json
import numpy as np
import os
import json
import pandas as pd
from tqdm import tqdm 




alllabels=np.empty((0,1203),dtype=np.float32)
countsum=0
allimgids=[]
mainpath='/mnt/raptor/hassan/data/lvis/train/'
for file in os.listdir(mainpath)[:2]:
	data=pd.read_hdf(os.path.join(mainpath,file),key='df')
	labels=data.iloc[:,:-1].to_numpy()
	tempimgids=data.iloc[:,-1].to_numpy().tolist()
	imgids=[]
	for t in tempimgids:
		imgids.append(t.rjust(16,'0'))
	

	allimgids+=imgids
	alllabels=np.concatenate((alllabels,labels),axis=0)

	countsum+=labels.shape[0]


# (str(imgdata['id'])+'.jpg').rjust(16,'0')



# print(alllabels.shape,countsum,len(allimgids))

# tempdata=pd.DataFrame(alllabels)
# tempdata['ids']=imgids
# tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/trainlabels.h5',key='df',mode='a')


0/0


cat='train'
imagespath='/mnt/raptor/datasets/LVIS/train2017/'
storefile='/mnt/raptor/hassan/data/lvis/'+str(cat)+'labels.h5'
data=json.load(open('/mnt/raptor/datasets/LVIS/lvis_v1_'+str(cat)+'.json','r'))
num_classes=len(data['categories'])

alllabels=np.empty((0,num_classes+1),dtype=np.float32)

imgids=[]
notexist=0
for idx,imgdata in tqdm(enumerate(data['images'])):
	
#     print(os.path.join(imagespath,(str(imgdata['id'])+'.jpg').rjust(16,'0')))
	if not os.path.exists(os.path.join(imagespath,(str(imgdata['id'])+'.jpg').rjust(16,'0'))):
		notexist+=1
		continue
		
	templabel=np.zeros((1,num_classes+1),dtype=np.float32)
	templabel[0,imgdata['not_exhaustive_category_ids']]=1.0
	templabel[0,imgdata['neg_category_ids']]=-1.0
	
	alllabels=np.concatenate((alllabels,templabel),axis=0)
	imgids.append(str(imgdata['id'])+'.jpg')
	if idx%2000==0:
		print(len(imgids),alllabels.shape)
		tempdata=pd.DataFrame(alllabels[:,1:])
		tempdata['ids']=imgids
		tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/trainlabels.h5',key='df',mode='a')
		alllabels=np.empty((0,num_classes+1),dtype=np.float32)
		imgids=[]

tempdata=pd.DataFrame(alllabels[:,1:])
tempdata['ids']=imgids
tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/trainlabels.h5',key='df',mode='a')

print('not exist:',notexist)
#     if (idx+1)%2000==0:
#         print('Completed:',idx)
#         tempdata=pd.DataFrame(alllabels[:idx+1,1:])
#         tempdata['ids']=imgids
#         tempdata.to_hdf(storefile,key='df',mode='w')
		
# tempdata=pd.DataFrame(alllabels[:,1:])
# tempdata['ids']=imgids
# tempdata.to_hdf('',key='df',mode='w')