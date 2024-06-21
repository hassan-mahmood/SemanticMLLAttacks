import torch
import numpy as np 
from tqdm import tqdm
import pandas as pd
import os 
import pickle

pickle.HIGHEST_PROTOCOL=4

mid_to_classnames_file='/home/hassan/orionrepo/mllattacks/metadata/oi/mid_to_classes.pth'
model_weights='/home/hassan/orionrepo/mllattacks/weights/oi/asl/Open_ImagesV6_TRresNet_L_448.pth'
csv_file='/home/hassan/orionrepo/datasets/OpenImages/Labels/clean/data.csv'
clean_label_store_dir='/home/hassan/orionrepo/datasets/OpenImages/Labels/clean/train/labelsfiles/'
images_dir='/home/hassan/orionrepo/datasets/OpenImages/Images/'
if not os.path.exists(clean_label_store_dir):
    os.makedirs(clean_label_store_dir)



# If you want to build the train data, set data_type='train', and for test data, set data_type='val'
data_type='train'

state = torch.load(model_weights, map_location='cpu')
idx_to_classname=state['idx_to_class']
num_classes=state['num_classes']

# classnames_to_idx={str(v.strip("\'\"\'")).lower():k for k,v in idx_to_classname.items()}
#classnames_to_idx={str(v):k for k,v in idx_to_classname.items()}

# midmapping='/mnt/raptor/hassan/data/oi/Labels/class-descriptions.csv'

tempdict=torch.load(mid_to_classnames_file)
# mid_to_classnames_data=pd.read_csv(midmapping)
# tempdict=dict(mid_to_classnames_data.values)
# print(len(list(tempdict.keys())))
#print(mid_to_classnames)
mid_to_classnames={k:str(v.strip("\'\"\'")).lower() for k,v in tempdict.items()}

classnamekeys=[l.lower() for l in mid_to_classnames.values()]
midkeys=[l.lower() for l in mid_to_classnames.keys()]

#i need mid to classid

#idx_to_mid={k:classnames_to_mid[v.strip('\"')] for k,v in idx_to_classname.items()}
mid_to_idx={}

#classnamekeys=

for idx_k,v in idx_to_classname.items():
    v=v.strip('\"\'').lower()
    
    if v not in classnamekeys:
        # print(v,'not in classnames')
        for idxclassname,k in enumerate(classnamekeys):
            tempv = v.strip(' ')
            if tempv in k:
                # Find the position of v in k
                start_index=k.find(tempv)
                end_index=start_index+len(tempv)-1
                
                if start_index>0:
                    # make sure the previous character is not among alphabets
                    if k[start_index-1].isalpha():
                        continue 
                if end_index<len(k)-1:
                    if k[end_index+1].isalpha():
                        continue

                print(v,':',k,classnamekeys[idxclassname],idx_k,idxclassname)
                mid_to_idx[midkeys[idxclassname]]=idx_k

            if k in tempv:
                # Find the position of v in k
                start_index=tempv.find(k)
                end_index=start_index+len(k)-1
                
                if start_index>0:
                    # make sure the previous character is not among alphabets
                    if tempv[start_index-1].isalpha():
                        continue 
                if end_index<len(tempv)-1:
                    if tempv[end_index+1].isalpha():
                        continue

                print(v,':',k,classnamekeys[idxclassname],idx_k,idxclassname)
                mid_to_idx[midkeys[idxclassname]]=idx_k
    else:
        idxclassname=classnamekeys.index(v)
        mid_to_idx[midkeys[idxclassname]]=idx_k
        #idx_to_mid[idx_k]=classnames_to_mid[v]


from ast import literal_eval

imgids=[]
#testlabels=np.zeros(shape=(end_idx-start_idx,9605),dtype=np.float32)
testlabels=[]
# allindices=[]
count=0
midkeys=list(mid_to_idx.keys())
data=pd.read_csv(csv_file)
data=data.loc[data.iloc[:,-1]==data_type]

start_idx=0
end_idx=data.shape[0]
countfile=0
for kidx,k in enumerate(tqdm(range(start_idx,end_idx))):
    if((kidx+1)%50000==0):
        print('Completed:',kidx,', count:',count)
        newdata=pd.DataFrame(np.array(testlabels).squeeze())
        newdata['imgids']=imgids
        newdata.to_hdf(os.path.join(clean_label_store_dir,str(countfile)+'.h5'),key='df')
        countfile+=1
        testlabels=[]
        imgids=[]
        
    if(os.path.exists(os.path.join(images_dir,data.iloc[k,0]))):
        assert(data.iloc[k,-1]==data_type)
        imgids.append(data.iloc[k,0])
        count+=1
        labels=literal_eval(data.iloc[k,1])
        labelindices=[mid_to_idx[l] for l in labels if l in midkeys]
        # allindices+=labelindices
        templabels=np.zeros(shape=(1,num_classes),dtype=np.float32)
        templabels[0,labelindices]=1.0
        
        neglabels=literal_eval(data.iloc[k,2])
        labelindices=[mid_to_idx[l] for l in neglabels if l in midkeys]
        # allindices+=labelindices
        templabels[0,labelindices]=-1.0
        
        testlabels.append(templabels)
        
    else:
        pass


print('Completed:',kidx,', count:',count)
newdata=pd.DataFrame(np.array(testlabels).squeeze())
newdata['imgids']=imgids
newdata.to_hdf(os.path.join(clean_label_store_dir,str(countfile)+'.h5'),key='df')
countfile+=1
testlabels=[]
imgids=[]

print('Total images:',count)
