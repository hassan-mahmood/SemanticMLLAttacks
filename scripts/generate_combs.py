
import random
import itertools
from tqdm import tqdm
import math
import pickle
import os 
import pandas as pd 
import numpy as np 

targetsize=2
leastnumsamples=50
selected_indices=[]
allkeys=[]



#consistentlabels=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/testlabelsconsistent.h5',key='df').iloc[:,:-1].to_numpy()
consistentlabels=pd.read_hdf('/mnt/raptor/hassan/data/oi/Labels/preds/allconsistent.h5',key='df').iloc[:,:-1].to_numpy()
consistentlabels=np.where(consistentlabels>0,1,0)

tree=pickle.load(open(os.path.join('/mnt/raptor/hassan/data/KG_data/oi/tree'),'rb'))
mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))
mapping_inverse = {v: k for k, v in mapping.items()}
origindices=list(mapping.keys())
destindices=list(mapping.values())

temp_no_child_indices=np.where(np.logical_and(tree.sum(1)==0,tree.sum(0)>=1)==True)[0]
no_child_indices=[]


for t in temp_no_child_indices:
    if t in destindices:
        no_child_indices.append(mapping_inverse[t])



topindices=np.where(np.sum(consistentlabels,axis=0)>=leastnumsamples)[0]

toptreeindices=[]
for i in topindices:
    if i in no_child_indices:
        toptreeindices.append(i)


topotherindices=[]
for i in topindices:
    if i not in toptreeindices:
        topotherindices.append(i)

print('Total:',len(topindices),', tree indices:',len(toptreeindices),', all others:',len(topotherindices))

fromtree=destindices
fromall = [k for k in range(consistentlabels.shape[1]) if k not in fromtree]


alltreecombs=list(itertools.combinations(toptreeindices, math.ceil(targetsize/2)))
allothercombs=list(itertools.combinations(topotherindices, targetsize-len(alltreecombs[0])))


# for it in range(500000):
#     if it%10000==0:
#         print(it)
#for it,comb in tqdm(enumerate(itertools.combinations(allindices, targetsize))):
print(toptreeindices)
print('Tree combinations:',len(alltreecombs),', Other combinations:',len(allothercombs))
print('Target set size:',targetsize)
allsumvals=np.sum(consistentlabels,axis=1)

alreadydone=[]
#for it, z in enumerate(tqdm(itertools.product(alltreecombs,allothercombs))):
for epoch in range(100):
    print('Epoch:',epoch)
    for it in tqdm(range(int(1e7))):
        z1=random.sample(alltreecombs,1)
        z2=random.sample(allothercombs,1)
        pindices=sum([list(e) for e in z1+z2],[])
        
        flag=False
        for p in pindices:
            if p in alreadydone:
                flag=True
                break 

        if flag is True:
            continue

        assert(len(list(set(pindices)))==targetsize)
        
        #pindices=random.sample(allindices,targetsize)
    #     pindices=list(comb)
        sumval=np.count_nonzero(np.logical_and(np.sum(consistentlabels[:,pindices],axis=1)==targetsize,allsumvals>targetsize))
        if(sumval>leastnumsamples):
            pindices.sort()
            tempkey='_'.join([str(px) for px in pindices])
            if tempkey not in allkeys:
                allkeys.append(tempkey)
            else:
                continue

            selected_indices.append(pindices)
            print(pindices,sumval)
            for p in pindices:
                alreadydone.append(p)

            # if len(selected_indices)==100:
            #     break

    print(selected_indices)
    pickle.dump(selected_indices,open('/mnt/raptor/hassan/bmvc/sequences/oiconsistentON/least'+str(leastnumsamples)+'_'+str(targetsize),'wb'))



