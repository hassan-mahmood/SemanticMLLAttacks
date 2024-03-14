import pickle
import torchtext
import numpy as np 
from tqdm import tqdm 
import pandas as pd 

# data=pd.read_hdf('/mnt/raptor/hassan/data/nus/Labels/newtrainlabels.h5',key='df').iloc[:,:-1].to_numpy()

# print(np.sum(data,axis=0))
# 0/0
# out=np.dot(data.T,data)
# count=np.diagonal(out).copy()
# print(count)
# np.fill_diagonal(out,0)

# 0/0


classnames=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/orig_classnames','rb'))
#classnames=['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'monitor']
#f=open('/mnt/raptor/hassan/data/KG_data/nus/nus_classnames','wb')
# pickle.dump(classnames,f)
# f.close()
#nodenames=pickle.load(open('/mnt/raptor/hassan/data/KG_data/nus/node_names','rb'))
#print(list(zip(nodenames,classnames)))

#pickle.store(classnames,'/mnt/raptor/hassan/data/KG_data/nus/orig_classnames')

# print(classnames)
# print(classnames[:20])
# data=np.array([
# [1,0,0,0,1],
# [0,1,1,0,1],
# [1,0,0,1,0],
# [1,1,0,0,1],
# [0,0,0,1,1]])
# tuples=np.where(data==1)
# print(tuples)
# print(data)




# data=pickle.load(open('MLGCN/data/nus/nus_glove_word2vec.pkl','rb'))
# print(data.shape)


#0/0
# For adj
data=pd.read_hdf('/mnt/raptor/hassan/data/nus/Labels/newtrainlabels.h5',key='df')
data=pd.read_hdf('/mnt/raptor/hassan/data/voc/vocdevkit/voc2012/ImageSets/Main/train_labels.h5',key='df')
data=data.iloc[:,:-1].to_numpy()

out=np.dot(data.T,data)
count=np.diagonal(out).copy()
print(count)
np.fill_diagonal(out,0)


tostore={'nums':count,'adj':out}

f=open('MLGCN/data/voc/voc_adj.pkl','wb')
pickle.dump(tostore,f)
f.close()
data=pickle.load(open('MLGCN/data/voc/voc_adj.pkl', 'rb'))
print(data['adj'].shape)

# 0/0
nodenames=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/node_names','rb'))
# print(nodenames)
# 0/0

#print(list(zip(classnames,nodenames)))

glove = torchtext.vocab.GloVe(name="6B", dim=300)
all_embs=np.zeros(shape=(len(nodenames),300),dtype=np.float32)

for cidx,c in tqdm(enumerate(nodenames)):
	if ('_' in c):
		c=c.split('_')
	else:
		c = [c]
	embedding=np.zeros(shape=(300,),dtype=np.float32)

	for idx in c:
		embedding+=np.array(glove[idx])
	
	embedding=embedding/len(c)
	all_embs[cidx,:]=embedding



# print(glove['cat'][:10])
#print(glove['vertebrate'][:10])
#0/0
#out=(glove['dining']+glove['table'])/2
#out=glove['sofa']
#print(out[:5])
#data=pickle.load(open('MLGCN/data/voc/voc_glove_word2vec.pkl', 'rb'))

f=open('MLGCN/data/voc/voc_glove_word2vec.pkl','wb')
pickle.dump(all_embs,f)
f.close()
#print(np.array(data[:,:5]))
#print(data)
#print(data.shape)




