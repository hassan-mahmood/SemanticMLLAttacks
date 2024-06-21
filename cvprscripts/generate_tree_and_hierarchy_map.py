import json 
import pickle
import pandas as pd 
import torch
import numpy as np
pickle.HIGHEST_PROTOCOL=4

hierarchy_file='/home/hassan/orionrepo/mllattacks/metadata/oi/boxable_classes_hierarchy.json'
boxable_classes='/home/hassan/orionrepo/mllattacks/metadata/oi/boxable_classes.csv'
model_path='/home/hassan/orionrepo/mllattacks/weights/oi/asl/Open_ImagesV6_TRresNet_L_448.pth'
hierarchy_mapping_file='/home/hassan/orionrepo/mllattacks/metadata/oi/hierarchy_mapping'
oi_tree_path='/home/hassan/orionrepo/mllattacks/metadata/oi/tree'

hierarchydata=json.load(open(hierarchy_file,'r'))
hierarchyclasses=pd.read_csv(boxable_classes)

state = torch.load(model_path, map_location='cpu')
idx_to_classname=state['idx_to_class']
classnames_to_idx={}
oiclassnames=[]
for k,tempv in idx_to_classname.items():
    v=str(tempv.strip("\'\"\'")).lower()
    oiclassnames.append(v)
    if v not in classnames_to_idx:
        classnames_to_idx[v]=[k]
        
    else:
        classnames_to_idx[v].append(k)
        # print(tempv,v,classnames_to_idx[v])
    
hierarchy_index_mapping = {v: k for k, v in dict(hierarchyclasses['LabelName']).items()}


################################################
#Get the mapping from the output to the hierarchy
cpcount=0
cncount=0
out_to_hierarchy={}
for hier_idx,l in enumerate(hierarchyclasses['DisplayName']):
    l=l.lower()
    
    if l not in oiclassnames:
        cncount+=1
    else:
        out_idx=oiclassnames.index(l.strip('\'"').lower())
        
        out_to_hierarchy[out_idx]=hier_idx
        cpcount+=1
#        print(l)
#        print(cpcount)
#        0/0
print('Hierarchy Classes found:',cpcount,', not found:',cncount)

pickle.dump(out_to_hierarchy,open(hierarchy_mapping_file,'wb'))

#######################################################################

#Generate Tree

oitree=np.zeros(shape=(601,601),dtype=np.float32)
def process(maplist,current_parent):
    global oitree
    
    #the input will be a map list
    for mapdata in maplist:
        current_child=hierarchy_index_mapping[mapdata['LabelName']]

        if current_parent!=-1:
            oitree[current_parent,current_child]=1.0
        
        #iterate through each map in the list
        for kt in mapdata.keys():
            
            #if isinstance(mapdata[kt],list):
            if kt=='Subcategory':
                process(mapdata[kt],current_child)
                

tempdata=hierarchydata['Subcategory']
process(tempdata,-1)


# Rather than multiple parents, simplify to have only one parent for each node. 
# Randomly select a single parent among all the existing parents

nodes_with_multiple_parents=np.where(oitree.sum(axis=0)>1)[0]
# print(nodes_with_multiple_parents)
for n in nodes_with_multiple_parents:
    temp_ones=np.where(oitree[:,n]==1)[0]
    selected_one=np.random.choice(temp_ones,1)
    oitree[:,n]=0.0
    oitree[selected_one,n]=1.0
    # print(n,temp_ones,selected_one)

print('Node parents:',oitree.sum(axis=0))

pickle.dump(oitree,open(oi_tree_path,'wb'))
#######################################################################



