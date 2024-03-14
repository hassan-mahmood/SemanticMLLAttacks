import sys
sys.path.append('./')
from scripts.tree_loss import *
from tqdm import tqdm 
import pandas as pd 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


target_on_off=1
target_set_size=1
tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))
#num_classes=tree.shape[0]
mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))
mapping_inverse = {v: k for k, v in mapping.items()}
origindices=list(mapping.keys())
destindices=list(mapping.values())

norm='inf'
eps_val=0.05


allsuccesses=[]
allconsistent=[]
alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/oiconsistentON/',str(target_set_size)),'rb'))

# alltargetlabels=[[1423], [1562], [1694], [1706], [1816], [1954], [2193], [2643], [2788], [2903], [344], [371], [940]]
# allnt=[]
# for attackname in ['mla','mlagraph','mlaproj','gmla']:
#     consistentpercent=[]
#     successpercent=[]
#     classnt=[]
#     for current_target_labels in tqdm(alltargetlabels):
#         current_store_folder=os.path.join('/mnt/raptor/hassan/rebut/OFF/1/',str(attackname),str(current_target_labels[0]))
#         #current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/oi/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
#         if not os.path.exists(os.path.join(current_store_folder,'clean.h5')):
#         	print('Skipping',current_target_labels)
#         	continue
#         print('Working',current_target_labels)
        
#         nt=np.array(pickle.load(open(os.path.join(current_store_folder,'storent'),'rb')))
#         indices=nt!=0
#         classnt.append(((np.sum(-1*nt[indices]))/np.count_nonzero(indices))/9604)

#         #print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])

#     print(classnt)
#     meansuc=np.mean(np.array(classnt))
    
#     allnt.append(meansuc)
    

# print('NT:',allnt)

# 0/0

# # target_on_off=1
# # target_set_size=1
# # tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))
# # #num_classes=tree.shape[0]
# # mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))
# # mapping_inverse = {v: k for k, v in mapping.items()}
# # origindices=list(mapping.keys())
# # destindices=list(mapping.values())

# # norm=2
# # eps_val=0.004


# # allsuccesses=[]
# # allconsistent=[]
# # alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/oiconsistentON/',str(target_set_size)),'rb'))
# # alltargetlabels=[[1423], [1562], [1694], [1706], [1816], [1954], [2193], [2643], [2788], [2903], [344], [371], [940]]
# # for attackname in ['mla','mlagraph','mlaproj','gmla']:
# #     consistentpercent=[]
# #     successpercent=[]
# #     for current_target_labels in tqdm(alltargetlabels):
# #         current_store_folder=os.path.join('/mnt/raptor/hassan/rebut/OFF/1/',str(attackname),str(current_target_labels[0]))
# #         #current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/oi/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
# #         if not os.path.exists(os.path.join(current_store_folder,'clean.h5')):
# #         	print('Skipping',current_target_labels)
# #         	continue
# #         print('Working',current_target_labels)
# #         clean=pd.read_hdf(os.path.join(current_store_folder,'clean.h5'),key='df').iloc[:,:-1].to_numpy()
# #         adv=pd.read_hdf(os.path.join(current_store_folder,'adv.h5'),key='df').iloc[:,:-1].to_numpy()
# #         clean=np.where(clean>0,1,0)
# #         adv=np.where(adv>0,1,0)
# #         mytargetlabels=current_target_labels

# #         templabels=np.copy(clean)

# #         #templabels[:,mytargetlabels]=(1 - target_on_off)	

# #         # cleanconsistentcount=0

# #         # for c in tqdm(range(adv.shape[0])):
# #         # 	current_label=np.zeros((601,),dtype=np.int32)
# #         # 	current_label[destindices]=np.copy(templabels[c,:][origindices])
# #         # 	_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
# #         # 	#print('G:',g)
# #         # 	if g is True:
# #         # 		cleanconsistentcount+=1

# #         # print(templabels.shape,cleanconsistentcount)


        

# #         select_indices=np.ones((adv.shape[0],),bool)
# #         #print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
# #         for k in mytargetlabels:
# #             #print('Select indices:',k,select_indices.shape)
# #             select_indices=np.logical_and(select_indices,templabels[:,k]==1)

# #         cleanindices=np.copy(select_indices)

# #         templabels=np.copy(adv)
# #         success_indices=templabels[:,mytargetlabels[0]]==0

# #         success=np.count_nonzero(np.logical_and(select_indices,success_indices))/np.count_nonzero(select_indices)

# #         # select_indices=np.where(select_indices==True)[0]
# #         select_indices=np.where(np.logical_and(select_indices,success_indices)==True)[0]
# #         #templabels[:,mytargetlabels]=(1 - target_on_off)	

# #         cleanconsistentcount=0
# #         local=0
# #         for c in select_indices:
# #             current_label=np.zeros((601,),dtype=np.int32)
# #             current_label[destindices]=np.copy(templabels[c,:][origindices])

# #             targetlabels=[mapping[t] for t in mytargetlabels if t in mapping.keys()]

# #             l=True 
# #             for k in targetlabels:
# #                 templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
# #                 l=l and templ


# #             #print('G:',g)
# #             if l is True:
# #                 local+=1

# #         #successpercent.append(len(select_indices)/templabels.shape[0])
# #         successpercent.append(success)
# #         consistentpercent.append(local/np.count_nonzero(cleanindices))
# #         #print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])

# #     meansuc=np.mean(np.array(successpercent))
# #     meanconsistent=np.mean(np.array(consistentpercent))
# #     print('target size:',target_set_size,', Mean success:',meansuc,', Consistent:',meanconsistent)
# #     allsuccesses.append(meansuc)
# #     allconsistent.append(meanconsistent)

# # print('Success:',allsuccesses)
# # print('Consistent:',allconsistent)
# # 0/0

# # target_on_off=0

# # target_set_size=1
# # tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/lvis/tree','rb'))
# # #num_classes=tree.shape[0]

# # norm=float(np.inf)
# # eps_val=0.004
# # attackname='pgd_consistent_mlagmla_linf' 

# # allsuccesses=[]
# # allconsistent=[]

# # for target_set_size in [3]:
# # 	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/lvisconsistent/',str(target_set_size)),'rb'))

# # 	consistentpercent=[]
# # 	successpercent=[]
# # 	for current_target_labels in tqdm(alltargetlabels):
# # 		current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/lvis/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		
# # 		clean=pd.read_hdf(os.path.join(current_store_folder,'clean.h5'),key='df').iloc[:,:-1].to_numpy()

# # 		adv=pd.read_hdf(os.path.join(current_store_folder,'adv.h5'),key='df').iloc[:,:-1].to_numpy()
# # 		if adv.shape[0]==0:
# # 			continue
		
		
# # 		clean=np.where(clean>0,1,0)
# # 		adv=np.where(adv>0,1,0)
# # 		mytargetlabels=current_target_labels
		
# # 		templabels=np.copy(adv)
		
# # 		select_indices=np.ones((adv.shape[0],),bool)

# # 		#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
# # 		for k in mytargetlabels:
# # 			#print('Select indices:',k,select_indices.shape)
# # 			select_indices=np.logical_and(select_indices,templabels[:,k]==1)

# # 		success=np.count_nonzero(select_indices)/select_indices.shape[0]

# # 		select_indices=np.where(select_indices==True)[0]
# # 		#templabels[:,mytargetlabels]=(1 - target_on_off)	

		
# # 		cleanconsistentcount=0
# # 		local=0
# # 		glo=0
# # 		for c in select_indices:#tqdm(select_indices):
# # 			current_label=np.copy(adv[c,:])
# # 			targetlabels=mytargetlabels

# # 			_,clean_g=check_local_global_consistency(np.copy(clean[c,:]),k,tree.shape[0],tree)
# # 			if clean_g is True:
# # 				cleanconsistentcount+=1

# # 			l=True 
# # 			for k in targetlabels:
# # 				templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
# # 				l=l and templ
			
# # 			#print('G:',g)
# # 			if l is True:
# # 				local+=1
# # 			if g is True:
# # 				glo+=1


# # 		print('Total:',templabels.shape[0],', Success:',len(select_indices)/templabels.shape[0],', Clean consistent:',cleanconsistentcount,', Local:',local/templabels.shape[0],', Global :',glo/templabels.shape[0])
		
# # 		successpercent.append(len(select_indices)/templabels.shape[0])
# # 		consistentpercent.append(local/templabels.shape[0])
		
# # 		#print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])
# # 		# print('Global :',glo/templabels.shape[0],', Local Consistent:',local/templabels.shape[0])


		
# # 	meansuc=np.mean(np.array(successpercent))
# # 	meanconsistent=np.mean(np.array(consistentpercent))
# # 	print('target size:',target_set_size,', Mean success:',meansuc,', Consistent:',meanconsistent)
# # 	allsuccesses.append(meansuc)
# # 	allconsistent.append(meanconsistent)

# # print('Success:',allsuccesses)
# # print('Consistent:',allconsistent)

# # print(attackname)
# # 0/0

# #lvis

# meansuc=[[1.0, 0.9714285714285715, 0.6688571428571428, 0.2874285714285715, 0.08571428571428569],[0.0,0.0,0.0,0.0,0.0],[0.9962857142857143, 0.8545714285714286, 0.2637142857142857, 0.02114285714285714, 0.005142857142857143],[0.9417142857142857, 0.44799999999999995, 0.03342857142857143, 0.0005714285714285714, 0.0]]
# meanconsistent=[[0.002857142857142857, 0.0002857142857142857, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.0, 0.0, 0.0, 0.0, 0.0],[0.9417142857142857, 0.44799999999999995, 0.03342857142857143, 0.0005714285714285714, 0.0]]

# target_on_off=0
# target_set_size=/10 
# tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/lvis/tree','rb'))
# #num_classes=tree.shape[0]

# norm=float(np.inf)
# eps_val=0.004
# attackname='pgd_consistent_mlabeta_linf'

# allsuccesses=[]
# allconsistent=[]

# for target_set_size in [3]:
# 	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/nusconsistent/',str(target_set_size)),'rb'))

# 	consistentpercent=[]
# 	successpercent=[]
# 	for current_target_labels in tqdm(alltargetlabels):
# 		current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/lvis/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		
# 		clean=pd.read_hdf(os.path.join(current_store_folder,'clean.h5'),key='df').iloc[:,:-1].to_numpy()
# 		adv=pd.read_hdf(os.path.join(current_store_folder,'adv.h5'),key='df').iloc[:,:-1].to_numpy()

# 		#adv=/mnt/raptor/hassan/data/lvis/'+str(cat)+'labels.h5
# 		# adv=pd.read_hdf('/mnt/raptor/hassan/data/lvis/vallabels.h5',key='df')

# 		allimgids=adv.iloc[:,-1].to_numpy().tolist()
# 		finalimgids=[]
# 		finalindices=[]

# 		adv=adv.iloc[:,:-1].to_numpy()
# 		# tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/lvis/tree','rb'))

# 		clean=np.where(clean>0,1,0)
# 		adv=np.where(adv>0,1,0)
# 		mytargetlabels=current_target_labels
		
# 		templabels=np.copy(clean)
		
# 		#templabels[:,mytargetlabels]=(1 - target_on_off)	

# 		# cleanconsistentcount=0

# 		# for c in tqdm(range(adv.shape[0])):
# 		# 	current_label=np.zeros((601,),dtype=np.int32)
# 		# 	current_label[destindices]=np.copy(templabels[c,:][origindices])
# 		# 	_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
# 		# 	#print('G:',g)
# 		# 	if g is True:
# 		# 		cleanconsistentcount+=1

# 		# print(templabels.shape,cleanconsistentcount)


# 		templabels=np.copy(adv)
		
# 		select_indices=np.ones((adv.shape[0],),bool)
# 		#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
# 		for k in mytargetlabels:
# 			#print('Select indices:',k,select_indices.shape)
# 			select_indices=np.logical_and(select_indices,templabels[:,k]==1)

# 		success=np.count_nonzero(select_indices)/select_indices.shape[0]

# 		select_indices=np.where(select_indices==True)[0]
# 		#templabels[:,mytargetlabels]=(1 - target_on_off)	

# 		select_indices=list(range(adv.shape[0]))

# 		cleanconsistentcount=0
# 		local=0
# 		glo=0
# 		for c in tqdm(select_indices):
# 			current_label=templabels[c,:]
# 			targetlabels=mytargetlabels

# 			# l=True 
# 			# for k in targetlabels:
# 			# 	templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
# 			# 	l=l and templ
# 			l,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)

			
# 			#print('G:',g)
# 			if l is True:
# 				local+=1
# 			if g is True:
# 				finalindices.append(c)
# 				finalimgids.append(allimgids[c])
# 				glo+=1

# 			if (c+1)%500==0:
# 				print('Total:',len(finalindices),len(finalimgids),', Local:',local/templabels.shape[0],', Global :',glo/templabels.shape[0])
# 				tempdata=pd.DataFrame(adv[np.array(finalindices),:])
# 				tempdata['ids']=finalimgids
# 				tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/consistentvallabels.h5',key='df',mode='w')


# 		print('Total:',len(finalindices),len(finalimgids),', Local:',local/templabels.shape[0],', Global :',glo/templabels.shape[0])
# 		tempdata=pd.DataFrame(adv[np.array(finalindices),:])
# 		tempdata['ids']=finalimgids
# 		tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/consistentvallabels.h5',key='df',mode='w')

# 		successpercent.append(len(select_indices)/templabels.shape[0])
# 		consistentpercent.append(local/templabels.shape[0])
# 		#print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])
# 		# print('Global :',glo/templabels.shape[0],', Local Consistent:',local/templabels.shape[0])


# 		0/0

# 	meansuc=np.mean(np.array(successpercent))
# 	meanconsistent=np.mean(np.array(consistentpercent))
# 	print('target size:',target_set_size,', Mean success:',meansuc,', Consistent:',meanconsistent)
# 	allsuccesses.append(meansuc)
# 	allconsistent.append(meanconsistent)

# print('Success:',allsuccesses)
# print('Consistent:',allconsistent)

# 0/0


target_on_off=1
target_set_size=1
tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))
#num_classes=tree.shape[0]
mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))
mapping_inverse = {v: k for k, v in mapping.items()}
origindices=list(mapping.keys())
destindices=list(mapping.values())

norm='inf'
eps_val=0.05


attacktype='mlabeta'
attackname='pgd_consistent_'+attacktype+'_linf_ON'
allsuccesses=[]
allconsistent=[]
allsuccessesstd=[]
allconsistentstd=[]
for target_set_size in range(1,6):

	alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/oiconsistentON/','new'+str(target_set_size)),'rb'))
	# alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/csloiconsistentON/','all'+str(target_set_size)),'rb'))['target_classes']
	# print(alltargetlabels)
	consistentpercent=[]
	successpercent=[]
	for current_target_labels in tqdm(alltargetlabels):
		######################
		#/mnt/raptor/hassan/bmvc/stores/oiBetaOFF_graphonly/

		######################
		# current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/stores/oi/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		#/mnt/raptor/hassan/aaai24/oi/OFFBetaOrthThresh/
		# current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/oi/All/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]),attackname)
		# current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/oi/OFFBetaOrthThresh/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]),attackname)
		# current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/supp/cs/All',str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]),attackname)
		current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/supp/asloi/All/0.05/',str(target_set_size),attacktype,'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		# current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/oi/OFFGMLABeta/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]),attackname)
		#current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/oi/OFFGMLAAlpha/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		#current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/oi/OFFGMLAAlpha/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		#current_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/oi/OFFGMLAAlpha/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(i) for i in current_target_labels]),attackname)#'pgd_consistent_mlaalpha_linf')
		if not os.path.exists(current_store_folder):
			print(current_store_folder)
			continue
		if len(os.listdir(current_store_folder))==0:
			print(current_store_folder)
			continue
		clean=pd.read_hdf(os.path.join(current_store_folder,'clean.h5'),key='df').iloc[:,:-1].to_numpy()
		adv=pd.read_hdf(os.path.join(current_store_folder,'adv.h5'),key='df').iloc[:,:-1].to_numpy()
		clean=np.where(clean>0,1,0)
		adv=np.where(adv>0,1,0)
		mytargetlabels=current_target_labels
		
		templabels=np.copy(clean)
		
		#templabels[:,mytargetlabels]=(1 - target_on_off)	

		# cleanconsistentcount=0

		# for c in tqdm(range(adv.shape[0])):
		# 	current_label=np.zeros((601,),dtype=np.int32)
		# 	current_label[destindices]=np.copy(templabels[c,:][origindices])
		# 	_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
		# 	#print('G:',g)
		# 	if g is True:
		# 		cleanconsistentcount+=1

		# print(templabels.shape,cleanconsistentcount)


		templabels=np.copy(adv)
		
		select_indices=np.ones((adv.shape[0],),bool)
		#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
		for k in mytargetlabels:
			#print('Select indices:',k,select_indices.shape)
			select_indices=np.logical_and(select_indices,templabels[:,k]==0)

		success=np.count_nonzero(select_indices)/select_indices.shape[0]

		select_indices=np.where(select_indices==True)[0]
		#templabels[:,mytargetlabels]=(1 - target_on_off)	

		cleanconsistentcount=0
		local=0
		for c in select_indices:
			current_label=np.zeros((601,),dtype=np.int32)
			current_label[destindices]=np.copy(templabels[c,:][origindices])
			
			targetlabels=[mapping[t] for t in mytargetlabels if t in mapping.keys()]

			l=True 
			for k in targetlabels:
				templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
				l=l and templ

			
			#print('G:',g)
			if l is True:
				local+=1

		successpercent.append(len(select_indices)/templabels.shape[0])
		consistentpercent.append(local/templabels.shape[0])

		#print('Success:',len(select_indices)/templabels.shape[0],', Consistent:',local/templabels.shape[0])

	meansuc=np.mean(np.array(successpercent))
	meanconsistent=np.mean(np.array(consistentpercent))
	print('target size:',target_set_size,', Mean success:',meansuc,', Consistent:',meanconsistent)
	allsuccesses.append(meansuc)
	allconsistent.append(meanconsistent)
	allsuccessesstd.append(np.std(np.array(successpercent)))
	allconsistentstd.append(np.std(np.array(consistentpercent)))
	

print('Success:',allsuccesses)
print(allsuccessesstd)
print('Consistent:',allconsistent)
print(allconsistentstd)
0/0



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



