# import numpy as np 
# import os 
# import sys
# from tqdm import tqdm 
# import pandas as pd 
# from PIL import Image
# from torchvision import transforms
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# import torch 
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore")

# mytransform=transforms.Compose([
# transforms.Resize(224),
# transforms.CenterCrop(224),
# transforms.ToTensor(),
# ])
# ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

# datasetname='voc'
# if datasetname=='voc':
# 	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/tree','rb'))
# 	#maindir='/mnt/raptor/hassan/aaai24/voc/OFFBetaOrthThresh/'
# 	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
# 	imagespath='/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/JPEGImages/'
# 	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/vocconsistentON/'
# 	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']
# if datasetname =='nus':
# 	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/nus/tree','rb'))
# 	# maindir='/mnt/raptor/hassan/aaai24/nus/OFFBetaOrthThresh/'
# 	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
# 	# maindir='/mnt/raptor/hassan/aaai24/supp/nus/All/'
# 	# maindir='/mnt/raptor/hassan/cvpr24/supp/nus/All/'
# 	imagespath='/mnt/raptor/hassan/data/nus/'
# 	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/nusconsistentON/'
# 	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']


# eps_val=0.01
# for model_name in ['mldecoder']:
# 	for attacktype in ['mlabeta']:#['mlaalpha','mlabeta','mllp','gmla']:
# 		for target_set_size in [1,2]:
# 			alltargetlabels=pickle.load(open(os.path.join(targetclasspath,'all'+str(target_set_size)),'rb'))['target_classes']
			
# 			successpercent=[]
# 			consistentpercent=[]
# 			ntrpercent=[]
# 			ssim_vals=[]
# 			for current_target_labels in alltargetlabels:
# 				pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/attack/voc/',str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
# 				pert_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/perts/voc/',str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))


# 				clean=pd.read_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df')
# 				adv=pd.read_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df')
# 				imageids=adv.iloc[:,-1].to_numpy().tolist()
# 				clean=np.where(clean.iloc[:,:-1].to_numpy()>0,1,0)
# 				adv=np.where(adv.iloc[:,:-1].to_numpy()>0,1,0)

# 				templabels=np.copy(adv)
# 				cleantemplabels=np.copy(clean)
		
# 				select_indices=np.ones_like(adv[:,0])
				

# 				for k in current_target_labels:
# 					select_indices=np.logical_and(select_indices==1.0,templabels[:,k]==0).astype(np.float32)

# 				success=np.count_nonzero(select_indices==1.0)/select_indices.shape[0]

# 				select_indices=np.where(select_indices==True)[0]
				
# 				############################################
# 				# Get SSIM
# 				for c in select_indices:
# 					orig_image=mytransform(Image.open(os.path.join(imagespath,imageids[c]))).unsqueeze(dim=0)
# 					perturbation=torch.from_numpy(np.load(os.path.join(pert_store_folder,imageids[c]+'.npy')))
# 					adv_image=torch.clip(orig_image+perturbation,0.0,1.0)
# 					ssim_vals.append(ssim(adv_image,orig_image))
# 				# ssim_vals.append(0.0)
					

				
# 				successpercent.append(success)
				


# 			dictvals={
# 			'Model':model_name,
# 			'Attack':attacktype,
# 			'Target':target_set_size,
# 			'Success':"{:.3f}".format(np.mean(np.array(successpercent))),
# 			'SSIM':"{:.3f}".format(np.mean(np.array(ssim_vals)))
# 			}
# 			lossst='\n'+', '.join([x+': '+str(dictvals[x]) for x in dictvals.keys()])

# 			print(lossst)

# import numpy as np 
# import os 
# import sys
# from tqdm import tqdm 
# import pandas as pd 
# from PIL import Image
# from torchvision import transforms
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# import torch 
# import warnings
# import pickle
# import sys
# sys.path.append('./')
# from scripts.tree_loss import *


# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore")

# mytransform=transforms.Compose([
# transforms.Resize(224),
# transforms.CenterCrop(224),
# transforms.ToTensor(),
# ])
# ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

# datasetname='voc'
# if datasetname=='voc':
# 	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/tree','rb'))
# 	#maindir='/mnt/raptor/hassan/aaai24/voc/OFFBetaOrthThresh/'
# 	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
# 	imagespath='/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/JPEGImages/'
# 	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/vocconsistentON/'
# 	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']
# if datasetname =='nus':
# 	tree=pickle.load(open('/mnt/raptor/hassan/meta/MLL/KG_data/nus/tree','rb'))
# 	# maindir='/mnt/raptor/hassan/aaai24/nus/OFFBetaOrthThresh/'
# 	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
# 	# maindir='/mnt/raptor/hassan/aaai24/supp/nus/All/'
# 	# maindir='/mnt/raptor/hassan/cvpr24/supp/nus/All/'
# 	imagespath='/mnt/raptor/hassan/data/nus/'
# 	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/nusconsistentON/'
# 	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']


# eps_val=0.01
# target_set_size=1
# for model_name in ['gcn','asl','mldecoder']:
# 	for source_model in ['gcn','asl','mldecoder']:
# 		if model_name==source_model:
# 			continue
# 		for attacktype in ['mlaalpha','mlabeta','mllp','gmla']:#['mlaalpha','mlabeta','mllp','gmla']:
# 		# for target_set_size in [1,2]:
# 			alltargetlabels=pickle.load(open(os.path.join(targetclasspath,'all'+str(target_set_size)),'rb'))['target_classes']
			
# 			successpercent=[]
# 			consistentpercent=[]
# 			ntrpercent=[]
# 			ssim_vals=[]
# 			for current_target_labels in alltargetlabels:
# 				# pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/attack/'+datasetname,str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
# 				pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/attack/'+datasetname,source_model+'_'+str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
# 				pert_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/perts/'+datasetname,str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))

# 				if not os.path.exists(os.path.join(pgd_store_folder,'clean.h5')):
# 					continue
# 				clean=pd.read_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df')
# 				adv=pd.read_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df')
# 				imageids=adv.iloc[:,-1].to_numpy().tolist()
# 				clean=np.where(clean.iloc[:,:-1].to_numpy()>0,1,0)
				
# 				adv=np.where(adv.iloc[:,:-1].to_numpy()>0,1,0)

# 				templabels=np.copy(adv)
# 				cleantemplabels=np.copy(clean)
		
# 				select_indices=np.ones_like(adv[:,0])
				

# 				for k in current_target_labels:
# 					select_indices=np.logical_and(select_indices==1.0,templabels[:,k]==0).astype(np.float32)

# 				success=np.count_nonzero(select_indices==1.0)/select_indices.shape[0]
# 				successpercent.append(success)

# 				select_indices=np.where(select_indices==True)[0]
				
# 				# ############################################
# 				# # Get SSIM
# 				# for c in select_indices:
# 				# 	orig_image=mytransform(Image.open(os.path.join(imagespath,imageids[c]))).unsqueeze(dim=0)
# 				# 	perturbation=torch.from_numpy(np.load(os.path.join(pert_store_folder,imageids[c].replace('images/','')+'.npy')))
# 				# 	adv_image=torch.clip(orig_image+perturbation,0.0,1.0)
# 				# 	ssim_vals.append(ssim(adv_image,orig_image))
# 				# # ssim_vals.append(0.0)
				
# 				# ############################################
# 				# Compute NTR
# 				ntr_selection_mask=np.zeros_like(adv)
# 				for c in select_indices:
# 					current_label=np.copy(cleantemplabels[c,:])
# 					treenodes=compute_tree_off_loss(np.copy(current_label),current_target_labels,tree)
# 					ntr_selection_mask[c,treenodes]=1.0
# 					# totaltargetcount+=len(treenodes)

# 				ntr_selection_mask=1-ntr_selection_mask

# 				changed=(clean!=adv).astype(np.float32)
# 				changed=changed*(ntr_selection_mask)
# 				tempval=np.sum(changed,axis=1)/np.sum(ntr_selection_mask,axis=1)
# 				ntrpercent.append(np.mean(tempval))

				
# 				############################################
# 				# Compute consistent

# 				cleanconsistentcount=0
# 				local=0
# 				for c in select_indices:
				
# 					current_label=np.copy(adv[c,:])
			
# 					l=True

# 					for k in current_target_labels:
# 						templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
# 						l=l and templ

# 					#print('G:',g)
# 					if l is True:
# 						local+=1

# 				consistentpercent.append(local/templabels.shape[0])
				
# # 				ntrpercent.append((changed[select_indices].sum()-target_set_size*len(select_indices))/(clean.shape[-1]*len(select_indices)-target_set_size*len(select_indices)))

# 				# ############################################

			
# 			dictvals={
# 			'Source Model':source_model,
# 			'Model':model_name,
# 			'Attack':attacktype,
# 			'Target':target_set_size,
# 			'Success':"{:.3f}".format(np.mean(np.array(successpercent)))+str('+')+"{:.3f}".format(np.std(np.array(successpercent))),
# 			'Consistent':"{:.3f}".format(np.mean(np.array(consistentpercent)))+str('+')+"{:.3f}".format(np.std(np.array(consistentpercent))),
# 			'NTR':"{:.3f}".format(np.mean(np.array(ntrpercent)))+str('+')+"{:.3f}".format(np.std(np.array(ntrpercent))),
# 			# 'SSIM':"{:.3f}".format(np.mean(np.array(ssim_vals)))
# 			}
# 			lossst='\n'+', '.join([x+': '+str(dictvals[x]) for x in dictvals.keys()])

# 			print(lossst)

import numpy as np 
import os 
import sys
from tqdm import tqdm 
import pandas as pd 
from PIL import Image
from torchvision import transforms
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch 
import warnings
import pickle
import sys
sys.path.append('./')
from scripts.tree_loss import *


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")

mytransform=transforms.Compose([
transforms.Resize(224),
transforms.CenterCrop(224),
transforms.ToTensor(),
])
ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

datasetname='voc'
if datasetname=='voc':
	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/tree','rb'))
	#maindir='/mnt/raptor/hassan/aaai24/voc/OFFBetaOrthThresh/'
	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
	imagespath='/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/JPEGImages/'
	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/vocconsistentON/'
	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']
if datasetname =='nus':
	tree=pickle.load(open('/mnt/raptor/hassan/meta/MLL/KG_data/nus/tree','rb'))
	# maindir='/mnt/raptor/hassan/aaai24/nus/OFFBetaOrthThresh/'
	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
	# maindir='/mnt/raptor/hassan/aaai24/supp/nus/All/'
	# maindir='/mnt/raptor/hassan/cvpr24/supp/nus/All/'
	imagespath='/mnt/raptor/hassan/data/nus/'
	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/nusconsistentON/'
	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']




eps_val=0.01

for model_name in ['gcn','asl','mldecoder']:
	#for attacktype in ['mlaalpha','mlabeta','gmla_equal','mllp']:
	for attacktype in ['mlaalpha','mlabeta','mllp','gmla_equal']:
		
		for target_set_size in [3]:#['mlaalpha','mlabeta','mllp','gmla']:
		# for target_set_size in [1,2]:
			alltargetlabels=pickle.load(open(os.path.join(targetclasspath,'all'+str(target_set_size)),'rb'))['target_classes']
			
			successpercent=[]
			consistentpercent=[]
			ntrpercent=[]
			ssim_vals=[]
			for current_target_labels in alltargetlabels:
				# pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/attack/'+datasetname,str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
				pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/attack/'+datasetname,str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
				pert_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/perts/'+datasetname,str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))

				if not os.path.exists(os.path.join(pgd_store_folder,'clean.h5')):
					continue
				clean=pd.read_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df')
				adv=pd.read_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df')
				imageids=adv.iloc[:,-1].to_numpy().tolist()
				clean=np.where(clean.iloc[:,:-1].to_numpy()>0,1,0)
				
				adv=np.where(adv.iloc[:,:-1].to_numpy()>0,1,0)

				templabels=np.copy(adv)
				cleantemplabels=np.copy(clean)
		
				select_indices=np.ones_like(adv[:,0])
				

				for k in current_target_labels:
					select_indices=np.logical_and(select_indices==1.0,templabels[:,k]==0).astype(np.float32)

				success=np.count_nonzero(select_indices==1.0)/select_indices.shape[0]
				successpercent.append(success)

				select_indices=np.where(select_indices==True)[0]
				
				# ############################################
				# # Get SSIM
				# for c in select_indices:
				# 	orig_image=mytransform(Image.open(os.path.join(imagespath,imageids[c]))).unsqueeze(dim=0)
				# 	perturbation=torch.from_numpy(np.load(os.path.join(pert_store_folder,imageids[c].replace('images/','')+'.npy')))
				# 	adv_image=torch.clip(orig_image+perturbation,0.0,1.0)
				# 	ssim_vals.append(ssim(adv_image,orig_image))
				# # ssim_vals.append(0.0)
				
				# ############################################
				# Compute NTR
				ntr_selection_mask=np.zeros_like(adv)
				for c in select_indices:
					current_label=np.copy(cleantemplabels[c,:])
					treenodes=compute_tree_off_loss(np.copy(current_label),current_target_labels,tree)
					ntr_selection_mask[c,treenodes]=1.0
					# totaltargetcount+=len(treenodes)

				ntr_selection_mask=1-ntr_selection_mask

				changed=(clean!=adv).astype(np.float32)
				changed=changed*(ntr_selection_mask)
				tempval=np.sum(changed,axis=1)/np.sum(ntr_selection_mask,axis=1)
				ntrpercent.append(np.mean(tempval))

				
				############################################
				# Compute consistent

				cleanconsistentcount=0
				local=0
				for c in select_indices:
				
					current_label=np.copy(adv[c,:])
			
					l=True

					for k in current_target_labels:
						templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
						l=l and templ

					#print('G:',g)
					if l is True:
						local+=1

				consistentpercent.append(local/templabels.shape[0])
				
# 				ntrpercent.append((changed[select_indices].sum()-target_set_size*len(select_indices))/(clean.shape[-1]*len(select_indices)-target_set_size*len(select_indices)))

				# ############################################

			
			dictvals={
			
			'Model':model_name,
			'Attack':attacktype,
			'Target':target_set_size,
			'Success':"{:.3f}".format(np.mean(np.array(successpercent)))+str('+')+"{:.3f}".format(np.std(np.array(successpercent))),
			'Consistent':"{:.3f}".format(np.mean(np.array(consistentpercent)))+str('+')+"{:.3f}".format(np.std(np.array(consistentpercent))),
			'NTR':"{:.3f}".format(np.mean(np.array(ntrpercent)))+str('+')+"{:.3f}".format(np.std(np.array(ntrpercent))),
			# 'SSIM':"{:.3f}".format(np.mean(np.array(ssim_vals)))
			}
			lossst='\n'+', '.join([x+': '+str(dictvals[x]) for x in dictvals.keys()])

			print(lossst)








# import numpy as np 
# import os 
# import sys
# sys.path.append('./')
# from scripts.tree_loss import *
# from tqdm import tqdm 
# import pandas as pd 
# from PIL import Image
# from torchvision import transforms
# from torchmetrics.image import StructuralSimilarityIndexMeasure
# import torch 
# import warnings

# warnings.filterwarnings("ignore", category=DeprecationWarning)
# warnings.filterwarnings("ignore")

# mytransform=transforms.Compose([
# transforms.Resize(224),
# transforms.CenterCrop(224),
# transforms.ToTensor(),
# ])
# ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

# datasetname='voc'
# if datasetname=='voc':
# 	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/tree','rb'))
# 	#maindir='/mnt/raptor/hassan/aaai24/voc/OFFBetaOrthThresh/'
# 	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
# 	imagespath='/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/JPEGImages/'
# 	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/vocconsistentON/'
# 	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']
# if datasetname =='nus':
# 	tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/nus/tree','rb'))
# 	# maindir='/mnt/raptor/hassan/aaai24/nus/OFFBetaOrthThresh/'
# 	#maindir='/mnt/raptor/hassan/aaai24/supp/voc/ALL/'
# 	# maindir='/mnt/raptor/hassan/aaai24/supp/nus/All/'
# 	# maindir='/mnt/raptor/hassan/cvpr24/supp/nus/All/'
# 	imagespath='/mnt/raptor/hassan/data/nus/'
# 	targetclasspath='/mnt/raptor/hassan/aaai24/sequences/nusconsistentON/'
# 	# alltargetlabels=pickle.load(open(os.path.join(,'all'+str(target_set_size)),'rb'))['target_classes']


# eps_val=0.01
# for model_name in ['mldecoder']:
# 	for attacktype in ['mlabeta']:#['mlaalpha','mlabeta','mllp','gmla']:
# 		for target_set_size in [1,2]:
# 			alltargetlabels=pickle.load(open(os.path.join(targetclasspath,'all'+str(target_set_size)),'rb'))['target_classes']
			
# 			successpercent=[]
# 			consistentpercent=[]
# 			ntrpercent=[]
# 			ssim_vals=[]
# 			for current_target_labels in alltargetlabels:
# 				pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/attack/voc/',str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
# 				pert_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/perts/voc/',str(model_name),str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))


# 				clean=pd.read_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df')
# 				adv=pd.read_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df')
# 				imageids=adv.iloc[:,-1].to_numpy().tolist()
# 				clean=np.where(clean.iloc[:,:-1].to_numpy()>0,1,0)
# 				adv=np.where(adv.iloc[:,:-1].to_numpy()>0,1,0)

# 				templabels=np.copy(adv)
# 				cleantemplabels=np.copy(clean)
		
# 				select_indices=np.ones_like(adv[:,0])
				

# 				for k in current_target_labels:
# 					select_indices=np.logical_and(select_indices==1.0,templabels[:,k]==0).astype(np.float32)

# 				success=np.count_nonzero(select_indices==1.0)/select_indices.shape[0]

# 				select_indices=np.where(select_indices==True)[0]
				
# 				############################################
# 				# Get SSIM
# 				for c in select_indices:
# 					orig_image=mytransform(Image.open(os.path.join(imagespath,imageids[c]))).unsqueeze(dim=0)
# 					perturbation=torch.from_numpy(np.load(os.path.join(pert_store_folder,imageids[c]+'.npy')))
# 					adv_image=torch.clip(orig_image+perturbation,0.0,1.0)
# 					ssim_vals.append(ssim(adv_image,orig_image))
# 				# ssim_vals.append(0.0)
					

# 				############################################
# 				# Compute NTR
# 				changed=(clean!=adv).astype(np.float32)
# 				ntrpercent.append((changed[select_indices].sum()-target_set_size*len(select_indices))/(clean.shape[-1]*len(select_indices)-target_set_size*len(select_indices)))



# 				############################################
			
# 				cleanconsistentcount=0
# 				local=0
# 				for c in select_indices:
				
# 					current_label=np.copy(adv[c,:])
			
# 					l=True

# 					for k in current_target_labels:
# 						templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
# 						l=l and templ

# 					#print('G:',g)
# 					if l is True:
# 						local+=1

# 				successpercent.append(success)
# 				consistentpercent.append(local/templabels.shape[0])


# 			dictvals={
# 			'Model':model_name,
# 			'Attack':attacktype,
# 			'Target':target_set_size,
# 			'Success':"{:.3f}".format(np.mean(np.array(successpercent))),
# 			'Consistent':"{:.3f}".format(np.mean(np.array(consistentpercent))),
# 			'NTR':"{:.3f}".format(np.mean(np.array(ntrpercent))),
# 			'SSIM':"{:.3f}".format(np.mean(np.array(ssim_vals)))
# 			}
# 			lossst='\n'+', '.join([x+': '+str(dictvals[x]) for x in dictvals.keys()])

# 			print(lossst)










