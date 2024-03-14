import os
import sys
sys.path.append('./')
from scripts.tree_loss import *
from utils.utility import *
from utils.confparser import *
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
#from torch.autograd.functional import jacobian
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy
from tqdm import tqdm 
from Models import *
#from Datasets.NUSDataset import NUSImageDataset as ImageDataset
from Datasets import * 
import re
from Logger.Logger import *
import configparser
import ast, json
import argparse
import pickle
from datetime import datetime
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from Attacks.apgd import *
from Attacks.mll import * 
from Attacks.projected_gradient_descent import *
import random 
import gc 
import time
import itertools 
from scripts.tree_loss import *
sys.path.append('ASL/')
#from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
import numpy as np
import os 


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')

seed=999
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# np.random.seed(seed)  # Numpy module.
# random.seed(seed)  # Python random module.
# torch.manual_seed(seed)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


class Tester:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'test'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	#[25, 25, 22, 25, 27, 22, 25, 35, 37, 33, 41]
	#[25, 25, 22, 25, 27, 21, 24, 35, 39, 37, 46]
	#[25, 25, 22, 25, 27, 23, 25, 36, 40, 37, 43]
	#[25, 25, 22, 24, 27, 20, 25, 35, 38, 35, 42]
	#[25, 25, 22, 24, 27, 20, 25, 36, 40, 34, 43]
	#[25, 25, 22, 24, 27, 22, 25, 33, 41, 36, 42]

	def compute_apgd(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, criterion,allotherdata,selection_mask_target):
		apgdt = APGDAttack(model, eps=0.05, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		with torch.no_grad():
			return apgdt.perturb(inputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,use_target_indices=True,selection_mask=selection_mask_target)


	def test(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		combine_uaps=params['combine_uaps']
		class_names=np.array(params['class_names'])
		device=params['device']
		
		test_dataset=params['test_dataset']
		test_dataloader=params['test_dataloader']
		
		
		writer=params['writer']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		p_norm=params['p_norm']


		####################
		args={'image_size':224,'model_name':'tresnet_l',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		#'model_path':'/mnt/raptor/hassan/weights/nus/sep/nus_sep_asl_1/model-81.pt',
		'sep_features':0,
		#'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		'model_path':'/mnt/raptor/hassan/weights/lvis/asl/model-52.pt',
		'num_classes':1203,'workers':0
		}

		args=argparse.Namespace(**args)
		state = torch.load(args.model_path, map_location='cpu')
		args.num_classes=1203
		args.do_bottleneck_head = False
		model = create_model(args).cuda()
		model.load_state_dict(state['model_state'], strict=True)
		
		
		model=model.cuda()
		model.apply(self.set_bn_eval)

		for module in model.modules():
			#print(module)
			module.eval()

		current_target_value=0.0

		eps_val=0.004#2.5#0.5#0.005
		
		norm=float(np.inf)
		target_set_size=3
		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')
		dataset_name='lvis'
		#tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/lvis/tree','rb'))
		tree=pickle.load(open(os.path.join('/mnt/raptor/hassan/data/KG_data/',str(dataset_name),'tree'),'rb'))
		
		alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/'+str(dataset_name)+'consistent/',str(target_set_size)),'rb'))
		for current_target_labels in alltargetlabels:
			print('Current target label:',current_target_labels)
			max_batch=70
			max_num_batches=1
			allset={}
			target_on_off=0
			#print(current_target_labels,target_on_off)
			
			print('Starting target labels:',current_target_labels, ', Goal:',('Turn ON','Turn OFF')[target_on_off])
			#current_store_folder=os.path.join('/mnt/raptor/hassan/MLLFinal/stores/',str(target_on_off),'_'.join([str(i) for i in current_target_labels]))
			current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/',dataset_name,'norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]))

			orth_store_folder=os.path.join(current_store_folder,'orth_linf','0.8')
			pgd_store_folder=os.path.join(current_store_folder,'pgd_consistent_mlagmla09_linf')
			pgd_fixed_store_folder=os.path.join(current_store_folder,'pgd_fixed_linf')
			# aorth_store_folder=os.path.join(current_store_folder,'aorth_l2')
			# apgd_store_folder=os.path.join(current_store_folder,'pgd_l2')
			print(pgd_store_folder)
			#create_folder(os.path.join(orth_store_folder,'pert'))
			#create_folder(os.path.join(pgd_store_folder,'pert'))
			#create_folder(os.path.join(pgd_fixed_store_folder,'pert'))
			# create_folder(orth_store_folder)
			create_folder(pgd_store_folder)
			
			batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
			orth_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
			#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
			#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
			pgd_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
			pgd_fixed_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
			clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

			batch_img_ids=[]
			batch_inputs=torch.empty((0,3,224,224),dtype=torch.float32).cuda()
			number_of_batch_iterations=0 

			storecleanoutputs=[]
			storeadvoutputs=[]
			storent=[]
			storefinalindices=[]
			storedots=[]
			storeimgids=[]


			for i, data in enumerate(tqdm(test_dataloader)):
				# if i==0:
				# 	continue
				# get the inputs; data is a list of [image_ids, inputs, labels]
				#img_ids,inputs, labels = data 
				if number_of_batch_iterations>=max_num_batches:
					break

				(all_inputs,img_ids),labels=data
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()					

				model.eval()
				
				with torch.no_grad():
					outputs,features = model(all_inputs,{'epsilon_norm':0.0})
					#print('outputs shape:',outputs.shape)
				
				labels=torch.clone(outputs).detach()
				labels=torch.where(labels>0,1,0).float()

				mytargetlabels=current_target_labels

				select_indices=torch.ones(size=(all_inputs.shape[0],)).type(torch.ByteTensor)
				#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
				for k in mytargetlabels:
					#print('Select indices:',k,select_indices.shape)
					select_indices=torch.logical_and(select_indices,labels[:,k].cpu()==target_on_off)

				temp_select_indices=torch.where(select_indices==True)[0]

				# from here, find the select_indices which are inconsistent after turning the labels ON/OFF
				templabels=torch.clone(labels).detach().cpu().numpy()
				
				templabels[:,mytargetlabels]=(1 - target_on_off)	

				select_indices=[]

				for c in temp_select_indices:
					current_label=templabels[c,:]
					_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
					#print('G:',g)
					if g is False:
						select_indices.append(c)
				# Ensure that now these are not consistent

				print('Total:',len(select_indices))
				
				if(len(select_indices)==0):
					continue 

				#0/0
				labels=labels[select_indices,:]
				inputs=all_inputs[select_indices,:]
				outputs=outputs[select_indices,:]
				img_ids=np.array(img_ids)[select_indices].tolist()
				
				#print(batch_labels.shape,labels.shape)
				batch_labels=torch.cat((batch_labels,labels),dim=0)
				clean_outputs=torch.cat((clean_outputs,outputs),dim=0)
				batch_inputs=torch.cat((batch_inputs,inputs),dim=0)

				if(len(inputs)==1):
					img_ids=[img_ids]

				batch_img_ids+=img_ids

				del inputs 
				if(batch_inputs.shape[0]<max_batch):
					continue

				del all_inputs 

				inputs=batch_inputs[:max_batch]
				labels=batch_labels[:max_batch]
				img_ids=batch_img_ids[:max_batch]
				clean_outputs=clean_outputs[:max_batch]
				
				for jk in range(1,2):
					to_use_selection_mask=True

					#newlabels=torch.clone(labels)
					newlabels=labels
					tempsucc=0
					flipped_labels=0

					model_params={
					'p_norm':p_norm,
					'epsilon_norm':epsilon_norm,
					'target_label':newlabels,
					'combine_uaps':combine_uaps,
					'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
					}

					selection_mask_target=torch.ones_like(newlabels,dtype=torch.float32)
					target_selection_mask=torch.zeros_like(newlabels,dtype=torch.float32)
					nontarget_selection_mask=torch.zeros_like(newlabels,dtype=torch.float32)

					# ###########################
					# templabels=torch.clone(labels).detach().cpu().numpy()
					# templabels=np.where(templabels<=0,0,1)
					
					# # for t in mytargetlabels:
					# # 	newlabels[:,t]=1-newlabels[:,t]
					
					# countbefore=0
					# countafter=0
					# for c in range(templabels.shape[0]):
					# 	#current_label=np.zeros((601,),dtype=np.int32)
					# 	#current_label[destindices]=np.copy(templabels[c,:][origindices])
					# 	current_label=np.copy(templabels[c,:])
					# 	# _,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)
					# 	# print('Before:',g)
					# 	# if g is True:
					# 	# 	countbefore+=1
					# 	# _,g=check_local_global_consistency(np.copy(templabels[c,:]),0,tree.shape[0],tree)
						
					# 	# treenodes=compute_tree_on_loss(np.copy(current_label),mytargetlabels,tree)
					# 	treenodes=compute_tree_on_loss(np.copy(current_label),mytargetlabels,tree)

					# 	#print(treenodes,mytargetlabels)
					# 	#print(class_names[treenodes],class_names[mytargetlabels])
					# 	#current_label=np.zeros_like(templabels[c,:])
					# 	current_label=np.copy(templabels[c,:])
					# 	current_label[treenodes]=1.0
					# 	current_label[mytargetlabels]=1.0
						
					# 	l=True 
					# 	for tl in mytargetlabels:
					# 		templ,g=check_local_global_consistency(np.copy(current_label),tl,tree.shape[0],tree)
					# 		l = l and templ
					# 	templ,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)

					# 	# print('G:',g)
					# 	# if g is True:
					# 	# 	countafter+=1

					# 	# origidx=list(set([mapping_inverse[t] for t in treenodes if t in mapping_inverse.keys()]+current_target_labels))
					# 	origidx=mytargetlabels+treenodes
						
					# 	target_selection_mask[c,origidx]=1.0
					# 	tempnontarget=list(set(list(range(num_classes))).difference(set(origidx)))
					# 	nontarget_selection_mask[c,tempnontarget]=1.0
					# 	# print(newlabels[c,origidx])
					# 	newlabels[c,origidx]=1#-newlabels[c,origidx]
					# 	#assert(torch.sum(newlabels[c,origidx])==len(origidx))
					# 	assert(l is True)


					# # # print(countbefore,countafter)
					# # # 0/0
					# ###########################

					nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))

					target_selection_mask[:,mytargetlabels]=1.0
					nontarget_selection_mask[:,nontargetlabels]=1.0
					newlabels[:,mytargetlabels]=1-newlabels[:,mytargetlabels]
					
					###########################

					beforestore=clean_outputs
					
					inputs.requires_grad=True

					
					scale_factor=0.0
					
					model.zero_grad()
					model.eval()
					#print('inputs:',inputs.shape,newlabels.shape,inputs.requires_grad)
					print('Scale:',scale_factor)
					currentinputs=torch.clone(inputs).detach()
					
					currentinputs.requires_grad=True

					allotherdata={'scale':scale_factor,
					'target_selection_mask':target_selection_mask,
					'nontarget_selection_mask':nontarget_selection_mask,
					}

					pert_image,otherdata=self.compute_apgd(model,currentinputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,selection_mask_target=selection_mask_target)
					
					#model,inputs,newlabels,criterion,selection_mask_target
					#best_e=orthmll.pgd_optimize_perturb(model,inputs,newlabels,mytargetlabels,nontargetlabels,eps_val,norm,orth_step_size ,criterion, out_iterations,in_iterations)
					#pert_image=inputs+best_e.cuda()
					#print('Min max:',torch.min(pert_image),torch.max(pert_image))
					
					#pert_image=pert_image.detach().cuda()
					

					with torch.no_grad():
						outputs,_ = model(pert_image,{'epsilon_norm':0.0})
						#afterstore=torch.clone(outputs)
						afterstore=outputs.detach()
						koutputs=torch.where(outputs>0,1,0).float()
					

					tempoutputs=torch.clone(koutputs).detach().cpu().numpy()
					select_indices=[]
					local,glo=0,0
					target_selection_mask=target_selection_mask.detach().cpu().numpy()
					for c in range(outputs.shape[0]):
						current_label=np.copy(tempoutputs[c,:])
						#current_label=np.zeros((601,),dtype=np.int32)
						#current_label[destindices]=np.copy(tempoutputs[c,:][origindices])
						#origtargetlabels=np.where(target_selection_mask[c,:]==1.0)[0]
						#targetlabels=[mapping[t] for t in origtargetlabels if t in mapping.keys()]

						l=True 
						for k in mytargetlabels:
							current_label=np.copy(tempoutputs[c,:])
							templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
							#print(l,templ)
							l = l and templ 
							
						#print('G:',g)
						if g is True:
							glo+=1
						if l is True:
							local+=1

					print('L,g:',local,glo)
					

					success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
					e=pert_image-currentinputs
					#print(torch.linalg.vector_norm(e.view(e.shape[0],-1),ord=2,dim=-1))
					#print('Norm '+str(norm)+':',torch.linalg.vector_norm(e.view(e.shape[0],-1),ord=norm,dim=-1))
					#print('Norm 1:',torch.linalg.vector_norm(torch.abs(e.view(e.shape[0],-1)),ord=1,dim=-1))
					#print('\nSuccess:',len(success_indices),torch.min(e).item(),torch.max(e).item())
					
					scores=(koutputs==newlabels).float()#.sum(1)
					#ac_scores=torch.count_nonzero(scores[:,mytargetlabels+nontargetlabels].sum(1)==len(mytargetlabels+nontargetlabels)).item()
					ac_scores=scores[:,nontargetlabels].sum(1)-len(nontargetlabels)
					#[ -50.,  -86.,   -30.,  -63.,  -30.,  -53., -105.,  -10., -112.,  -42.], #moving orth
					#[ -5., -12.,   0., -11.,  -4.,  -2., -48.,   0., -12., -11.] #moving orth + nontargetgrad
					#[ -62., -126.,   -109., -87.,  -84.,  -47., -127.,  -138., -93., -192.] # moving in target grad
					#[ -220., -1699., -292., -43.,  -36.,  -24.,  -83.,  -263., -50., -381.] # moving with svd orth
					#ac_scores=torch.count_nonzero(scores==koutputs.shape[1]).item()

					out=torch.abs(beforestore-afterstore)
					#print('PGD Fixed Success: %d, All Class success: %d'%(len(success_indices),ac_scores),', min: %.3f, max: %.3f, target sum: %.3f, nontarget sum: %.3f'%(torch.min(e).item(),torch.max(e).item(),torch.sum(out[:,mytargetlabels]).item(),torch.sum(out[:,nontargetlabels]).item()))
					print(ac_scores)
					print('PGD Fixed Success: %d, All Class success: %d'%(len(success_indices),1),', min: %.3f, max: %.3f, target sum: %.3f, nontarget sum: %.3f'%(torch.min(e).item(),torch.max(e).item(),torch.sum(out[:,mytargetlabels]).item(),torch.sum(out[:,nontargetlabels]).item()))
					#print('Scores:',scores)
					# pickle.dump(ac_scores,open('tempdata2/ntscores'+str(scale_factor)+'.npy','wb'))



					storecleanoutputs.append(torch.clone(clean_outputs).detach())
					storeadvoutputs.append(torch.clone(outputs).detach())
					storeimgids+=list(img_ids)
					storent+=ac_scores.cpu().tolist()
					storefinalindices+=otherdata['finalindices'].cpu().tolist()
					storedots+=otherdata['dots']

					
					

					# # #################################
					#continue 	
					del pert_image
					del inputs 
					del newlabels
					#del apgdt
					del outputs 
					del afterstore 
					# del orth_outputs
					# del aorth_outputs
					del orth_outputs
					del pgd_outputs

					del batch_inputs
					#del apgd_outputs
					del clean_outputs

					number_of_batch_iterations+=1
					
					batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
					orth_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
					
					pgd_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
					pgd_fixed_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

					#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
					#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
					clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

					batch_img_ids=[]
					batch_inputs=torch.empty((0,3,224,224),dtype=torch.float32).cuda()
					gc.collect()
					torch.cuda.empty_cache()
					
			
			#store
			0/0
			if len(storecleanoutputs)!=0:
				storecleanoutputs=torch.cat(storecleanoutputs,dim=0).detach().cpu().numpy()
				storeadvoutputs=torch.cat(storeadvoutputs,dim=0).detach().cpu().numpy()
				print('Store cleanoutputs:',storecleanoutputs.shape)
				print('Store advoutputs:',storeadvoutputs.shape)				

			tempdata=pd.DataFrame(storecleanoutputs)
			tempdata['ids']=storeimgids
			tempdata.to_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df',mode='w')
			

			
			tempdata=pd.DataFrame(storeadvoutputs)
			tempdata['ids']=storeimgids
			tempdata.to_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df',mode='w')
			



			pickle.dump(storent,open(os.path.join(pgd_store_folder,'storent'),'wb'))
			pickle.dump(storefinalindices,open(os.path.join(pgd_store_folder,'storefinalindices'),'wb'))
			pickle.dump(storedots,open(os.path.join(pgd_store_folder,'storedots'),'wb'))

			print(pgd_store_folder)
			

			
			




	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

