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
from Attacks.apgd import *
import random 
import time
import itertools 
from scripts.tree_loss import *
sys.path.append('ASL/')
#from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
from src.models import create_model
import numpy as np
import os 


pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/oi.ini')
parser.add_argument('expname')


seed=999
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
np.random.seed(seed)  # Numpy module.
random.seed(seed)  # Python random module.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True



class SemanticMLLAttack:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'test'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()


	def compute_apgd(self,model,eps_val,inputs,newlabels,mytargetlabels,nontargetlabels, criterion,allotherdata,selection_mask_target):
		apgdt = APGDAttack(model, eps=eps_val, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		with torch.no_grad():
			return apgdt.perturb(inputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,use_target_indices=True,selection_mask=selection_mask_target)


	def attack(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		tree_path=params['tree_path']
		
		device=params['device']
		
		test_dataset=params['test_dataset']
		test_dataloader=params['test_dataloader']
		hierarchy_mapping_path=params['hierarchy_mapping_path']
		target_labels_path=params['target_labels_path']
		
		writer=params['writer']
		epsilon_value=params['eps_value']
		p_norm=params['p_norm']

		
		################################################################################
		# Load the asymmetric loss model
		imgsize=448
		args={'image_size':imgsize,'model_name':'tresnet_l',
		'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		'num_classes':80,'workers':0
		}
		args=argparse.Namespace(**args)
		state = torch.load(args.model_path, map_location='cpu')
		args.num_classes = state['num_classes']
		args.do_bottleneck_head = True
		model = create_model(args).cuda()
		model.load_state_dict(state['model'], strict=True)
		################################################################################

		self.logger.write('Loss:',criterion)
		
		
		model=model.cuda()
		model.apply(self.set_bn_eval)

		for module in model.modules():
			#print(module)
			module.eval()

		tree=pickle.load(open(tree_path,'rb'))
		mapping=pickle.load(open(hierarchy_mapping_path,'rb'))
		mapping_inverse = {v: k for k, v in mapping.items()}
		origindices=list(mapping.keys())
		destindices=list(mapping.values())
		max_batch=10

		#apgdt = APGDAttack(model, eps=0.20, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		
		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

		
		for target_set_size in [1,2,3,4,5]:
			#current_target_value=1.0-current_target_value
			avg_meter=AverageMeter()
			
			attacktype=['mlaalpha','mlabeta','gmla','gmla_alpha','gmla_beta','gmla_orthonly'][1]
			alltargetlabels=pickle.load(open(os.path.join(target_labels_path,'new'+str(target_set_size)),'rb'))
			
			for current_target_labels in alltargetlabels:
				
				treetargetlabels=[mapping[c] for c in current_target_labels if c in mapping.keys()]

				print('Current target label:',current_target_labels,', eps:',eps_val)
				
				batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
				clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

				batch_img_ids=[]
				batch_inputs=torch.empty((0,3,imgsize,imgsize),dtype=torch.float32).cuda()


				storecleanoutputs=[]
				storeadvoutputs=[]
				storent=[]
				storefinalindices=[]
				storedots=[]
				storeimgids=[]

				
				for i, data in enumerate(tqdm(test_dataloader)):
					

					(all_inputs,img_ids),labels=data
					all_inputs=all_inputs.to(device)
					labels=labels.to(device).float()					

					model.eval()
					
					with torch.no_grad():
						outputs,features = model(all_inputs,{'epsilon_norm':0.0})
					
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
					
					mytargetlabels=current_target_labels

					select_indices=torch.ones(size=(all_inputs.shape[0],)).type(torch.ByteTensor)
					
					for k in mytargetlabels:
						select_indices=torch.logical_and(select_indices,labels[:,k].cpu()==1)

						#select_indices=torch.logical_and(select_indices,labels[:,12].cpu()==0)


					temp_select_indices=torch.where(select_indices==True)[0]

					# # from here, find the select_indices which are inconsistent after turning the labels ON/OFF
					# templabels=torch.clone(labels).detach().cpu().numpy()
					# print('Labels sum:',torch.sum(labels))
					# templabels[:,mytargetlabels]=(1 - target_on_off)	

					# select_indices=[]

					# for c in temp_select_indices:
					# 	current_label=np.zeros((601,),dtype=np.int32)
					# 	current_label[destindices]=np.copy(templabels[c,:][origindices])
					# 	current_label=np.where(current_label<=0,0,1)
					# 	_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
					# 	#print('G:',g)
					# 	if g is False:
					# 		select_indices.append(c)
					# # Ensure that now these are not consistent

					select_indices=temp_select_indices
					#print('Total:',len(select_indices))
					
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


						if 'gmla' in attacktype:
							templabels=torch.clone(labels).detach().cpu().numpy()
							templabels=np.where(templabels<=0,0,1)
							# for t in mytargetlabels:
							# 	newlabels[:,t]=1-newlabels[:,t]
							countbefore=0
							countafter=0
							for c in range(templabels.shape[0]):
								current_label=np.zeros((601,),dtype=np.int32)
								current_label[destindices]=np.copy(templabels[c,:][origindices])
								# _,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)
								# print('Before:',g)
								# if g is True:
								# 	countbefore+=1
								
								treenodes=compute_tree_off_loss(np.copy(current_label),treetargetlabels,tree)
								
								current_label=np.zeros((601,),dtype=np.int32)
								current_label[destindices]=templabels[c,:][origindices]
								current_label[treetargetlabels]=1.0
								current_label[treenodes]=1.0
								

								# _,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)

								# print('G:',g)
								# if g is True:
								# 	countafter+=1

								origidx=list(set([mapping_inverse[t] for t in treenodes if t in mapping_inverse.keys()]+current_target_labels))
								
								target_selection_mask[c,origidx]=1.0
								tempnontarget=list(set(list(range(num_classes))).difference(set(origidx)))
								nontarget_selection_mask[c,tempnontarget]=1.0
								newlabels[c,origidx]=1-newlabels[c,origidx]
								#assert(torch.sum(newlabels[c,origidx])==len(origidx))
								#assert(g is True)
							
							# print(countbefore,countafter)
							# 0/0

							nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))

						else:
							
							nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
							target_selection_mask[:,mytargetlabels]=1.0
							nontarget_selection_mask[:,nontargetlabels]=1.0
							newlabels[:,mytargetlabels]=1-newlabels[:,mytargetlabels]

						

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
						'attacktype':attacktype,
						}
						pert_image,otherdata=self.compute_apgd(model,eps_val,currentinputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,selection_mask_target=selection_mask_target)
						
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
							current_label=np.zeros((601,),dtype=np.int32)
							current_label[destindices]=np.copy(tempoutputs[c,:][origindices])
							origtargetlabels=np.where(target_selection_mask[c,:]==1.0)[0]
							targetlabels=[mapping[t] for t in origtargetlabels if t in mapping.keys()]

							l=True 
							g = False 
							for k in targetlabels:
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
						pickle.dump(ac_scores,open('tempdata2/ntscores'+str(scale_factor)+'.npy','wb'))



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
						
						

						del batch_inputs
						
						del clean_outputs


						batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
						
						
						
						

						#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
						#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
						clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

						batch_img_ids=[]
						batch_inputs=torch.empty((0,3,imgsize,imgsize),dtype=torch.float32).cuda()
						gc.collect()
						torch.cuda.empty_cache()
						
				
				
				#store
				storecleanoutputs=torch.cat(storecleanoutputs,dim=0)
				tempdata=pd.DataFrame(storecleanoutputs.detach().cpu().numpy())
				tempdata['ids']=storeimgids
				tempdata.to_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df',mode='w')
				print('Store cleanoutputs:',storecleanoutputs.shape,tempdata.shape)

				storeadvoutputs=torch.cat(storeadvoutputs,dim=0)
				tempdata=pd.DataFrame(storeadvoutputs.detach().cpu().numpy())
				tempdata['ids']=storeimgids
				tempdata.to_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df',mode='w')
				print('Store advoutputs:',storeadvoutputs.shape,tempdata.shape)				



				pickle.dump(storent,open(os.path.join(pgd_store_folder,'storent'),'wb'))
				pickle.dump(storefinalindices,open(os.path.join(pgd_store_folder,'storefinalindices'),'wb'))
				pickle.dump(storedots,open(os.path.join(pgd_store_folder,'storedots'),'wb'))


				
args=parser.parse_args()
Attack_Module=SemanticMLLAttack(args.configfile,args.expname)
Attack_Module.attack()

