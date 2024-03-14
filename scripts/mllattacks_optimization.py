import os
import sys
sys.path.append('./')
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
import random 

pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')

torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)

# from sympy import Matrix
# from sympy.physics.quantum import TensorProduct
# def my_nullspace(At, rcond=None):

#     ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)

#     #At[2,:]=At[1,:]*2
#     #ut, st, vht = torch.linalg.svd(At, full_matrices=True)
#     print(st)
#     print(st.shape)
#     print(ut.shape,vht.shape)


#     #At = Matrix(At.detach().numpy())
#     #nspace=At.nullspace()

#     #print(TensorProduct(nspace,At))
#     #print(nspace[0])
#     print(At.shape)
#     print(torch.matmul(vht,At))
#     print(torch.matmul(uht,At))

#     0/0

#     vht=vht[st.shape[0]:,:]
#     print(torch.matmul(vht,At.t()))
#     0/0
#     vht=vht.T        

#     print(torch.matmul(vht.t(),At.t()))
#     0/0
#     Mt, Nt = ut.shape[0], vht.shape[1] 
#     if rcond is None:

#         rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
#     tolt = torch.max(st) * rcondt
#     numt= torch.sum(st > tolt, dtype=int)
#     nullspace = vht[numt:,:].T.cpu().conj()
#     # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
#     return nullspace

# G = torch.randn((10,3),dtype=torch.float32)
# G_orth = torch.qr(G)[0].t()

# n=my_nullspace(G_orth)
# # print(n.shape)
# # print(torch.matmul(G_orth,n))
# print(G.shape,G_orth.shape)
# print(torch.matmul(G_orth,G))



# a=pd.read_hdf(os.path.join('/mnt/raptor/hassan/UAPs/best_7_10_13/','best_mll_0.h5'),key='df',mode='r')
# a=a.iloc[:,:-1].to_numpy()
# print(a)
# 0/0
# 0/0
['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining_table', 'pot_plant', 'sofa', 'tv_monitor']
#print(pickle.load(open('/mnt/raptor/hassan/UAPs/KG_data/voc/class_names','rb')))
#0/0
#bird 1, Dog 4, bicycle 8, chair 15

class Trainer:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'train'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()

	def train(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		combine_uaps=params['combine_uaps']
		
		device=params['device']
		train_optimizer=params['optimizer']
		scheduler=params['scheduler']
		train_dataloader=params['train_dataloader']
		val_dataset=params['val_dataset']
		val_dataloader=params['val_dataloader']
		
		start_epoch=params['start_epoch']
		num_epochs=params['num_epochs']
		weight_store_every_epochs=params['weight_store_every_epochs']
		writer=params['writer']
		min_val_loss=params['min_val_loss']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		p_norm=params['p_norm']
		
		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		self.logger.write('Loss:',criterion)
		
        
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		self.U = torch.nn.Parameter(torch.zeros((2,num_classes,3*224*224),dtype=torch.float32).cuda())
		self.U.requires_grad=False 

		model=model.cuda()

		model.dotemp()

		#train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		#train_optimizer = optim.SGD([self.U], lr=float(0.01))#,weight_decay=1e-4)
		mstones=[5,10,20,40,60]
		scheduler = MultiStepLR(train_optimizer, milestones=mstones, gamma=0.1)
		current_target_value=0.0
		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
		#model.Normalize_UAP(p_norm)
		#start_epoch=0
		lrval=0.1
		eps_val=0.05
		num_iterations=50
		alpha=eps_val/float(num_iterations)

		#apgdt = APGDAttack(model, eps=0.20, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		apgdt = APGDAttack(model, eps=eps_val, norm='Linf', n_iter=num_iterations, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

		#for epoch in range(start_epoch,start_epoch+num_epochs):
		for epoch in range(0,1):
			#current_target_value=1.0-current_target_value
			avg_meter=AverageMeter()
			if epoch in mstones:
 				lrval=lrval*0.1

			model.train()
			model.apply(self.set_bn_eval)
			model.model.eval()		# do not update BN
			#torch.set_grad_enabled(True)
			#self.logger.write('\nModel Training:')
			totals={i:torch.Tensor([1]) for i in range(num_classes)}
			success={i:torch.Tensor([0]) for i in range(num_classes)}
			flipped={i:torch.Tensor([0]) for i in range(num_classes)}
			perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

			#mytargetlabels=[7]
			mytargetlabels=[1,4,8,15]
			nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
			for i, data in enumerate(tqdm(train_dataloader)):
				# if i==0:
				# 	continue
				# get the inputs; data is a list of [image_ids, inputs, labels]
				#img_ids,inputs, labels = data 
				(all_inputs,img_ids),labels=data
				all_inputs=all_inputs.to(device)
				labels=labels.to(device).float()
				
				actualstore=None
				afterstore=None

				with torch.no_grad():
					outputs,features = model(all_inputs,{'epsilon_norm':0.0})
					#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
					actualstore=torch.clone(outputs)
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()


				df=pd.DataFrame(outputs.detach().cpu().numpy())
				df['ids']=np.array(img_ids)
				df.to_hdf(os.path.join('/mnt/raptor/hassan/UAPs/orig_preds/','best_mll_'+str(i)+'.h5'),key='df',mode='w')

				continue
				#for jk in range(num_classes):
				for jk in range(1,2):
					to_use_selection_mask=True
					# for totargetchange in [0.0,1.0]:
						#for jk in range(1):
					#for jk in range(1,2):

					#select_indices=torch.where(torch.logical_and((labels[:,jk]==current_target_value),(labels[:,2]==1)))[0]


					#newlabels=torch.clone(labels)
					#img_ids=torch.stack(list(img_ids), dim=0)
					#img_ids=img_ids[select_indices,:]

					newlabels=torch.clone(labels)
					
					inputs=torch.clone(all_inputs)
					########### experiment part #############
					#inputs= torch.rand_like(inputs)
					#newlabels=torch.zeros_like(newlabels)
					########### experiment part #############

					#if(current_target_value==0.0):
					# print(model.U[0,:5,:5])
					
					# for ts in mytargetlabels:
					# 	newlabels[:,ts] = 1 - newlabels[:,ts]
					# 	# if not torch.all(newlabels[:,ts]==1.0):
					# 	# 	print(newlabels[:,ts])
					# 	# assert(torch.all(newlabels[:,ts]==1.0))

					# gradmask=torch.zeros_like(model.U,dtype=torch.float32)
					# gradmask[:,jk,:]=1.0
					# model.U.register_hook(lambda grad: torch.sign(grad) * 0.1*gradmask)
					#model.U.register_hook(lambda grad: torch.sign(grad) * 0.1)
					
					#self.U.register_hook(lambda grad: torch.sign(grad) * 0.01)

					#newlabels[:,jk]=totargetchange

					tempsucc=0
					flipped_labels=0

					model_params={
					'p_norm':p_norm,
					'epsilon_norm':epsilon_norm,
					'target_label':newlabels,
					'combine_uaps':combine_uaps,
					'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
					}
					
					#train_optimizer.zero_grad()
					#inputs=torch.clone(all_inputs[select_indices,:,:,:])
	 				#inputs=torch.clone(all_inputs)
					#inputs.requires_grad=True
					
					# outputs,features = model(inputs,model_params)
					
					current_batch_size=outputs.shape[0]


					
					

					#selection_mask_target=selection_mask_target#.cpu().numpy()
					#selection_mask_non_target=selection_mask_non_target#.cpu().numpy()

					#inputs=inputs[:5]
					#newlabels=newlabels[:5]
					#selection_mask_target=selection_mask_target[:5,:]

					#mytargetlabels=[1,2]
					#nontargetlabels=[0]+list(range(3,num_classes))

					

					#newinputs=torch.clone(inputs)
					#self.e = torch.nn.Parameter(torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda())
					
					# 0/0
					# self.e = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32).cuda()
					# best_e = torch.zeros((inputs.shape[0],3*224*224),dtype=torch.float32)
					# best_preds_e=torch.zeros((inputs.shape[0],num_classes),dtype=torch.float32)

					# best_score = -1*torch.ones((inputs.shape[0]),dtype=torch.float32)

					# best_pert = torch.zeros((inputs.shape[0]),dtype=torch.float32)					

					# 0/0
					
					#newinputs=torch.clone(inputs)
					
					#train_optimizer = optim.SGD([self.e], lr=float(alpha))#,weight_decay=1e-4)

					mytargetlabels=[7,10,13]
					#mytargetlabels=[2]
					#mytargetlabels=[1,4,8,15]
					for ts in mytargetlabels:
						newlabels[:,ts]=1-newlabels[:,ts]
					
					selection_mask_target=torch.zeros_like(newlabels,dtype=torch.float32)
					for ts in mytargetlabels:
						selection_mask_target[:,ts]=1.0

					#selection_mask_non_target=1.0-torch.clone(selection_mask_target)


					# inputs.requires_grad=True 
					# outputs,globalfeature = model(inputs,{'epsilon_norm':0.0})

					#print(outputs1)


					#outputs2,_=model(inputs+torch.normal(torch.zeros_like(inputs), 0.01, generator=None, out=None),{'epsilon_norm':0.0})

					#print(torch.abs(outputs1-outputs2))
					#print('Sum with gaussian:',torch.sum(torch.abs(outputs1-outputs2)))

					#0/0
					#lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))
					# lossval = torch.square(outputs-torch.clone(outputs).detach())
					# print('outputs:',outputs.shape,', global feature:',globalfeature.shape)
					# allnorms=torch.zeros_like(outputs)
					# for class_idx in range(outputs.shape[1]):
					# 	for batch_idx in range(outputs.shape[0]):
					# 		#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
					# 		gradval=torch.autograd.grad(outputs=lossval[batch_idx,class_idx],inputs=inputs,retain_graph=True)[0].view(inputs.shape[0],3*224*224)
					# 		print(torch.linalg.vector_norm(gradval,dim=-1))
					# 		0/0
					# 		#allnorms[batch_idx,class_idx]=torch.linalg.vector_norm(gradval,ord=2,dim=-1)
						
					# print(allnorms)
					# 0/0

					pert_image=apgdt.perturb(inputs,newlabels,criterion,use_target_indices=True,selection_mask=selection_mask_target)
					outputs2,_ = model(pert_image,{'epsilon_norm':0.0})

					#afterstore=torch.clone(outputs2)
					koutputs=torch.where(outputs2>0,1,0).float()
					
					success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
					#print('Success:',len(success_indices),torch.min(self.e),torch.max(self.e))
					scores=(koutputs==newlabels).float().sum(1)
					print('Scores:',scores)

					#print(torch.sum(torch.abs(outputs1-outputs2)))
					#print(torch.sum(torch.abs(torch.special.expit(outputs1)-torch.special.expit(outputs2))))

					#out=torch.special.expit(actualstore)-torch.special.expit(afterstore)
					#print('Sum:',torch.sum(torch.abs(out)))


					# 0/0
					df=pd.DataFrame(outputs2.detach().cpu().numpy())
					df['ids']=np.array(img_ids)
					df.to_hdf(os.path.join('/mnt/raptor/hassan/UAPs/best_7_10_13/','apgd_'+str(i)+'.h5'),key='df',mode='w')


					continue 


					nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))

					mll_num_iterations=80
					for it in tqdm(range(mll_num_iterations)):
						for sub_it in range(5):
							print('Sub it:',sub_it)
							#train_optimizer.zero_grad()
							#self.e.register_hook(lambda grad: torch.sign(grad) * alpha)
							# if(it%10==0):
							# 	print('\n\nE values:',torch.sum(self.e,dim=1),', Norms:',torch.linalg.norm(self.e,ord=float('inf')))
							#newinputs=torch.clone(inputs)+self.e.view(inputs.shape[0],3,224,224)
							#inputs.requires_grad=False 
							#self.e.requires_grad=False 
							model.zero_grad()

							newinputs=torch.clone(inputs)+self.e.view(inputs.shape[0],3,224,224)
							newinputs.requires_grad=True
							outputs,_ = model(newinputs,{'epsilon_norm':0.0})

							afterstore=torch.clone(outputs)
							koutputs=torch.where(outputs>0,1,0).float()
							# if it%20==0:
							# 	print(koutputs[1,:])
							# 	print(newlabels[1,:])
							# 	print(torch.equal(koutputs[1,:].float(),newlabels[1,:].float()))

							
							success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
							print('Success:',len(success_indices),torch.min(self.e),torch.max(self.e))
							scores=(koutputs==newlabels).float().sum(1)
							
							
							
							for s_idx,s in enumerate(success_indices):
								if scores[s_idx]>=best_score[s]:
									best_score[s]=scores[s_idx]
									best_e[s]=torch.clone(self.e[s])
									best_preds_e[s]=outputs[s]
									best_pert[s]=torch.max(newinputs[1]-inputs[1]).item()

									#print('best score for s',s,best_score[s])
							
							#0/0
							
							lossval = criterion(outputs, newlabels.type(torch.cuda.FloatTensor))
							print('Scores:',scores,', Loss:',torch.sum(lossval))

							#lossval = torch.mul(allloss,selection_mask_target)/newinputs.shape[0]
							#print('BCE Loss:',lossval)
							#lossval=lossval*50
							
							#nontargetlabels=[1,2,3]
							
							
							mytargetgrads=[]



							if sub_it==0:
								nontargetgrads=[]
								for idx_ts,ts in enumerate(nontargetlabels):

									#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(nontargetlabels+mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
									gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
									nontargetgrads.append(gradval.view(gradval.shape[0],-1,1))
								
								nontargetgrads=torch.cat(nontargetgrads,-1)
								nontargetgrads=torch.qr(nontargetgrads)[0]


							for idx_ts,ts in enumerate(mytargetlabels):

								gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=(True,False)[idx_ts==len(mytargetlabels)-1])[0].view(newinputs.shape[0],3*224*224)
								#gradval=-1*torch.autograd.grad(outputs=lossval[:,ts].sum(),inputs=newinputs,retain_graph=True)[0].view(newinputs.shape[0],3*224*224)
								mytargetgrads.append(gradval.view(gradval.shape[0],-1,1))



							mytargetgrads=torch.cat(mytargetgrads,-1)

							##############################
							mytargetgrads=torch.sum(mytargetgrads,-1).view(mytargetgrads.shape[0],-1,1)
							#print(torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1))

							#mytargetgrads=normalize_vec(mytargetgrads,max_norm=1.0,norm_p=2)
							a=torch.linalg.vector_norm(mytargetgrads,ord=2,dim=1)
							mytargetgrads=mytargetgrads*(1.0/(a+1e-10))[:,None]

							#print(mytargetgrads.shape)


							# A = QR -> Take Q as the orthonormal basis that span the columns of nontargetgrads
							

							proj_target_grads=mytargetgrads-torch.bmm(nontargetgrads,torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))

							# print(torch.linalg.vector_norm(proj_target_grads,ord=2,dim=1))
							# for t in range(10):#nontargetgrads.shape[0]):
							# 	print(torch.matmul(torch.transpose(nontargetgrads[t].squeeze(),0,1),proj_target_grads[t]))
							# 0/0

							#print(torch.bmm(torch.transpose(nontargetgrads,1,2),mytargetgrads))

							#print(torch.bmm(torch.transpose(nontargetgrads,1,2),proj_target_grads))

							#lossval.backward()
							#train_optimizer.step()
							with torch.no_grad():
								#self.e = self.e+normalize_vec(proj_target_grads,max_norm=eps_val/float(num_iterations),norm_p=float('inf')).squeeze()
								self.e = normalize_vec(self.e+proj_target_grads.squeeze(),max_norm=(it+1)*eps_val/float(mll_num_iterations),norm_p=float('inf')).squeeze()

							
							#print('Sum:',torch.sum(self.e))
							del newinputs 
							del outputs 
							del lossval 
							self.e.requires_grad=False 

							# if(it%20==0):
							# 	print('\n\n\nOutputs:',outputs[1,:5],torch.sum((koutputs[1]==newlabels[1]).float()))
							# 	#print('Dot product:',torch.dot(u,v))
							# 	print('Min max:',torch.min(newinputs[1]-inputs[1]).item(),torch.max(newinputs[1]-inputs[1]).item())

							
							#u[-1] = -(v[:-1] @ u[:-1]) / v[-1]

							#newinputs[0]=newinputs[0].detach()+0.001*torch.sign(t_grad[0]).cuda()
							#newinputs[0]=newinputs[0].detach()+0.00001*torch.sign(a.view(1,3,224,224)).cuda()
							#with torch.no_grad():
							#	self.e.data=normalize_vec(self.e,max_norm=eps_val,norm_p=float('inf'))

					#print(best_score)


					#nt_grad=torch.autograd.grad(outputs=outputs[:, nontargetlabels], inputs=inputs,grad_outputs=torch.ones_like(outputs[:,nontargetlabels]),retain_graph=False)[0].view(current_batch_size,-1)
					

					#np.save(os.path.join('/mnt/raptor/hassan/UAPs/best_7/','best_preds_'+str(i)),best_preds_e.detach().cpu().numpy())
					#np.save(os.path.join('/mnt/raptor/hassan/UAPs/best_7/','newlabels_'+str(i)),newlabels.cpu().numpy())
					#np.save(os.path.join('/mnt/raptor/hassan/UAPs/best_7/','best_pert_'+str(i)),best_pert.detach().cpu().numpy())
					
					#print(actualstore)
					#print(afterstore)
					#print(actualstore-afterstore)
					#out=torch.special.expit(actualstore)-torch.special.expit(afterstore)
					#print(out)
					#print('Sum:',torch.sum(torch.abs(out)))

					newinputs=torch.clone(inputs).cuda()+best_e.view(inputs.shape[0],3,224,224).cuda()
					#newinputs.requires_grad=True
					outputs,_ = model(newinputs,{'epsilon_norm':0.0})
					koutputs=torch.where(outputs>0,1,0).float()
					
					
					#print('Success:',len(success_indices),torch.min(self.e),torch.max(self.e))
					scores=(koutputs==newlabels).float().sum(1)
					print('Scores:',scores)

					afterstore=torch.clone(outputs)
					#print(torch.abs(actualstore-afterstore))
					print('Abs sum:',torch.sum(torch.abs(actualstore-afterstore)))
					#print(torch.abs(actualstore-afterstore))
					print('Expit sum:',torch.sum(torch.abs(torch.special.expit(actualstore)-torch.special.expit(afterstore))))

					#np.save(os.path.join('/mnt/raptor/hassan/UAPs/best_7_10_13/','best_mll_'+str(i)),outputs.detach().cpu().numpy())
					df=pd.DataFrame(outputs)
					df['ids']=np.array(img_ids)
					df.to_hdf(os.path.join('/mnt/raptor/hassan/UAPs/best_1_4_8_15/','best_mll_'+str(i)+'.h5'),key='df',mode='w')


					continue



					
					pert_image=apgdt.perturb(inputs,newlabels,criterion,use_target_indices=True,selection_mask=selection_mask_target)
					outputs,_ = model(pert_image,{'epsilon_norm':0.0})

					afterstore=torch.clone(outputs)
					koutputs=torch.where(outputs>0,1,0).float()
					
					success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
					#print('Success:',len(success_indices),torch.min(self.e),torch.max(self.e))
					scores=(koutputs==newlabels).float().sum(1)
					print('Scores:',scores)

					out=torch.special.expit(actualstore)-torch.special.expit(afterstore)
					print(torch.abs(actualstore-afterstore))
					print('Sum:',torch.sum(torch.abs(out)))



					0/0
					



					

					# pert_image=apgdt.perturb(inputs,newlabels,criterion,use_target_indices=False,selection_mask=selection_mask)
					# print(torch.min(pert_image),torch.max(pert_image))
					# print(torch.min(pert_image-inputs),torch.max(pert_image-inputs))

					# out,_=model(pert_image,None,None)
					# out=torch.where(out>0,1,0)
					# for ts in mytargetlabels:
					# 	print('TS:',out[:,ts])
					# #print(pert_image.shape)
					# 0/0

					
					#out=torch.autograd.grad(outputs=outputs[0, :], inputs=inputs[0])[0]#,grad_outputs=torch.ones_like(outputs[0,:]),create_graph=False,retain_graph=True,only_inputs=False)[0]

					#print(out.shape)
					# print(outputs.shape)
					# myout 
					# def myfunc(x):
					# 	outputs,features = model(x,model_params)
					# 	myout=outputs
					# 	return outputs[0,1:5]
					# 	#return model(x,model_params)[0][0,1:5]

					# #grad=jacobian(myfunc,inputs, create_graph=True)



					
					# 0/0
					# losses_dict,loss = criterion(outputs,newlabels,jk,use_selection_mask=to_use_selection_mask,model=model)
					
					
					#print('sum before update:',torch.sum(model.U.grad))
					
					# loss.backward()
					#
					

					
					#print(model.uap_weights.flatten())
					# 
					
					# with torch.no_grad():
					# 	model.Normalize_UAP(p_norm,epsilon_norm)



					# if((i+1)%5==0):
					# 	with torch.no_grad():
					# 		k=self.U[torch.ones_like(newlabels[:,0]).long(),torch.Tensor([1]).long(),:]
					# 		X=inputs+ k.view(inputs.shape[0],3,224,224)
					# 		X=torch.clamp(X,min=0.0,max=1.0)
					# 		outputs,features = model(X,model_params)
					# 		print('Loss:',totalloss,', Total:',inputs.shape[0],', count:',torch.count_nonzero(outputs[:,1]>0).item())
					# 		print(torch.sum(self.U,dim=2)[1,:])

					# with torch.no_grad():
					# 	#self.U.div_(torch.norm(self.U, dim=float('inf'), keepdim=True))
					# 	#self.U.div_(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2,keepdim=True)+1e-10)
					# 	self.U.data=normalize_vec(self.U,max_norm=0.2,norm_p=p_norm)
					
					# self.U.requires_grad=True 

					# outputs=torch.where(outputs>0,1,0)
					# flip_select=outputs[:,jk]==newlabels[:,jk]
					# tempsucc=torch.count_nonzero(flip_select).cpu()
					# flipped_labels=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()

					# perf_match=outputs!=newlabels
					# perf_match=torch.sum(perf_match.float(),dim=1)
					# perf_match=torch.count_nonzero(perf_match==0.0)

					
					# for l in losses_dict.keys():
					# 	avg_meter.update('tr_'+l,losses_dict[l].item())

					# totals[jk]+=outputs.shape[0]
					# flipped[jk]+=flipped_labels
					# success[jk]+=tempsucc
					# perf_matches[jk]+=perf_match.cpu().item()

					# #print(', '.join(["{:.3f}".format(x) for x in [bceloss.item(),pw_bceloss.item(),U_sum_bceloss.item(),Up_sum_bceloss.item(),orthloss.item(),ind_loss.item(),normloss.item(),sumval.item()]]))

					
					# lossst=', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])

					# print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Perf:',perf_match.item(),', Flipped:',"{:.3f}".format(flipped_labels/(tempsucc.item()+1e-10)),"weights: {:.2f}".format(torch.sum(model.U).item()),lossst)
					# #print(torch.sum(model.U,dim=2))
					# #print(loss)
					
					# avg_meter.update('tr_train_loss',loss.item())
					
					
					# if(tempsucc==inputs.shape[0]):
					#  	break
					
					# # totals[jk]+=outputs.shape[0]
					# # flipped[jk]+=flipped_labels
					# # success[jk]+=tempsucc
					# # perf_matches[jk]+=perf_match

					# #outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
					# #percents.append((torch.count_nonzero(outputs[:,jk]<=0)/outputs.shape[0]).item())
					
					# # for p in model.parameters():
					# # 	if(p.requires_grad is True):
					# # 		#print(p.shape)
					# # 		p.grad=0.0001*torch.sign(p.grad)
					# #nn.utils.clip_grad_norm(model.parameters(), max_norm=10.0)
					
					# #scheduler.step()



				# print(torch.where(newlabels[0,:]==1)[0],torch.where(outputs[0,:]>0)[0])
				# print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
				# 	)
				
				
				# 	
			# print statistics
			# trainstatout = '\nTrain - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]) +'\nFlipped:'+', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nPerfect:'+', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])

			# scheduler.step()
			#losses=avg_meter.get_values()

			# lossst=''
			# for l in losses.keys():
			# 	if 'tr' in l:
			# 		lossst+=l+':'+str(losses[l])+', '
			
			# print('\n',lossst)
			# continue
			# model.eval()

			# #torch.set_grad_enabled(False)
			# self.logger.write('Model Evaluation:')
			# #vallabels=np.zeros(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			# #valpreds=np.empty(shape=(len(val_dataset)*num_classes,num_classes),dtype=np.float32)
			
			# totals={i:torch.Tensor([1]) for i in range(num_classes)}
			# success={i:torch.Tensor([0]) for i in range(num_classes)}
			# flipped={i:torch.Tensor([0]) for i in range(num_classes)}
			# perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

			# #current_target_value=0.0
			# with torch.no_grad():
			# 	for i,data in enumerate(tqdm(val_dataloader)):
			# 		#img_ids,inputs, labels = data 
			# 		(all_inputs,img_ids),labels=data
			# 		all_inputs=all_inputs.to(device)
			# 		labels=labels.to(device).float()
					
			# 		outputs,features = model(all_inputs,{'epsilon_norm':0.0})
			# 		#outputs,features = model(all_inputs,epsilon_norm=0.0,target_label=labels,target_class=torch.Tensor([0]).long())
			# 		labels=torch.clone(outputs).detach()
			# 		labels=torch.where(labels>0,1,0).float()
					
					
			# 		#for jk in range(num_classes):
			# 		for jk in range(1,3):
			# 			#for totargetchange in [0.0,1.0]:
			# 		#for jk in range(1):
			# 		#for jk in range(1,2):
			# 			select_indices=torch.where(labels[:,jk]==current_target_value)[0]
			# 			#select_indices=torch.where(torch.logical_and((labels[:,jk]==current_target_value),(labels[:,2]==1)))[0]
			# 			#print(torch.count_nonzero(select_indices))
			# 			# if(torch.count_nonzero(select_indices)==0):
			# 			# 	continue
					
			# 			newlabels=torch.clone(labels[select_indices,:])
			# 			inputs=torch.clone(all_inputs[select_indices,:,:,:])

			# 			#newlabels=torch.clone(labels)
			# 			#inputs=torch.clone(all_inputs)

			# 			########### experiment part #############
			# 			#inputs= torch.rand_like(inputs)
			# 			#newlabels=torch.zeros_like(newlabels)
			# 			########### experiment part #############

			# 			#
			# 			#newlabels = 1 - newlabels
			# 			newlabels[:,jk] = 1 - newlabels[:,jk]
			# 			model_params={
			# 			'p_norm':p_norm,
			# 			'epsilon_norm':epsilon_norm,
			# 			'target_label':newlabels,
			# 			'combine_uaps':combine_uaps,
			# 			'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
			# 			}
						
			# 			outputs,features = model(inputs,model_params)
			# 			#newlabels[:,jk] = totargetchange
						
			# 			#print(inputs.shape,newlabels.shape)
					
			# 			#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=jk)
			# 			#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.arange(num_classes))
						
			# 			#outputs,features = model(inputs,epsilon_norm=epsilon_norm,target_label=newlabels,target_class=torch.Tensor([jk]).long())
			# 			#print(torch.where(outputs[0,:]>0,1,0))
						

			# 			# if(current_target_value==1.0):
			# 			# 	success[jk]+=torch.count_nonzero(outputs[:,jk]<=0).cpu()
			# 			# else:
			# 			# 	success[jk]+=torch.count_nonzero(outputs[:,jk]>0).cpu()	
			# 			losses_dict,loss = criterion(outputs,newlabels,jk,use_selection_mask=False,model=model)
						
						
			# 			lossst=', '.join([x[:4]+" {:.3f}".format(losses_dict[x]) for x in losses_dict.keys()])

			# 			for l in losses_dict.keys():
			# 				avg_meter.update('val_'+l,losses_dict[l].item())

			# 			avg_meter.update('val_val_loss',loss.item())

			# 			outputs=torch.where(outputs>0,1,0)
						
			# 			perf_match=outputs!=newlabels
			# 			perf_match=torch.sum(perf_match.float(),dim=1)
			# 			perf_match=torch.count_nonzero(perf_match==0.0)
						
						

			# 			totals[jk]+=outputs.shape[0]
			# 			flip_select=outputs[:,jk]==newlabels[:,jk]
			# 			tempsucc=torch.count_nonzero(flip_select).cpu()
			# 			success[jk]+=tempsucc
			# 			flipped[jk]+=torch.count_nonzero(outputs[flip_select,:].flatten()!=newlabels[flip_select,:].flatten()).cpu().item()#-tempsucc.item()
			# 			perf_matches[jk]+=perf_match.cpu().item()
						
						
			# 			print('Total:',inputs.shape[0],', Success:',tempsucc.item(),', Flipped:',flipped_labels/num_classes,torch.sum(model.U).item())

				
					
			# 		#print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk])).item()) for jk in range(num_classes)]))
			# 		print('Percentage:',', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),
			# 			'\nFlipped:',', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)]),
			# 			'\nPerfect:',', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			# 			)
					
					
			# 		#bceloss,normloss,parent_margin_loss,neg_margin_loss,loss=criterion(outputs,labels,model.get_fc_weights())
			# 	# writer.add_scalar('Val Accuracy',val_acc,epoch)

			# now = datetime.now()
			
			# #val_acc=eval_performance(valpreds,vallabels,all_class_names)
			# statout=now.strftime("%d/%m/%Y %H:%M:%S")
			# #train_loss,val_loss,newstatout=avg_meter.get_stats(epoch,writer)
			# #print('Weights:',model.uap_weights.flatten())
			# statout = statout + '\nEpoch: '+str(epoch)+'\n'
			# #statout=statout+' - '+newstatout+', Val Acc: %.3f'%(val_acc)+', Current target value: '+str(current_target_value)
			# statout = statout + trainstatout
			# #print('\nVal - Percentage: '+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)]),', Flipped:',', '.join(["{:.3f}".format((flipped[jk]/(totals[jk]*num_classes+1e-10)).item()) for jk in range(num_classes)]))
			# #statout = statout +'\nVal - Percentage: '+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			# statout = statout + '\nVal - Percentage:,'+', '.join(["{:.3f}".format((success[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nFlipped:'+', '.join(["{:.3f}".format((flipped[jk]/(success[jk]+1e-10)).item()) for jk in range(num_classes)])+'\nPerfect:'+', '.join(["{:.3f}".format((perf_matches[jk]/(totals[jk]+1e-10)).item()) for jk in range(num_classes)])
			
			# losses=avg_meter.get_values()
			

			# statout+='\nTrain: '
			# for l in losses.keys():
			# 	if 'tr' in l:
			# 		statout+=l.split('tr_')[1]+': '+str("{:.3f}".format(losses[l]))+', '

			# statout+='\nVal: '

			# for l in losses.keys():
			# 	if 'val' in l:
			# 		statout+=l.split('val_')[1]+': '+str("{:.3f}".format(losses[l]))+', '


			# for l in losses.keys():
			# 	writer.add_scalar(l,losses[l],epoch)

			# #statout='\nEpoch: %d , Train loss: %.3f, Val loss: %.3f, Val Acc: %.3f, BCE: %.3f, Norm: %.3f, Parent Margin: %.3f, Neg Margin: %.3f' %(epoch, train_loss_meter.avg,val_loss_meter.avg,val_acc,bce_loss_meter.avg,norm_loss_meter.avg,parent_margin_loss_meter.avg,neg_margin_loss_meter.avg)

			# self.logger.write(statout)
			# weight_name='model-'

			# checkpointdict={
			# 	'optim_state':train_optimizer.state_dict(),
			# 	'model_state':model.state_dict(),
			# 	'epoch':epoch,
			# 	'min_val_loss':0.0,
			# 	'current_val_loss':0.0,
			# 	'training_loss':0.0,
			# 	'val_acc':0.0
			# }

			# if(epoch%weight_store_every_epochs==0):
			# 	store_checkpoint(checkpointdict,os.path.join(weights_dir,'model-'+str(epoch)+'.pt'))
			# 	print('Model stored')



	
args=parser.parse_args()
trainer=Trainer(args.configfile,args.expname)
trainer.train()

