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
from Attacks.mla_lp import *
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
from MLGCN.models import * 


#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

pickle.HIGHEST_PROTOCOL = 4
torch.set_printoptions(edgeitems=27)
parser=argparse.ArgumentParser()
parser.add_argument('--configfile',default='configs/voc.ini')
parser.add_argument('expname')

# k=time.time()
# dims=3*224*224
# a=torch.randn(3,dims,1)
# b=torch.randn(3,dims,1)

# #print(torch.dot(a.flatten(),a.flatten()))
# #print(torch.dot(b.flatten(),b.flatten()))

# C=torch.cat((a,b),dim=2)
# D=torch.cat((torch.transpose(a,1,2),torch.transpose(b,1,2)),dim=1)
# print('C shape:',C.shape,', D shape:',D.shape)

# C_T_C=torch.bmm(torch.transpose(C,1,2),C)

# D_D_T=torch.bmm(D,torch.transpose(D,1,2))

# CTC_inv = torch.linalg.pinv(C_T_C)
# DDT_inv = torch.linalg.pinv(D_D_T)

# mid = torch.bmm(DDT_inv,CTC_inv)
# print('Mid shape:',mid.shape)

# out = torch.bmm(mid,torch.transpose(C,1,2))

# out=torch.bmm(torch.transpose(D,1,2),out)
# print(out.shape)
# #A_inv=torch.bmm(torch.bmm(torch.transpose(D,1,2),DDT_inv),torch.bmm(CTC_inv,torch.transpose(C,1,2)))
# #print(A_inv.shape)
# #torch.matmul(D.T)




# #print(out)
# #print(out.T)
# 0/0
# #out=torch.outer(a.flatten(),b.flatten())
# #out=torch.matmul(a,b.T)

# out=torch.matmul(a,b)


# print(time.time()-k)
# #out=torch.einsum("bi,bj->bij",a,b)
# #out=torch.matmul(a,b.T)
# print(out.shape)
# 0/0
# norm_a=torch.linalg.vector_norm(a.flatten(),ord=2)
# norm_b=torch.linalg.vector_norm(b.flatten(),ord=2)

# print(norm_a,norm_b)


# normalized_a=a/(norm_a+1e-9)
# normalized_b=b/(norm_b+1e-9)

# print(normalized_a.shape,normalized_b.shape)
# print(torch.linalg.vector_norm(normalized_a.flatten(),ord=2),torch.linalg.vector_norm(normalized_b.flatten(),ord=2))

# A = torch.matmul(a,b.T)+torch.matmul(b,a.T)
# print(A)

# U = torch.cat((normalized_a,normalized_b),dim=1)


# S = torch.eye(2,2)
# S_inv = torch.eye(2,2)
# S[0,0]=norm_a*norm_b
# S[1,1]=norm_a*norm_b

# S_inv[0,0]=1.0/(norm_a*norm_b)
# S_inv[1,1]=1.0/(norm_a*norm_b)


# V_T = torch.cat((normalized_b.T,normalized_a.T),dim=0)

# print('\n Original A:\n',A)
# torch_inv=torch.linalg.pinv(A)
# print('\nTorch inverse:\n',torch_inv)
# inv=torch.matmul(V_T.T,torch.matmul(S_inv,U.T))
# print('Our inverse:\n',inv)
# # print(torch.matmul(U,torch.matmul(1.0/(S+1e-9),V_T)))

# print(torch.matmul(inv,torch.matmul(A,inv)))

# print(torch.matmul(torch_inv,A))



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




# import collections
# data=pd.read_hdf('/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/test_labels_AT.h5',key='df').iloc[:,:-1].to_numpy()
# freq={}
# for i in range(data.shape[1]):
# 	#print(collections.Counter(data[:,i]))
# 	temp=collections.Counter(data[:,i])
# 	freq[i]=temp[1]
	

# print(freq)
# 0/0

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
#[25, 17, 32, 22, 18, 23, 21, 21, 15, 21, 21, 19, 27, 22, 22, 24, 18, 25, 25, 26, 30]
#[25, 17, 31, 22, 18, 23, 21, 20, 17, 20, 20, 16, 26, 24, 21, 21, 16, 27, 21, 26, 26]
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

	def compute_apgd(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, criterion,allotherdata,eps_val,selection_mask_target):
		print('Eps Value:',eps_val)
		apgdt = APGDAttack(model, eps=eps_val, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		#apgdt = APGDAttack(model, eps=10 , norm='L2', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		with torch.no_grad():
			return apgdt.perturb(inputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,use_target_indices=True,selection_mask=selection_mask_target)


	def test(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		combine_uaps=params['combine_uaps']
		
		device=params['device']
		
		test_dataset=params['test_dataset']
		test_dataloader=params['test_dataloader']
		
		
		writer=params['writer']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		p_norm=params['p_norm']


		####################
		# args={'image_size':448,'model_name':'tresnet_l',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# 'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 'num_classes':80,'workers':0
		# }

		# args=argparse.Namespace(**args)
		# state = torch.load(args.model_path, map_location='cpu')

		# #classes_list = np.array(list(state['idx_to_class'].values()))
		

		# args.num_classes = state['num_classes']
		# args.do_bottleneck_head = True
		# model = create_model(args).cuda()
		
		# # print(state['model'].keys())
		# # 0/0
		# model.load_state_dict(state['model'], strict=True)

		# #print([module for module in model.modules() if not isinstance(module, nn.Sequential)])
		
		####################

		# self.localinp=torch.from_numpy(pickle.load(open('MLGCN/data/nus/nus_glove_word2vec.pkl','rb'))).cuda()
        # self.localinp=torch.from_numpy(pickle.load(open('MLGCN/data/voc/voc_glove_word2vec.pkl','rb'))).cuda()
		num_classes=116
		model = gcn_resnet101(num_classes=num_classes, t=0.4, inp_file='MLGCN/data/nus/nus_glove_word2vec.pkl',adj_file='/home/hassan/hassan/Code/MLLAttacks/MLGCN/data/nus/nus_adj.pkl')

		checkpoint=torch.load('/mnt/raptor/hassan/weights/nus/sep/nus_sep_gcn_1/backup31.pth')
		print(checkpoint.keys())
		model.load_state_dict(checkpoint['state_dict'])
		
		self.attack_model = MLLP(model)


		####################
		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		self.logger.write('Loss:',criterion)
		
		
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		model=model.cuda()
		model.apply(self.set_bn_eval)

		for module in model.modules():
			#print(module)
			module.eval()

		#model.dotemp()

		#train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
		#train_optimizer = optim.SGD([self.U], lr=float(0.01))#,weight_decay=1e-4)
		mstones=[5,10,20,40,60]
		
		current_target_value=1.0
		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
		#model.Normalize_UAP(p_norm)
		#start_epoch=0
		# lrval=0.1
		# eps_val=0.05
		# num_iterations=400
		# alpha=eps_val/float(num_iterations)
		eps_val=0.01#2.5#0.5#0.005
		num_iterations=300
		pgd_step_size=eps_val#/150#2.5/80#0.005/80#0.00004#0.02#eps_val/40  #0.02/80 works well

		orth_step_size=0.004/150#2.5/80#0.005/80#0.00004#0.02#eps_val/40
		out_iterations=num_iterations
		in_iterations=1
		norm=float(np.inf)
		#norm = 2

		#apgdt = APGDAttack(model, eps=0.20, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		orthmll=OrthMLLAttacks()
		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')
		# model = gcn_resnet101(num_classes=self.num_classes, t=0.4, adj_file='/home/hassan/hassan/VOC-OTS-Attacks/MLGCN/data/voc/voc_adj.pkl')
		
		
		#num_classes=601
		#for epoch in range(start_epoch,start_epoch+num_epochs):
		epoch=0
		for eps_val in [0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01]:
		#for eps_val in [0.006,0.007,0.008,0.009,0.01]:
		# for eps_val in [0.003,0.004]:
		#for epoch in range(0,1):
			#current_target_value=1.0-current_target_value
			avg_meter=AverageMeter()
			if epoch in mstones:
				lrval=lrval*0.1

			# do not update BN
			#torch.set_grad_enabled(True)
			#self.logger.write('\nModel Training:')
			totals={i:torch.Tensor([1]) for i in range(num_classes)}
			success={i:torch.Tensor([0]) for i in range(num_classes)}
			flipped={i:torch.Tensor([0]) for i in range(num_classes)}
			perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

			
			tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/nus/tree','rb'))
			#num_classes=tree.shape[0]
			
			target_set_size=2
			max_batch=50
			attacktype=['mlaalpha','mlabeta','gmla','gmlab'][2]
			# attacktype='gmla'

			alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/aaai24/sequences/nusconsistentON/','all'+str(target_set_size)),'rb'))['target_classes']
			
			#alltargetlabels=[[329, 344], [530, 940], [1378, 1423], [1647, 1706], [4035, 5218], [3416, 4307], [1614, 5018], [2054, 5220], [6145, 6506], [981, 6210], [1166, 6662], [4840, 8517], [6347, 9446]]
			# print(alltargetlabels)
			for current_target_labels in alltargetlabels:
				
				targetlabels=current_target_labels
				
				# attackstorefolder=os.path.join(storefolder,'mlaproj')
				# pgd_store_folder=os.path.join('rebut/OFF/',attacktype,'_'.join([str(k) for k in current_target_labels]))
				#pgd_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/nus/OFF/',str(target_set_size),'mlaproj','_'.join([str(k) for k in current_target_labels]))
				#pgd_store_folder=os.path.join('/mnt/raptor/hassan/aaai24/nus/OFFOrthThresh/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
				pgd_store_folder=os.path.join('/mnt/raptor/hassan/cvpr24/supp/nus/All/',str(eps_val),str(target_set_size),attacktype,'_'.join([str(k) for k in current_target_labels]))
				
				#storefolder=os.path.join('/mnt/raptor/hassan/data/oi/attackedimages/OFF/',str(target_set_size),'_'.join([str(k) for k in current_target_labels]))
				#imgstorefolder=os.path.join(storefolder,'images')
				#attackstorefolder=os.path.join(storefolder,attacktype)
				#create_folder(imgstorefolder)
				#create_folder(attackstorefolder)
				#print(attackstorefolder)

				#print(pgd_store_folder)
				#treetargetlabels=[mapping[c] for c in current_target_labels if c in mapping.keys()]

				# print('Current target label:',current_target_labels)
				
				max_num_batches=1
				target_on_off=1
				#print(current_target_labels,target_on_off)
				
				print('Starting target labels:',current_target_labels, ', Goal:',('Turn ON','Turn OFF')[target_on_off])
				#current_store_folder=os.path.join('/mnt/raptor/hassan/MLLFinal/stores/',str(target_on_off),'_'.join([str(i) for i in current_target_labels]))
				# current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/oi/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]))

				# orth_store_folder=os.path.join(current_store_folder,'orth_linf','0.8')
				#pgd_store_folder=os.path.join(current_store_folder,'pgd_consistent_'+str(attacktype)+'070_linf')
				

				# pgd_fixed_store_folder=os.path.join(current_store_folder,'pgd_fixed_linf')
				# aorth_store_folder=os.path.join(current_store_folder,'aorth_l2')
				# apgd_store_folder=os.path.join(current_store_folder,'pgd_l2')
				
				#create_folder(os.path.join(orth_store_folder,'pert'))
				#create_folder(os.path.join(pgd_store_folder,'pert'))
				#create_folder(os.path.join(pgd_fixed_store_folder,'pert'))
				# create_folder(orth_store_folder)
				create_folder(pgd_store_folder)
				print(pgd_store_folder)
				
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

				nonflag=False

				for i, data in enumerate(tqdm(test_dataloader)):
					

					if number_of_batch_iterations>=max_num_batches:
						break

					(all_inputs,img_ids),origlabels=data
					all_inputs=all_inputs.to(device)
					origlabels=origlabels.to(device).float()

					model.eval()
					
					with torch.no_grad():
						outputs,features = model(all_inputs,{'epsilon_norm':0.0})
					
					#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
					labels=torch.clone(outputs).detach()
					labels=torch.where(labels>0,1,0).float()
					mytargetlabels=current_target_labels

					select_indices=torch.ones(size=(all_inputs.shape[0],))
					#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
					for k in mytargetlabels:
						select_indices=torch.logical_and(select_indices.float()==1.0,torch.logical_and(labels[:,k].cpu()==target_on_off,origlabels[:,k].cpu()==target_on_off))
						#select_indices=torch.logical_and(select_indices,labels[:,12].cpu()==0)
					# print(torch.count_nonzero(select_indices))
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
					# print('Total:',len(select_indices))
					
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

					break 

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

					# print('Original target labels:',mytargetlabels)
					if 'gmla' in attacktype:
						templabels=torch.clone(labels).detach().cpu().numpy()
						templabels=np.where(templabels<=0,0,1)
						# for t in mytargetlabels:
						# 	newlabels[:,t]=1-newlabels[:,t]
						countbefore=0
						countafter=0
						for c in range(templabels.shape[0]):
							
							current_label=templabels[c,:]
							# _,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)
							# print('Before:',g)
							# if g is True:
							# 	countbefore+=1

							treenodes=compute_tree_off_loss(np.copy(current_label),current_target_labels,tree)

							# print('Current label:',np.where(current_label==1.0)[0],'Tree nodes:',treenodes)
							# print(len(treenodes))
							
							#current_label[current_target_labels]=0.0
							current_label[treenodes]=0.0
							

							_,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)

							# print('G:',g)
							
							# 0/0
							origidx=treenodes
							#origidx=current_target_labels+treenodes

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
					# print('Scale:',scale_factor)
					currentinputs=torch.clone(inputs).detach()
					
					currentinputs.requires_grad=True

					allotherdata={'scale':scale_factor,
					'target_selection_mask':target_selection_mask,
					'nontarget_selection_mask':nontarget_selection_mask,
					'attacktype':attacktype,
					}
					pert_image,otherdata=self.compute_apgd(model,currentinputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,eps_val=eps_val,selection_mask_target=selection_mask_target)
					nonflag=True


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
						# current_label=np.zeros((601,),dtype=np.int32)
						# current_label[destindices]=np.copy(tempoutputs[c,:][origindices])
						current_label=tempoutputs[c,:]
						origtargetlabels=np.where(target_selection_mask[c,:]==1.0)[0]
						targetlabels=origtargetlabels
						#targetlabels=[mapping[t] for t in origtargetlabels if t in mapping.keys()]

						l=True 
						g = True 
						for k in targetlabels:
							templ,tempg=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
							#print(l,templ)
							l = l and templ 
							g = g and tempg
							
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

					# for itx in range(e.shape[0]):
					# 	# np.save(os.path.join('/mnt/raptor/hassan/data/oi/attackedimages/OFF/images/',str(itx)+'.npy'),pert_image[itx,:,:,:].detach().cpu().numpy())
					# 	np.save(os.path.join(imgstorefolder,str(itx)+'.npy'),pert_image[itx,:,:,:].detach().cpu().numpy())
					# 	np.save(os.path.join(attackstorefolder,str(itx)+'.npy'),e[itx,:,:,:].detach().cpu().numpy())
					# 	# np.save(os.path.join('/mnt/raptor/hassan/data/oi/attackedimages/OFF/perts'+str(attacktype)+'/',str(itx)+'.npy'),e[itx,:,:,:].detach().cpu().numpy())
					# print(pert_image.shape)
					# #/mnt/raptor/hassan/data/oi/attackedimages
					# 0/0


					# 0/0
					# pickle.dump(ac_scores,open('rebut/ntscores.npy','wb'))



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
					break
					
					# break 	
				# #store

				if nonflag is False:
					print('Non flag is False')
					continue
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

				print(pgd_store_folder)



				#print(storeadvoutputs.shape)

				
				




	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

