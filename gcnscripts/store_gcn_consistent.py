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
		
		device=params['device']
		
		test_dataset=params['test_dataset']
		test_dataloader=params['test_dataloader']
		
		
		writer=params['writer']
		weights_dir=params['weights_dir']
		epsilon_norm=params['eps_norm']
		p_norm=params['p_norm']


		# ####################
		# args={'image_size':224,'model_name':'tresnet_l',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/nus/sep/nus_sep_asl_1/model-81.pt',
		# 'model_path':'/mnt/raptor/hassan/weights/nus/sep/nus_sep_asl_1_apgd/model-50.pt',
		# 'sep_features':1,
		# #'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 'num_classes':80,'workers':0
		# }
		# args=argparse.Namespace(**args)
		# state = torch.load(args.model_path, map_location='cpu')

		# #classes_list = np.array(list(state['idx_to_class'].values()))
		

		# # args.num_classes = state['num_classes']
		# args.num_classes=116
		# args.do_bottleneck_head = True
		# model = create_model(args).cuda()
		
		# # print(state['model'].keys())
		# # 0/0
		# print(state.keys())
		# model.load_state_dict(state['model_state'], strict=True)

		# #print([module for module in model.modules() if not isinstance(module, nn.Sequential)])
		# # ####################
		# model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='/home/hassan/hassan/Code/MLLAttacks/MLGCN/data/nus/nus_adj.pkl')

		# checkpoint=torch.load('/mnt/raptor/hassan/weights/nus/sep/nus_sep_gcn_1/backup31.pth')
		# print(checkpoint.keys())
		# model.load_state_dict(checkpoint['state_dict'])
		# # ####################

		model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='/home/hassan/hassan/Code/MLLAttacks/MLGCN/data/voc/voc_adj.pkl')

		#checkpoint=torch.load('/mnt/raptor/hassan/weights/nus/sep/nus_sep_gcn_1/backup31.pth')
		checkpoint=torch.load('/mnt/raptor/hassan/weights/voc/sep/voc_sep_gcn_1/backup30.pth')
		print(checkpoint.keys())
		model.load_state_dict(checkpoint['state_dict'])
		
		# ####################

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
		
		tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/voc/tree','rb'))
		batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
		batch_img_ids=[]
		

		for i, data in enumerate(tqdm(test_dataloader)):

			(all_inputs,img_ids),labels=data
			all_inputs=all_inputs.to(device)
			labels=labels.to(device).float()					

			model.eval()
			
			with torch.no_grad():
				outputs,features = model(all_inputs)
				
			#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
			labels=torch.clone(outputs).detach()
			labels=torch.where(labels>0,1,0).float()
			
			templabels=torch.clone(labels).detach().cpu().numpy()
			#templabels[:,mytargetlabels]=(1 - target_on_off)
			select_indices=[]

			for c in range(templabels.shape[0]):
				current_label=templabels[c,:]
				_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
				#print('G:',g)
				if g is True:
					select_indices.append(c)
			# Ensure that now these are not consistent

			print('Total:',len(select_indices))
			
			if(len(select_indices)==0):
				continue 

			#0/0
			labels=labels[select_indices,:]
			img_ids=np.array(img_ids)[select_indices].tolist()
			#print(batch_labels.shape,labels.shape)
			batch_labels=torch.cat((batch_labels,labels),dim=0)

			if(len(all_inputs)==1):
				img_ids=[img_ids]

			batch_img_ids+=img_ids


			if (i+1)%20==0:
				tempdata=pd.DataFrame(batch_labels.detach().cpu().numpy())
				tempdata['ids']=batch_img_ids
				#tempdata.to_hdf('/mnt/raptor/hassan/data/nus/Labels/nusgcntestlabelsconsistent.h5',key='df',mode='w')
				tempdata.to_hdf('/mnt/raptor/hassan/data/voc/Labels/vocgcntestlabelsconsistent.h5',key='df',mode='w')
				print(batch_labels.shape)

		tempdata=pd.DataFrame(batch_labels.detach().cpu().numpy())
		tempdata['ids']=batch_img_ids
		# tempdata.to_hdf('/mnt/raptor/hassan/data/nus/Labels/nusgcntestlabelsconsistent.h5',key='df',mode='w')
		tempdata.to_hdf('/mnt/raptor/hassan/data/voc/Labels/vocgcntestlabelsconsistent.h5',key='df',mode='w')
		print(batch_labels.shape)
				
				
				
				




	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

