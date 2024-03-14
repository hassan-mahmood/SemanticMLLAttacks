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


		####################
		# args={'image_size':224,'model_name':'tresnet_l',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
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
		
		args={'image_size':224,'model_name':'tresnet_m',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# 'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		'model_path':'/mnt/raptor/hassan/weights/csl/mtresnet_opim_86.72.pth',
		#'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		'num_classes':80,'workers':0
		}
		args=argparse.Namespace(**args)
		state = torch.load(args.model_path, map_location='cpu')

		#classes_list = np.array(list(state['idx_to_class'].values()))
		

		args.num_classes = state['num_classes']
		args.do_bottleneck_head = True
		model = create_model(args).cuda()
		
		# print(state['model'].keys())
		# 0/0
		model.load_state_dict(state['model'], strict=True)

		# ####################

		
		model=model.cuda()
		model.apply(self.set_bn_eval)

		for module in model.modules():
			#print(module)
			module.eval()

		
		target_on_off=1	
		clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
		batch_img_ids=[]
				

		for i, data in enumerate(tqdm(test_dataloader)):
			
			(all_inputs,img_ids),labels=data
			all_inputs=all_inputs.to(device)
			labels=labels.to(device).float()					

			model.eval()
			
			with torch.no_grad():
				outputs,features = model(all_inputs,{'epsilon_norm':0.0})

			clean_outputs=torch.cat((clean_outputs,outputs),dim=0)
			batch_img_ids+=img_ids

			if (i+1)%20==0:
				tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
				tempdata['ids']=batch_img_ids
				tempdata.to_hdf('/mnt/raptor/hassan/data/csloi/Labels/cleanoutputs.h5',key='df',mode='w')
				print('Tempdata:',tempdata.shape)
					
						
				
		tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
		tempdata['ids']=batch_img_ids
		tempdata.to_hdf('/mnt/raptor/hassan/data/csloi/Labels/cleanoutputs.h5',key='df',mode='w')
		print('Tempdata:',tempdata.shape)
	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()



# import os
# import sys
# sys.path.append('./')
# from scripts.tree_loss import *
# from utils.utility import *
# from utils.confparser import *
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import numpy as np
# import torchvision
# #from torch.autograd.functional import jacobian
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import time
# import copy
# from tqdm import tqdm 
# from Models import *
# #from Datasets.NUSDataset import NUSImageDataset as ImageDataset
# from Datasets import * 
# import re
# from Logger.Logger import *
# import configparser
# import ast, json
# import argparse
# import pickle
# from datetime import datetime
# from torch.optim.lr_scheduler import MultiStepLR
# from torch.utils.tensorboard import SummaryWriter
# from Attacks.apgd import *
# from Attacks.mll import * 
# from Attacks.projected_gradient_descent import *
# import random 
# import gc 
# import time
# import itertools 
# from scripts.tree_loss import *
# sys.path.append('ASL/')
# #from src.helper_functions.helper_functions import mAP, AverageMeter, CocoDetection
# from src.models import create_model
# import numpy as np
# import os 


# #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = 'max_split_size_mb:512'

# pickle.HIGHEST_PROTOCOL = 4
# torch.set_printoptions(edgeitems=27)
# parser=argparse.ArgumentParser()
# parser.add_argument('--configfile',default='configs/voc.ini')
# parser.add_argument('expname')

# # k=time.time()
# # dims=3*224*224
# # a=torch.randn(3,dims,1)
# # b=torch.randn(3,dims,1)

# # #print(torch.dot(a.flatten(),a.flatten()))
# # #print(torch.dot(b.flatten(),b.flatten()))

# # C=torch.cat((a,b),dim=2)
# # D=torch.cat((torch.transpose(a,1,2),torch.transpose(b,1,2)),dim=1)
# # print('C shape:',C.shape,', D shape:',D.shape)

# # C_T_C=torch.bmm(torch.transpose(C,1,2),C)

# # D_D_T=torch.bmm(D,torch.transpose(D,1,2))

# # CTC_inv = torch.linalg.pinv(C_T_C)
# # DDT_inv = torch.linalg.pinv(D_D_T)

# # mid = torch.bmm(DDT_inv,CTC_inv)
# # print('Mid shape:',mid.shape)

# # out = torch.bmm(mid,torch.transpose(C,1,2))

# # out=torch.bmm(torch.transpose(D,1,2),out)
# # print(out.shape)
# # #A_inv=torch.bmm(torch.bmm(torch.transpose(D,1,2),DDT_inv),torch.bmm(CTC_inv,torch.transpose(C,1,2)))
# # #print(A_inv.shape)
# # #torch.matmul(D.T)




# # #print(out)
# # #print(out.T)
# # 0/0
# # #out=torch.outer(a.flatten(),b.flatten())
# # #out=torch.matmul(a,b.T)

# # out=torch.matmul(a,b)


# # print(time.time()-k)
# # #out=torch.einsum("bi,bj->bij",a,b)
# # #out=torch.matmul(a,b.T)
# # print(out.shape)
# # 0/0
# # norm_a=torch.linalg.vector_norm(a.flatten(),ord=2)
# # norm_b=torch.linalg.vector_norm(b.flatten(),ord=2)

# # print(norm_a,norm_b)


# # normalized_a=a/(norm_a+1e-9)
# # normalized_b=b/(norm_b+1e-9)

# # print(normalized_a.shape,normalized_b.shape)
# # print(torch.linalg.vector_norm(normalized_a.flatten(),ord=2),torch.linalg.vector_norm(normalized_b.flatten(),ord=2))

# # A = torch.matmul(a,b.T)+torch.matmul(b,a.T)
# # print(A)

# # U = torch.cat((normalized_a,normalized_b),dim=1)


# # S = torch.eye(2,2)
# # S_inv = torch.eye(2,2)
# # S[0,0]=norm_a*norm_b
# # S[1,1]=norm_a*norm_b

# # S_inv[0,0]=1.0/(norm_a*norm_b)
# # S_inv[1,1]=1.0/(norm_a*norm_b)


# # V_T = torch.cat((normalized_b.T,normalized_a.T),dim=0)

# # print('\n Original A:\n',A)
# # torch_inv=torch.linalg.pinv(A)
# # print('\nTorch inverse:\n',torch_inv)
# # inv=torch.matmul(V_T.T,torch.matmul(S_inv,U.T))
# # print('Our inverse:\n',inv)
# # # print(torch.matmul(U,torch.matmul(1.0/(S+1e-9),V_T)))

# # print(torch.matmul(inv,torch.matmul(A,inv)))

# # print(torch.matmul(torch_inv,A))



# seed=999
# # torch.manual_seed(seed)
# # torch.cuda.manual_seed(seed)
# # torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
# # np.random.seed(seed)  # Numpy module.
# # random.seed(seed)  # Python random module.
# # torch.manual_seed(seed)
# # torch.backends.cudnn.benchmark = False
# # torch.backends.cudnn.deterministic = True
# # random.seed(seed)
# # os.environ['PYTHONHASHSEED'] = str(seed)
# # np.random.seed(seed)
# # torch.manual_seed(seed)
# # torch.cuda.manual_seed(seed)
# # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
# # torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True




# # import collections
# # data=pd.read_hdf('/mnt/raptor/hassan/data/voc/vocdevkit/voc2007/ImageSets/Main/test_labels_AT.h5',key='df').iloc[:,:-1].to_numpy()
# # freq={}
# # for i in range(data.shape[1]):
# # 	#print(collections.Counter(data[:,i]))
# # 	temp=collections.Counter(data[:,i])
# # 	freq[i]=temp[1]
	

# # print(freq)
# # 0/0

# # from sympy import Matrix
# # from sympy.physics.quantum import TensorProduct
# # def my_nullspace(At, rcond=None):

# #     ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)

# #     #At[2,:]=At[1,:]*2
# #     #ut, st, vht = torch.linalg.svd(At, full_matrices=True)
# #     print(st)
# #     print(st.shape)
# #     print(ut.shape,vht.shape)


# #     #At = Matrix(At.detach().numpy())
# #     #nspace=At.nullspace()

# #     #print(TensorProduct(nspace,At))
# #     #print(nspace[0])
# #     print(At.shape)
# #     print(torch.matmul(vht,At))
# #     print(torch.matmul(uht,At))

# #     0/0

# #     vht=vht[st.shape[0]:,:]
# #     print(torch.matmul(vht,At.t()))
# #     0/0
# #     vht=vht.T        

# #     print(torch.matmul(vht.t(),At.t()))
# #     0/0
# #     Mt, Nt = ut.shape[0], vht.shape[1] 
# #     if rcond is None:

# #         rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
# #     tolt = torch.max(st) * rcondt
# #     numt= torch.sum(st > tolt, dtype=int)
# #     nullspace = vht[numt:,:].T.cpu().conj()
# #     # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
# #     return nullspace

# # G = torch.randn((10,3),dtype=torch.float32)
# # G_orth = torch.qr(G)[0].t()

# # n=my_nullspace(G_orth)
# # # print(n.shape)
# # # print(torch.matmul(G_orth,n))
# # print(G.shape,G_orth.shape)
# # print(torch.matmul(G_orth,G))



# # a=pd.read_hdf(os.path.join('/mnt/raptor/hassan/UAPs/best_7_10_13/','best_mll_0.h5'),key='df',mode='r')
# # a=a.iloc[:,:-1].to_numpy()
# # print(a)
# # 0/0
# # 0/0
# ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train', 'bottle', 'chair', 'dining_table', 'pot_plant', 'sofa', 'tv_monitor']
# #print(pickle.load(open('/mnt/raptor/hassan/UAPs/KG_data/voc/class_names','rb')))
# #0/0
# #bird 1, Dog 4, bicycle 8, chair 15
# #[25, 17, 32, 22, 18, 23, 21, 21, 15, 21, 21, 19, 27, 22, 22, 24, 18, 25, 25, 26, 30]
# #[25, 17, 31, 22, 18, 23, 21, 20, 17, 20, 20, 16, 26, 24, 21, 21, 16, 27, 21, 26, 26]
# class Tester:
# 	def __init__(self,configfile,experiment_name):
# 		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'test'})		
# 		#self.parse_data(configfile,experiment_name,losstype)
	
# 	def set_bn_eval(self,module):
# 		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
# 			module.eval()

# 	#[25, 25, 22, 25, 27, 22, 25, 35, 37, 33, 41]
# 	#[25, 25, 22, 25, 27, 21, 24, 35, 39, 37, 46]
# 	#[25, 25, 22, 25, 27, 23, 25, 36, 40, 37, 43]
# 	#[25, 25, 22, 24, 27, 20, 25, 35, 38, 35, 42]
# 	#[25, 25, 22, 24, 27, 20, 25, 36, 40, 34, 43]
# 	#[25, 25, 22, 24, 27, 22, 25, 33, 41, 36, 42]

# 	def compute_apgd(self,model,inputs,newlabels,mytargetlabels,nontargetlabels, criterion,allotherdata,selection_mask_target):
# 		apgdt = APGDAttack(model, eps=0.05, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
# 		with torch.no_grad():
# 			return apgdt.perturb(inputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,use_target_indices=True,selection_mask=selection_mask_target)


# 	def test(self):
# 		params=self.parseddata.build()
# 		self.logger=params['logger']
# 		model=params['model']
# 		criterion=params['criterion']
# 		num_classes=params['num_classes']
# 		combine_uaps=params['combine_uaps']
		
# 		device=params['device']
		
# 		test_dataset=params['test_dataset']
# 		test_dataloader=params['test_dataloader']
		
		
# 		writer=params['writer']
# 		weights_dir=params['weights_dir']
# 		epsilon_norm=params['eps_norm']
# 		p_norm=params['p_norm']


# 		####################
# 		args={'image_size':448,'model_name':'tresnet_l',
# 		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
# 		'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
# 		#'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
# 		'num_classes':80,'workers':0
# 		}
# 		args=argparse.Namespace(**args)
# 		state = torch.load(args.model_path, map_location='cpu')

# 		#classes_list = np.array(list(state['idx_to_class'].values()))
		

# 		args.num_classes = state['num_classes']
# 		args.do_bottleneck_head = True
# 		model = create_model(args).cuda()
		
# 		# print(state['model'].keys())
# 		# 0/0
# 		model.load_state_dict(state['model'], strict=True)

# 		#print([module for module in model.modules() if not isinstance(module, nn.Sequential)])
		
		
# 		####################

# 		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
# 		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
# 		#print('loaded again')
# 		self.logger.write('Loss:',criterion)
		
		
		
# 		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
# 		model=model.cuda()
# 		model.apply(self.set_bn_eval)

# 		for module in model.modules():
# 			#print(module)
# 			module.eval()

# 		#model.dotemp()

# 		#train_optimizer = optim.SGD(model.parameters(), lr=float(1.0))#,weight_decay=1e-4)
# 		#train_optimizer = optim.SGD([self.U], lr=float(0.01))#,weight_decay=1e-4)
# 		mstones=[5,10,20,40,60]
		
# 		current_target_value=0.0
# 		#train_optimizer = optim.SGD(model.get_params(), lr=float(0.1))#,weight_decay=1e-4)
# 		#model.Normalize_UAP(p_norm)
# 		#start_epoch=0
# 		# lrval=0.1
# 		# eps_val=0.05
# 		# num_iterations=400
# 		# alpha=eps_val/float(num_iterations)
# 		eps_val=0.004#2.5#0.5#0.005
# 		num_iterations=400
# 		pgd_step_size=eps_val#/150#2.5/80#0.005/80#0.00004#0.02#eps_val/40  #0.02/80 works well

# 		orth_step_size=0.004/150#2.5/80#0.005/80#0.00004#0.02#eps_val/40
# 		out_iterations=num_iterations
# 		in_iterations=1
# 		norm=float(np.inf)

# 		#apgdt = APGDAttack(model, eps=0.20, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
# 		orthmll=OrthMLLAttacks()
# 		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

# 		#num_classes=601
# 		#for epoch in range(start_epoch,start_epoch+num_epochs):
# 		for epoch in range(0,1):
# 			#current_target_value=1.0-current_target_value
# 			avg_meter=AverageMeter()
# 			if epoch in mstones:
# 				lrval=lrval*0.1

# 			# do not update BN
# 			#torch.set_grad_enabled(True)
# 			#self.logger.write('\nModel Training:')
# 			totals={i:torch.Tensor([1]) for i in range(num_classes)}
# 			success={i:torch.Tensor([0]) for i in range(num_classes)}
# 			flipped={i:torch.Tensor([0]) for i in range(num_classes)}
# 			perf_matches={i:torch.Tensor([0]) for i in range(num_classes)}

# 			#mytargetlabels=[7]
# 			#mytargetlabels=[1,4,8,15,17]
			
# 			alltargetlabels=[[i] for i in range(num_classes)] #batch size = 40 x 3
# 			#alltargetlabels=[[0,15,18],[0,10,11],[0,15,16]] #45, 26, 24, 59 batch size: 24
# 			#alltargetlabels=[[0,14,16],[0,15,18],[0,10,11],[0,15,16],[0,15,19],[0,11,12],[15,16,17],[15,17,18]] #batch size: 10 x 4

# 			#alltargetlabels=[[0,5],[0,12],[15,18],[0,9],[0,18],[0,4], [0,11],[0,8], [15,19], [0,14],[15,16],[0,10],[10,11],[0,15] ] #40 x 3
# 			#182, 58,110, 148, 96,146,90, 137, 40,69, 95, 57,51, 70    
# 			#
# 			#target_on_off=1
# 			#alltargetlabels=[[93, 302, 567]]
# 			#alltargetlabels=[[160, 91,321,6]]
# 			#alltargetlabels=[[182, 58, 110, 148, 96,]]#146, 90, 137, 40, 69, 95, 57, 51, 70]]
# 			alltargetlabels=[[4328, 4332, 4333, 4336, 5220]]#,1584,4723]]
# 			#alltargetlabels=[[197,  198,  205,  307,  329,  813, 1570, 1647]]
# 			#alltargetlabels=[[3761, 4324, 4326, 4327, 4328, 4329, 4332, 4333, 4334, 4336, 4337, 9446]]
# 			alltargetlabels=[[2788, 3761, 4324, 4326, 4328, 4329, 4332]]
# 			# this results in low success rate. alltargetlabels=[[2788, 3761, 4324,1422,2255,4558,1228,392,57]]#,[184, 350, 444, 549, 551]]
# 			alltargetlabels=[[77, 104, 202, 233, 255, 340]]
# 			alltargetlabels=[[3792, 8693, 2793, 2454, 8997, 787]]

# 			#813 1954 2056 2461 2727 2788 3192 3265 3416 3438 3880 3920 4179 4255 5673 5790 5794 6134 6248 6495 6506 6662
# 			#alltargetlabels=[[813,1954,2056,2461,2727]]#2788,3192,3265,4179,5673,5790]]

# 			#alltargetlabels=[[8211, 3860, 2024, 7306,  930, 6347, 3243, 5138, 4891, 2898, 6423, 3585]]#, 7831, 2483, 7958, 5926, 5676, 4537, 1048, 2331]]
# 			#[1039 1212 1642 1888 3223 3560 4348 4441 4457 5832 5921 7828 7909 8571 9293]
# 			#alltargetlabels=[[512, 1255, 2308, 2455, 2792, 3232, 4021, 4189, 5236, 6438, 6505, 7459, 8609, 8653, 9267]]
# 			#alltargetlabels=[[601,  605,  722]]
# 			#alltargetlabels=[[97]]#,213,503,504,508,1092,2237, 2474, 2766]]
# 			#97  213  503  504  508  509  530  533  549  550  556  563 1092 1378 1609 1614 1848 2237 2474 2766 3139 3146 3185 3233 3237 3619 3856 4235 4237 4372 4442 4530 4533 4694 4840 4970 4975 4977 4986 5146 5316 5430 5473 5539 5646 5698 5865 6205 6207 6334 6353 6984 7002 7518 8034 8052 8059 8345 8593 8683 8908 9049 9050 9321 9360
# 			#alltargetlabels=[[2054, 4324, 4326, 4328, 4329, 4332, 4334, 4336, 4337, 6347]]
# 			#alltargetlabels=[[77, 104, 179, 202, 233, 255, 329, 340, 344, 371, 447]]
# 			#alltargetlabels=[[3761, 4324, 4326, 4327, 4328, 4329, 4332, 4333, 4334, 4336, 4337, 9446]]
# 			#alltargetlabels=[[57,170]]
# 			#alltargetlabels=[[75]]#341,276, 277, 6]]
			
# 			#nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))
			
# 			#for current_target_labels in alltargetlabels:
# 			#for current_target_labels,target_on_off in itertools.product(alltargetlabels,[0,1]):
# 			tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/tree','rb'))
# 			#num_classes=tree.shape[0]
# 			mapping=pickle.load(open('/mnt/raptor/hassan/data/KG_data/oi/hierarchy_mapping','rb'))
# 			mapping_inverse = {v: k for k, v in mapping.items()}
# 			origindices=list(mapping.keys())
# 			destindices=list(mapping.values())


# 			target_set_size=1
# 			alltargetlabels=pickle.load(open(os.path.join('/mnt/raptor/hassan/bmvc/sequences/oiconsistentON/','new'+str(target_set_size)),'rb'))
# 			for current_target_labels in alltargetlabels[:20]:
				
# 				treetargetlabels=[mapping[c] for c in current_target_labels if c in mapping.keys()]

# 				print('Current target label:',current_target_labels)
# 				max_batch=20
# 				max_num_batches=1
# 				allset={}
# 				target_on_off=1
# 				#print(current_target_labels,target_on_off)
				
# 				print('Starting target labels:',current_target_labels, ', Goal:',('Turn ON','Turn OFF')[target_on_off])
# 				#current_store_folder=os.path.join('/mnt/raptor/hassan/MLLFinal/stores/',str(target_on_off),'_'.join([str(i) for i in current_target_labels]))
# 				current_store_folder=os.path.join('/mnt/raptor/hassan/bmvc/stores/oi/','norm_'+str(norm),str(eps_val),str(target_on_off),str(target_set_size),'_'.join([str(i) for i in current_target_labels]))

# 				orth_store_folder=os.path.join(current_store_folder,'orth_linf','0.8')
# 				pgd_store_folder=os.path.join(current_store_folder,'pgd_consistent_gmla_linf_ON')
# 				pgd_fixed_store_folder=os.path.join(current_store_folder,'pgd_fixed_linf')
# 				# aorth_store_folder=os.path.join(current_store_folder,'aorth_l2')
# 				# apgd_store_folder=os.path.join(current_store_folder,'pgd_l2')
				
# 				#create_folder(os.path.join(orth_store_folder,'pert'))
# 				#create_folder(os.path.join(pgd_store_folder,'pert'))
# 				#create_folder(os.path.join(pgd_fixed_store_folder,'pert'))
# 				# create_folder(orth_store_folder)
# 				create_folder(pgd_store_folder)
				
# 				batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 				orth_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 				#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 				#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 				pgd_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 				pgd_fixed_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 				clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

# 				batch_img_ids=[]
# 				batch_inputs=torch.empty((0,3,224,224),dtype=torch.float32).cuda()
# 				number_of_batch_iterations=0 

# 				storecleanoutputs=[]
# 				storeadvoutputs=[]
# 				storent=[]
# 				storefinalindices=[]
# 				storedots=[]
# 				storeimgids=[]



# 				fileit=0
# 				for i, data in enumerate(tqdm(test_dataloader)):
# 					# if i==0:
# 					# 	continue
# 					# get the inputs; data is a list of [image_ids, inputs, labels]
# 					#img_ids,inputs, labels = data 
# 					if number_of_batch_iterations>=max_num_batches:
# 						break

# 					(all_inputs,img_ids),labels=data
# 					all_inputs=all_inputs.to(device)
# 					labels=labels.to(device).float()					

# 					# with torch.no_grad():
# 					# 	outputs,features = model(all_inputs,{'epsilon_norm':0.0})
# 					# 	#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
# 					# 	labels=torch.clone(outputs).detach()
# 					# 	labels=torch.where(labels>0,1,0).float()
# 					model.eval()
# 					# biases=model.model.classifier.bias
# 					# print(biases)
# 					# 0/0

# 					# with torch.enable_grad():
# 					# 	for _ in range(30):
# 					# 		all_inputs.requires_grad=True
# 					# 		outputs,features = model(all_inputs,{'epsilon_norm':0.0})
# 					# 		t=torch.autograd.grad(outputs.sum(), [all_inputs])[0].detach()
# 					# 		print(torch.sum(all_inputs).item(),torch.sum(outputs).item(),torch.sum(features).item(),torch.sum(t).item())
# 					# 		model.zero_grad()
# 					# 		all_inputs.grad=None
# 					# 0/0
# 					with torch.no_grad():
# 						outputs,features = model(all_inputs,{'epsilon_norm':0.0})
# 						#print('outputs shape:',outputs.shape)
						
# 						#print('Outputs:',outputs.shape)

# 					clean_outputs=torch.cat((clean_outputs,outputs),dim=0)
# 					batch_img_ids+=img_ids

# 					if (i+1)%100==0:
# 						tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
# 						tempdata['ids']=batch_img_ids
# 						tempdata.to_hdf('/mnt/raptor/hassan/data/oi/Labels/preds/'+str(fileit)+'.h5',key='df',mode='w')

# 						clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 						batch_img_ids=[]
# 						fileit+=1
# 					continue
# 					# print(outputs[0]-biases[0])
# 					# l=[]
# 					# for i in range(num_classes):
# 					# 	gradval=torch.autograd.grad(outputs=outputs[0,i],inputs=features,retain_graph=True)[0]#.view(all_inputs.shape[0],3*224*224)
# 					# 	l.append(torch.dot(gradval[0],features[0]).item())
# 					# 	continue
# 					# 	#print('Output:',torch.dot(gradval[0],features[0]))
						
# 					# 	#print(gradval.shape)
# 					# 	0/0
# 					# 	l.append(torch.linalg.vector_norm(gradval,ord=2,dim=1)[0].item())
						
# 					# l.append(torch.linalg.vector_norm(gradval,ord=2,dim=1)[0].item())

					
# 					# 0/0
# 					# l=[]
# 					# weights=model.model.classifier.weight
# 					# for i in range(weights.shape[0]):
# 					# 	l.append(torch.linalg.vector_norm(weights[i,:],ord=2).item())

# 					# print(l)
# 					# 0/0



# 					#outputs,features = model(all_inputs,epsilon_norm=epsilon_norm,target_label=labels,target_class=torch.arange(num_classes),combine_uaps=True)
# 					labels=torch.clone(outputs).detach()
# 					labels=torch.where(labels>0,1,0).float()
# 					# for i in range(labels.shape[0]):
# 					# 	print(torch.where(labels[i,:]==1.0)[0].cpu().numpy())

# 					# 0/0
# 					#print()



# 					# for i in range(labels.shape[0]):
# 					# 	indices=torch.where(labels[i,:]==1)[0].cpu().numpy().tolist()
# 					# 	if(len(indices)!=2):
# 					# 		continue
# 					# 	indices='_'.join([str(i) for i in indices])

# 					# 	if indices not in allset.keys():
# 					# 		allset[indices]=1
# 					# 	else:
# 					# 		allset[indices]+=1
						

# 					#mytargetlabels=[7,10,13]
# 					#mytargetlabels=[2,3,5,7]
# 					#mytargetlabels=[3,6]
# 					#mytargetlabels=[0,3,14,15,16]
# 					#mytargetlabels=[3,5,7,10]
# 					mytargetlabels=current_target_labels

# 					select_indices=torch.ones(size=(all_inputs.shape[0],)).type(torch.ByteTensor)
# 					#print(select_indices.shape,all_inputs.shape,'all inputs',labels.shape)
# 					for k in mytargetlabels:
# 						#print('Select indices:',k,select_indices.shape)
# 						select_indices=torch.logical_and(select_indices,labels[:,k].cpu()==target_on_off)

# 						#select_indices=torch.logical_and(select_indices,labels[:,12].cpu()==0)


# 					temp_select_indices=torch.where(select_indices==True)[0]

# 					# from here, find the select_indices which are inconsistent after turning the labels ON/OFF
# 					templabels=torch.clone(labels).detach().cpu().numpy()
# 					print('Labels sum:',torch.sum(labels))
# 					templabels[:,mytargetlabels]=(1 - target_on_off)	

# 					select_indices=[]

# 					for c in temp_select_indices:
# 						current_label=np.zeros((601,),dtype=np.int32)
# 						current_label[destindices]=np.copy(templabels[c,:][origindices])
# 						current_label=np.where(current_label<=0,0,1)
# 						_,g=check_local_global_consistency(current_label,0,tree.shape[0],tree)
# 						#print('G:',g)
# 						if g is False:
# 							select_indices.append(c)
# 					# Ensure that now these are not consistent

# 					print('Total:',len(select_indices))
					
# 					if(len(select_indices)==0):
# 						continue 

# 					#0/0
# 					labels=labels[select_indices,:]
# 					inputs=all_inputs[select_indices,:]
# 					outputs=outputs[select_indices,:]
# 					img_ids=np.array(img_ids)[select_indices].tolist()
					
# 					#print(batch_labels.shape,labels.shape)
# 					batch_labels=torch.cat((batch_labels,labels),dim=0)
# 					clean_outputs=torch.cat((clean_outputs,outputs),dim=0)
# 					batch_inputs=torch.cat((batch_inputs,inputs),dim=0)

# 					if(len(inputs)==1):
# 						img_ids=[img_ids]

# 					batch_img_ids+=img_ids

# 					del inputs 
# 					if(batch_inputs.shape[0]<max_batch):
# 						continue

# 					del all_inputs 

# 					inputs=batch_inputs[:max_batch]
# 					labels=batch_labels[:max_batch]
# 					img_ids=batch_img_ids[:max_batch]
# 					clean_outputs=clean_outputs[:max_batch]
					
# 					for jk in range(1,2):
# 						to_use_selection_mask=True

# 						#newlabels=torch.clone(labels)
# 						newlabels=labels
# 						tempsucc=0
# 						flipped_labels=0

# 						model_params={
# 						'p_norm':p_norm,
# 						'epsilon_norm':epsilon_norm,
# 						'target_label':newlabels,
# 						'combine_uaps':combine_uaps,
# 						'target_class':(torch.Tensor([jk]).long(),torch.arange(num_classes))[combine_uaps]
# 						}

						
						
						
# 						selection_mask_target=torch.ones_like(newlabels,dtype=torch.float32)
# 						target_selection_mask=torch.zeros_like(newlabels,dtype=torch.float32)
# 						nontarget_selection_mask=torch.zeros_like(newlabels,dtype=torch.float32)

# 						templabels=torch.clone(labels).detach().cpu().numpy()
# 						templabels=np.where(templabels<=0,0,1)
# 						# for t in mytargetlabels:
# 						# 	newlabels[:,t]=1-newlabels[:,t]
# 						countbefore=0
# 						countafter=0
# 						for c in range(templabels.shape[0]):
# 							current_label=np.zeros((601,),dtype=np.int32)
# 							current_label[destindices]=np.copy(templabels[c,:][origindices])
# 							# _,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)
# 							# print('Before:',g)
# 							# if g is True:
# 							# 	countbefore+=1
							
# 							treenodes=compute_tree_on_loss(np.copy(current_label),treetargetlabels,tree)
							
# 							current_label=np.zeros((601,),dtype=np.int32)
# 							current_label[destindices]=templabels[c,:][origindices]
# 							current_label[treetargetlabels]=1.0
# 							current_label[treenodes]=1.0
							

# 							# _,g=check_local_global_consistency(np.copy(current_label),0,tree.shape[0],tree)

# 							# print('G:',g)
# 							# if g is True:
# 							# 	countafter+=1

# 							origidx=list(set([mapping_inverse[t] for t in treenodes if t in mapping_inverse.keys()]+current_target_labels))
							
# 							target_selection_mask[c,origidx]=1.0
# 							tempnontarget=list(set(list(range(num_classes))).difference(set(origidx)))
# 							nontarget_selection_mask[c,tempnontarget]=1.0
# 							newlabels[c,origidx]=1-newlabels[c,origidx]
# 							#assert(torch.sum(newlabels[c,origidx])==len(origidx))
# 							#assert(g is True)
						
# 						# print(countbefore,countafter)
# 						# 0/0

# 						nontargetlabels=list(set(list(range(num_classes))).difference(set(mytargetlabels)))

# 						#target_selection_mask[:,mytargetlabels]=1.0
# 						#nontarget_selection_mask[:,nontargetlabels]=1.0
# 						#newlabels[:,mytargetlabels]=1-newlabels[:,mytargetlabels]

						

# 						beforestore=clean_outputs
						
# 						inputs.requires_grad=True

						
# 						scale_factor=0.0
						
# 						model.zero_grad()
# 						model.eval()
# 						#print('inputs:',inputs.shape,newlabels.shape,inputs.requires_grad)
# 						print('Scale:',scale_factor)
# 						currentinputs=torch.clone(inputs).detach()
						
# 						currentinputs.requires_grad=True

# 						allotherdata={'scale':scale_factor,
# 						'target_selection_mask':target_selection_mask,
# 						'nontarget_selection_mask':nontarget_selection_mask,

# 						}
# 						pert_image,otherdata=self.compute_apgd(model,currentinputs,newlabels,mytargetlabels,nontargetlabels,criterion,allotherdata,selection_mask_target=selection_mask_target)
						
# 						#model,inputs,newlabels,criterion,selection_mask_target
# 						#best_e=orthmll.pgd_optimize_perturb(model,inputs,newlabels,mytargetlabels,nontargetlabels,eps_val,norm,orth_step_size ,criterion, out_iterations,in_iterations)
# 						#pert_image=inputs+best_e.cuda()
# 						#print('Min max:',torch.min(pert_image),torch.max(pert_image))
						
# 						#pert_image=pert_image.detach().cuda()
						

# 						with torch.no_grad():
# 							outputs,_ = model(pert_image,{'epsilon_norm':0.0})
# 							#afterstore=torch.clone(outputs)
# 							afterstore=outputs.detach()
# 							koutputs=torch.where(outputs>0,1,0).float()
						

# 						tempoutputs=torch.clone(koutputs).detach().cpu().numpy()
# 						select_indices=[]
# 						local,glo=0,0
# 						target_selection_mask=target_selection_mask.detach().cpu().numpy()
# 						for c in range(outputs.shape[0]):
# 							current_label=np.zeros((601,),dtype=np.int32)
# 							current_label[destindices]=np.copy(tempoutputs[c,:][origindices])
# 							origtargetlabels=np.where(target_selection_mask[c,:]==1.0)[0]
# 							targetlabels=[mapping[t] for t in origtargetlabels if t in mapping.keys()]

# 							l=True 
# 							for k in targetlabels:
# 								templ,g=check_local_global_consistency(np.copy(current_label),k,tree.shape[0],tree)
# 								#print(l,templ)
# 								l = l and templ 
								
# 							#print('G:',g)
# 							if g is True:
# 								glo+=1
# 							if l is True:
# 								local+=1

# 						print('L,g:',local,glo)
						

# 						success_indices=torch.where(((koutputs[:,mytargetlabels]==newlabels[:,mytargetlabels]).float().sum(1)==len(mytargetlabels))==True)[0]
# 						e=pert_image-currentinputs
# 						#print(torch.linalg.vector_norm(e.view(e.shape[0],-1),ord=2,dim=-1))
# 						#print('Norm '+str(norm)+':',torch.linalg.vector_norm(e.view(e.shape[0],-1),ord=norm,dim=-1))
# 						#print('Norm 1:',torch.linalg.vector_norm(torch.abs(e.view(e.shape[0],-1)),ord=1,dim=-1))
# 						#print('\nSuccess:',len(success_indices),torch.min(e).item(),torch.max(e).item())
						
# 						scores=(koutputs==newlabels).float()#.sum(1)
# 						#ac_scores=torch.count_nonzero(scores[:,mytargetlabels+nontargetlabels].sum(1)==len(mytargetlabels+nontargetlabels)).item()
# 						ac_scores=scores[:,nontargetlabels].sum(1)-len(nontargetlabels)
# 						#[ -50.,  -86.,   -30.,  -63.,  -30.,  -53., -105.,  -10., -112.,  -42.], #moving orth
# 						#[ -5., -12.,   0., -11.,  -4.,  -2., -48.,   0., -12., -11.] #moving orth + nontargetgrad
# 						#[ -62., -126.,   -109., -87.,  -84.,  -47., -127.,  -138., -93., -192.] # moving in target grad
# 						#[ -220., -1699., -292., -43.,  -36.,  -24.,  -83.,  -263., -50., -381.] # moving with svd orth
# 						#ac_scores=torch.count_nonzero(scores==koutputs.shape[1]).item()

# 						out=torch.abs(beforestore-afterstore)
# 						#print('PGD Fixed Success: %d, All Class success: %d'%(len(success_indices),ac_scores),', min: %.3f, max: %.3f, target sum: %.3f, nontarget sum: %.3f'%(torch.min(e).item(),torch.max(e).item(),torch.sum(out[:,mytargetlabels]).item(),torch.sum(out[:,nontargetlabels]).item()))
# 						print(ac_scores)
# 						print('PGD Fixed Success: %d, All Class success: %d'%(len(success_indices),1),', min: %.3f, max: %.3f, target sum: %.3f, nontarget sum: %.3f'%(torch.min(e).item(),torch.max(e).item(),torch.sum(out[:,mytargetlabels]).item(),torch.sum(out[:,nontargetlabels]).item()))
# 						#print('Scores:',scores)
# 						pickle.dump(ac_scores,open('tempdata2/ntscores'+str(scale_factor)+'.npy','wb'))



# 						storecleanoutputs.append(torch.clone(clean_outputs).detach())
# 						storeadvoutputs.append(torch.clone(outputs).detach())
# 						storeimgids+=list(img_ids)
# 						storent+=ac_scores.cpu().tolist()
# 						storefinalindices+=otherdata['finalindices'].cpu().tolist()
# 						storedots+=otherdata['dots']

						
						

# 						# # #################################
# 						#continue 	
# 						del pert_image
# 						del inputs 
# 						del newlabels
# 						#del apgdt
# 						del outputs 
# 						del afterstore 
# 						# del orth_outputs
# 						# del aorth_outputs
# 						del orth_outputs
# 						del pgd_outputs

# 						del batch_inputs
# 						#del apgd_outputs
# 						del clean_outputs

# 						number_of_batch_iterations+=1
						
# 						batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 						orth_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
						
# 						pgd_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 						pgd_fixed_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

# 						#aorth_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 						#apgd_outputs=torch.empty((0,20),dtype=torch.float32).cuda()
# 						clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()

# 						batch_img_ids=[]
# 						batch_inputs=torch.empty((0,3,224,224),dtype=torch.float32).cuda()
# 						gc.collect()
# 						torch.cuda.empty_cache()
						
				
# 				tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
# 				tempdata['ids']=batch_img_ids
# 				tempdata.to_hdf('/mnt/raptor/hassan/data/oi/Labels/preds/'+str(fileit)+'.h5',key='df',mode='w')

# 				clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
# 				batch_img_ids=[]
# 				0/0
# 				#store
# 				storecleanoutputs=torch.cat(storecleanoutputs,dim=0)
# 				tempdata=pd.DataFrame(storecleanoutputs.detach().cpu().numpy())
# 				tempdata['ids']=storeimgids
# 				tempdata.to_hdf(os.path.join(pgd_store_folder,'clean.h5'),key='df',mode='w')
# 				print('Store cleanoutputs:',storecleanoutputs.shape,tempdata.shape)

# 				storeadvoutputs=torch.cat(storeadvoutputs,dim=0)
# 				tempdata=pd.DataFrame(storeadvoutputs.detach().cpu().numpy())
# 				tempdata['ids']=storeimgids
# 				tempdata.to_hdf(os.path.join(pgd_store_folder,'adv.h5'),key='df',mode='w')
# 				print('Store advoutputs:',storeadvoutputs.shape,tempdata.shape)				



# 				pickle.dump(storent,open(os.path.join(pgd_store_folder,'storent'),'wb'))
# 				pickle.dump(storefinalindices,open(os.path.join(pgd_store_folder,'storefinalindices'),'wb'))
# 				pickle.dump(storedots,open(os.path.join(pgd_store_folder,'storedots'),'wb'))





# 				print(storeadvoutputs.shape)

				
				




	
# args=parser.parse_args()
# trainer=Tester(args.configfile,args.expname)
# trainer.test()

