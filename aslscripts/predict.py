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
		args={'image_size':224,'model_name':'tresnet_l',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		#'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		#'model_path':'/mnt/raptor/hassan/weights/nus/sep/nus_sep_asl_1/model-81.pt',
		'sep_features':0,
		#'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		'model_path':'/mnt/raptor/hassan/weights/lvis/asl/model-20.pt',
		'num_classes':1203,'workers':0
		}
		args=argparse.Namespace(**args)
		state = torch.load(args.model_path, map_location='cpu')
		args.num_classes=1203
		args.do_bottleneck_head = False
		model = create_model(args).cuda()
		model.load_state_dict(state['model_state'], strict=True)


		####################
		
		# args={'image_size':448,'model_name':'tresnet_l',
		# #'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# 'model_path':'/mnt/raptor/hassan/weights/asl/Open_ImagesV6_TRresNet_L_448.pth',
		# #'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 'num_classes':80,'workers':0
		# }

		# args=argparse.Namespace(**args)
		# state = torch.load(args.model_path, map_location='cpu')
		# args.num_classes = state['num_classes']
		# args.do_bottleneck_head = True
		# model = create_model(args).cuda()
		
		# model.load_state_dict(state['model'], strict=True)

		
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		model=model.cuda()
		model.apply(self.set_bn_eval)

		for module in model.modules():
			#print(module)
			module.eval()

		
		current_target_value=0.0
		
		eps_val=0.004#2.5#0.5#0.005
		norm=float(np.inf)

		#apgdt = APGDAttack(model, eps=0.20, norm='Linf', n_iter=300, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
		orthmll=OrthMLLAttacks()
		criterion=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

		tree=pickle.load(open('/mnt/raptor/hassan/data/KG_data/lvis/tree','rb'))
		#num_classes=tree.shape[0]
		
		max_batch=70
		max_num_batches=1
		allset={}
		target_on_off=0
		#print(current_target_labels,target_on_off)
		
		batch_labels=torch.empty((0,num_classes),dtype=torch.float32).cuda()
		clean_outputs=torch.empty((0,num_classes),dtype=torch.float32).cuda()
		batch_img_ids=[]
		number_of_batch_iterations=0 


		for i, data in enumerate(tqdm(test_dataloader)):
			

			(all_inputs,img_ids),labels=data
			all_inputs=all_inputs.to(device)
			labels=labels.to(device).float()					

			model.eval()
			
			with torch.no_grad():
				outputs,features = model(all_inputs,{'epsilon_norm':0.0})
				
			labels=torch.clone(outputs).detach()
			labels=torch.where(labels>0,1,0).float()
			
			
			temp_select_indices=torch.arange(all_inputs.shape[0])

			# from here, find the select_indices which are inconsistent after turning the labels ON/OFF
			templabels=torch.clone(labels).detach().cpu().numpy()
			# print('Labels sum:',torch.sum(labels))
			#templabels[:,mytargetlabels]=(1 - target_on_off)	

			select_indices=[]

			for c in temp_select_indices:
				
				current_label=np.copy(templabels[c,:])
				current_label=np.where(current_label<=0,0,1)
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
			outputs=outputs[select_indices,:]
			img_ids=np.array(img_ids)[select_indices].tolist()
			
			
			batch_labels=torch.cat((batch_labels,labels),dim=0)
			clean_outputs=torch.cat((clean_outputs,outputs),dim=0)

			if(labels.shape[0]==1):
				img_ids=[img_ids]

			batch_img_ids+=img_ids

			print(len(batch_img_ids),clean_outputs.shape)
			if (i+1)%20==0:
				tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
				tempdata['ids']=batch_img_ids
				tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/labels/consistentvaloutputs.h5',key='df',mode='w')
				print('Done saving')

		
					
		print(clean_outputs.shape)
		#storecleanoutputs=torch.cat(storecleanoutputs,dim=0)
		tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
		tempdata['ids']=batch_img_ids
		tempdata.to_hdf('/mnt/raptor/hassan/data/lvis/labels/consistentvaloutputs.h5',key='df',mode='w')
		


args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

