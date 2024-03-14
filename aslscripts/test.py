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
from sklearn.metrics import accuracy_score
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
		apgdt = APGDAttack(model, eps=0.05, norm='Linf', n_iter=400, eot_iter=1, rho=.75, seed=0, device=torch.device('cuda:0'))
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
		# 'sep_features':0,
		# #'model_path':'/mnt/raptor/hassan/weights/nus/asl/NUS_WIDE_TRresNet_L_448_65.2.pth',
		# 'model_path':'/mnt/raptor/hassan/weights/lvis/asl/model-20.pt',
		# 'num_classes':1203,'workers':0
		# }
		# args=argparse.Namespace(**args)
		# state = torch.load(args.model_path, map_location='cpu')
		# #classes_list = np.array(list(state['idx_to_class'].values()))
		# #args.num_classes = state['num_classes']
		# args.num_classes=1203
		# args.do_bottleneck_head = False
		# model = create_model(args).cuda()
		
		
		# model.load_state_dict(state['model_state'], strict=True)

		# #print([module for module in model.modules() if not isinstance(module, nn.Sequential)])
		
		# ####################


		# #model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='/home/hassan/hassan/Code/MLLAttacks/MLGCN/data/nus/nus_adj.pkl')
		# model = gcn_resnet101(num_classes=num_classes, t=0.4, adj_file='/home/hassan/hassan/Code/MLLAttacks/MLGCN/data/voc/voc_adj.pkl')

		# #checkpoint=torch.load('/mnt/raptor/hassan/weights/nus/sep/nus_sep_gcn_1/backup31.pth')
		# checkpoint=torch.load('/mnt/raptor/hassan/weights/voc/sep/voc_sep_gcn_1/backup30.pth')
		# print(checkpoint.keys())
		# model.load_state_dict(checkpoint['state_dict'])

		

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

		####################

		#model.base_model.load_state_dict(torch.load(self.checkpoint_load_path)['model'])
		#model.base_model.load_state_dict(torch.load('/mnt/raptor/hassan/weights/nus/asl/new.pt')['model_state'])
		#print('loaded again')
		self.logger.write('Loss:',criterion)
		
		
		
		#train_dataloader=torch.utils.data.DataLoader(train_dataset, batch_size=self.batchsize, shuffle=True, num_workers=4)
		model=model.cuda()

		vallabels=np.zeros(shape=(len(test_dataset),num_classes),dtype=np.float32)
		valpreds=np.zeros(shape=(len(test_dataset),num_classes),dtype=np.float32)
		
		dataidx=0
		totals={i:torch.Tensor([1]) for i in range(num_classes)}
		success={i:torch.Tensor([0]) for i in range(num_classes)}
		current_target_value=0.0
		with torch.no_grad():
		    for i,data in enumerate(tqdm(test_dataloader)):
		        #img_ids,inputs, labels = data 
		        (all_inputs,img_ids),labels=data
		        all_inputs=all_inputs.to(device)
		        labels=labels.to(device).float()

		        outputs,features = model(all_inputs,[])
		        
		        #vallabels[dataidx:dataidx+outputs.shape[0]]=labels.detach().cpu().numpy()
		        #valpreds[dataidx:dataidx+outputs.shape[0]]=outputs.detach().cpu().numpy()
		        
		        dataidx=dataidx+outputs.shape[0]
		        outputs=torch.where(outputs>0,1,0)
		        for j in range(outputs.shape[0]):

		        	print('\n',torch.where(outputs[j,:]==1.0))
		        	print(torch.where(labels[j,:]==1.0))
		        0/0
		
		vallabels=np.where(vallabels>0,1,0)
		valpreds=np.where(valpreds>0,1,0)

		valscore=accuracy_score(vallabels,valpreds)
		print('Accuracy:',valscore)




	
args=parser.parse_args()
trainer=Tester(args.configfile,args.expname)
trainer.test()

