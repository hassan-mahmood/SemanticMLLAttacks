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



class Predictor:
	def __init__(self,configfile,experiment_name):
		self.parseddata=DataParser(params={'configfile':configfile,'experiment_name':experiment_name,'mode':'test'})		
		#self.parse_data(configfile,experiment_name,losstype)
	
	def set_bn_eval(self,module):
		if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
			module.eval()


	def predict(self):
		params=self.parseddata.build()
		self.logger=params['logger']
		model=params['model']
		criterion=params['criterion']
		num_classes=params['num_classes']
		tree_path=params['tree_path']
		pred_file=params['pred_file']
		checkpoint_load_path=params['checkpoint_load_path']

		device=params['device']
		
		test_dataset=params['test_dataset']
		test_dataloader=params['test_dataloader']
		hierarchy_mapping_path=params['hierarchy_mapping_path']
		target_labels_path=params['target_labels_path']
		
		writer=params['writer']
		
		################################################################################
		# Stores classnames 

		################################################################################
		# Load the model
		imgsize=448
		args={'image_size':imgsize,'model_name':'tresnet_l',
		'model_path':checkpoint_load_path,
		'num_classes':80,'workers':0
		}
		args=argparse.Namespace(**args)
		state = torch.load(args.model_path, map_location='cpu')
		args.num_classes = state['num_classes']

		args.do_bottleneck_head = True
		model = create_model(args).cuda()
		model.load_state_dict(state['model'], strict=True)
		num_classes=args.num_classes
		################################################################################

		self.logger.write('Loss:',criterion)
		
		
		model=model.cuda()
		model.apply(self.set_bn_eval)

		for module in model.modules():
			#print(module)
			module.eval()


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
				tempdata.to_hdf(pred_file,key='df',mode='w')
				print('Tempdata:',tempdata.shape)
					
						
				
		tempdata=pd.DataFrame(clean_outputs.detach().cpu().numpy())
		tempdata['ids']=batch_img_ids
		tempdata.to_hdf(pred_file,key='df',mode='w')
		print('Tempdata:',tempdata.shape)


				
args=parser.parse_args()
Attack_Module=Predictor(args.configfile,args.expname)
Attack_Module.predict()

