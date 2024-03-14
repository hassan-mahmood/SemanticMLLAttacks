

import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from  torch.nn.utils import weight_norm
from utils.utility import * 
import numpy as np 
import random 

torch.backends.cudnn.deterministic = True
np.random.seed(999)
random.seed(999)
torch.manual_seed(999)
#from torch.nn.utils import prune

# class UAPGen(torch.nn.Module):
# 	def __init__(self):
# 		super(UAPGen,self).__init__()

# 		self.model=ResNetModel({'num_classes':20})
# 		checkpoint=torch.load('/mnt/raptor/hassan/UAPs/weights/voc/br_l2_1/model-13.pt')
# 		self.model.load_state_dict(checkpoint['model_state'])
		
# 		self.imageSize=224
# 		self.conv = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(120, 32 * 8, 3, 1, 0, bias=True),
#             #nn.ConvTranspose2d(     100, 32 * 8, 3, 1, 0, bias=False),
#             nn.BatchNorm2d(32 * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 8, 32 * 4, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(32 * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 4, 32 * 2, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(32 * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(32 * 2,     32 , 3, 2, 1, bias=False),
#             nn.BatchNorm2d(32 ),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=True),
#             #nn.ConvTranspose2d(    32 ,      3, 3, 2, 1, bias=False),
#             nn.BatchNorm2d(3 ),
#             nn.ReLU(True),)

# 		self.fc = nn.Sequential(nn.Linear(3*33*33, 512),
# 			nn.BatchNorm1d(512 ),
# 			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
# 			nn.Linear(512, 1024),
# 			nn.BatchNorm1d(1024 ),
# 			nn.ReLU(True), # if we remove this, it seems predictions are more confident but are have greater perturbations
# 			nn.Linear(1024, 3*self.imageSize*self.imageSize))

# 		self.tanh = nn.Sequential(nn.Tanh(),)


# 	def forward(self,x,par,targetset=[]):
# 		epsilon_norm=par['epsilon_norm']
		
# 		#k=epsilon_norm*self.U[target_label[:,target_class].long(),target_class,:]
# 		# k=self.U[target_label[:,target_class].long(),target_class,:]
		

# 		# print(torch.linalg.vector_norm(self.U,ord=float('inf'),dim=2))
# 		# 0/0
		
# 		#print(out.shape)
# 		if(epsilon_norm>0.0):
# 			p_norm,target_label,target_class,combine_uaps=par['p_norm'],par['target_label'],par['target_class'],par['combine_uaps']

# 			x = self.conv(x)
# 			x = x.view(-1, 3*33*33)
# 			x = self.fc(x)
# 			x = x.view(-1, 3, self.imageSize, self.imageSize)
# 			return x

# 			#print(out.shape)
# 			#0/0
# 			#out=out.squeeze()
# 			#print(out.shape)
# 			# print(out[:5,:10])
# 			out=torch.sum(out,1)
			
# 			# with torch.no_grad():
# 			# 	p=torch.clone(out).detach()
# 			# 	mask=torch.zeros_like(p)
# 			# 	mask[torch.logical_or(p>epsilon_norm,p<-1*epsilon_norm)]=1.0
# 			# 	mask = (mask * p)/epsilon_norm + (1 - mask)

# 			# k = out/mask
# 			#with torch.no_grad():
# 			a=torch.linalg.vector_norm(out,ord=p_norm,dim=1)
			
# 			k=out*(epsilon_norm/(a+1e-10))[:,None]
# 			#print('Epsilon norm:',epsilon_norm)
# 			#print(torch.min(k),torch.max(k))


# 			#input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 			input_images = x + k.view(k.shape[0],x.shape[1],x.shape[2],-1)

# 		else:
# 			input_images = x 

# 		#a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)

# 		#k=out*(epsilon_norm/(a+1e-10))[:,None]

# 		#print(torch.min(k),torch.max(k))
# 		#k=out


		
		
# 		# 

# 		####
# 		#k=torch.sign(k)*torch.minimum(torch.abs(k),torch.Tensor([epsilon_norm]).cuda())
# 		#k=torch.sum(k,1)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=2))

# 		#0/0

# 		#print(k.shape)
# 		#0/0

# 		#return self.model(x)
# 		# x: input image
# 		# gt: ground truth label
# 		#print(torch.linalg.vector_norm(self.U,ord=2,dim=2))
# 		#print('Self u shape:',self.U.shape)
# 		# print('\n',torch.sum(self.U[:,:5,:],dim=2))
# 		#print('\n',torch.sum(self.U,dim=2))
# 		#print('\n',self.U[:,:5,:5])
# 		#k=self.U[target_label[:,target_class].long(),(torch.ones_like(target_label[:,target_class])*target_class).long(),:]
		
# 		#print(k[:5,:5])
# 		#print(k.shape)
# 		#print(torch.linalg.vector_norm(k,ord=float('inf'),dim=1,keepdim=True))
# 		# print(out.shape)
# 		# 0/0
		
# 		# out=self.U[target_label.long(),torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],-1).long()]
# 		# out=torch.sum(out,1)
# 		# a=torch.linalg.vector_norm(out,ord=float('inf'),dim=1)
# 		# # # #k=out*torch.min(torch.ones_like(a)*epsilon_norm,epsilon_norm/(a+1e-10))[:,None]
# 		# k=out*(epsilon_norm/(a+1e-10))[:,None]
# 		# # print(torch.min(k),torch.max(k))
# 		# # 0/0

		
		
# 		# k=k.view(k.shape[0],x.shape[1],x.shape[2],-1)
# 		# print(k[:5,0,0,:5])
		
		
# 		#input_images = x
		
# 		input_images=torch.clamp(input_images,min=0.0,max=1.0)
# 		#print('\n',torch.min(k).item(),torch.max(k).item(),torch.min(input_images).item(),torch.max(input_images).item())
# 		#0/0
	
# 		return self.model(input_images)

# 		# print(torch.linalg.vector_norm(k,ord=2,dim=1))


# class ResNetModel(torch.nn.Module):

# 	def __init__(self,model_params):
# 		super(ResNetModel,self).__init__()
# 		self.num_classes=model_params['num_classes']
# 		#resnetbasemodel=model_params['base_model'](pretrained=True)
# 		resnetbasemodel=models.resnet101(pretrained=True)
# 		self.basemodel = nn.Sequential(*list(resnetbasemodel.children())[:-1])
# 		self.prelu1=nn.PReLU()
# 		self.classifier=nn.Linear(in_features=2048,out_features=self.num_classes,bias=True)
# 		#self.get_model_weights=self.get_common_model_weights
	
# 	def forward(self,x):
# 		globalfeature=self.prelu1(self.basemodel(x))
# 		globalfeature=torch.flatten(globalfeature,1)
# 		output=self.classifier(globalfeature)
# 		return output,globalfeature


