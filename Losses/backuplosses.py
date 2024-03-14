from __future__ import print_function
import os
import sys
sys.path.append('./../')
#sys.path.append(os.getcwd())
from utils.utility import * 
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

class UAPLoss(nn.Module):
    #def __init__(self,loss_hyperparams,metadata,tree,common_parents_tree,common_hierarchy_tree,use_target_indices=False,isattack=False):
    def __init__(self,params):
        super(UAPLoss,self).__init__()
        #self.bceloss=nn.BCEWithLogitsLoss()
        self.orthscale=float(params['orthscale'])
        self.bceloss=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')
        #self.bceloss=AsymmetricLossOptimized()

    # def compute_orthogonal_loss(self,model):
    #     # get UAPs
    #     U=model.get_UAPs()
    #     #print('Shape of U:',U.shape)
    #     #print(U[0,:].shape)

    #     orthloss=0.0
    #     for k in range(2):
    #         tempU = U[k,:].squeeze()

    #         dotproduct=torch.matmul(tempU,tempU.T).cuda()
    #         #print(dotproduct.shape)

    #         mask=torch.ones_like(dotproduct).cuda() - torch.eye(dotproduct.shape[0]).cuda()
    #         #print(mask.shape)
    #         out=torch.mul(mask,dotproduct)
            
    #         orthloss += torch.sum(out**2)/(out.shape[0]*out.shape[1])

    #     tempU0 = U[0,:].squeeze()
    #     tempU1 = U[1,:].squeeze()

    #     dotproduct=torch.matmul(tempU0,tempU1.T).cuda()
    #     #print(dotproduct.shape)

    #     mask=torch.ones_like(dotproduct).cuda() - torch.eye(dotproduct.shape[0]).cuda()
    #     #print(mask.shape)
    #     out=torch.mul(mask,dotproduct)
        
    #     temploss = torch.sum(out**2)/(out.shape[0]*out.shape[1])
    #     orthloss+=temploss

    #     return orthloss,temploss
    def compute_orthogonal_loss(self,model):
        # get UAPs
        Features=model.get_UAPs_features()
        
        #print('Features length:',len(Features))
        #normalize feature vectors

        # with torch.no_grad():
        #     #U = U.div_(torch.norm(U,dim=1,keepdim=True))
        #     #U = U.div_(torch.linalg.vector_norm(U,ord=2,dim=1,keepdim=True))
        #     U = U.div_(torch.linalg.vector_norm(U,ord=2,dim=1,keepdim=True)+1e-10)

        #U = U.div_(torch.linalg.vector_norm(U,ord=2,dim=1,keepdim=True)+1e-10)
        orthloss=0.0

        for k in Features:
            normval=torch.linalg.vector_norm(k,ord=2,dim=1,keepdim=True)
            #print(normval)
            k=k/normval
            #print('k shape:',k.shape)

            dotproduct=torch.matmul(k,k.T).cuda()
            #print('dotproduct shape:',dotproduct.shape)

            #print(dotproduct)
            #diff=dotproduct*(1.0-torch.eye(dotproduct.shape[0])).cuda()
            expmat=torch.exp(dotproduct)*(1.0-torch.eye(dotproduct.shape[0])).cuda()
            #print(expmat)
            #print(diff)
            
            #diff=dotproduct - torch.eye(dotproduct.shape[0]).cuda()

            orthloss += 1.0/torch.sum(expmat)

            #orthloss += torch.sum(torch.square(diff))/(diff.shape[0]*diff.shape[1])
#            
            
        #print('U requires grad1:',U.requires_grad)
        #U=model.get_UAPs()
        #print('Shape of U:',U.shape)
        #print(U[0,:].shape)
        
        #print(dotproduct.shape)

        #mask=torch.ones_like(dotproduct).cuda() - torch.eye(dotproduct.shape[0]).cuda()
        #print(mask.shape)
        #out=torch.mul(mask,dotproduct)
        
        return orthloss

        # orthloss=0.0
        # for k in range(2):
        #     tempU = U[k,:,:].squeeze()

        #     dotproduct=torch.matmul(tempU,tempU.T).cuda()
        #     #print(dotproduct.shape)

        #     mask=torch.ones_like(dotproduct).cuda() - torch.eye(dotproduct.shape[0]).cuda()
        #     #print(mask.shape)
        #     out=torch.mul(mask,dotproduct)
            
        #     orthloss += torch.sum(out**2)/(out.shape[0]*out.shape[1])

        # tempU0 = U[0,:,:].squeeze()
        # tempU1 = U[1,:,:].squeeze()

        # dotproduct=torch.matmul(tempU0,tempU1.T).cuda()
        # #print(dotproduct.shape)

        # #mask=torch.ones_like(dotproduct).cuda() - torch.eye(dotproduct.shape[0]).cuda()
        # #print(mask.shape)
        # #out=torch.mul(mask,dotproduct)
        # out=dotproduct
        
        # temploss = torch.sum(out**2)/(out.shape[0]*out.shape[1])
        # orthloss+=temploss

        # return orthloss,temploss

    
    def forward(self,outputs,labels,target_class,use_selection_mask=True,model=None):
        with torch.no_grad():
            if use_selection_mask:
                selection_mask=torch.zeros_like(labels).float()
                selection_mask[:,target_class]=1.0
            else:
                selection_mask=torch.ones_like(labels).float()


        bceloss=torch.mul(self.bceloss(outputs,labels),selection_mask).sum()
        #bceloss=self.bceloss(outputs,labels).sum(1)

        
        
        #print(self._alpha*bceloss.item(),self._beta*normloss.item(),self._gamma*parent_margin_loss.item(),self._lambda*neg_margin_loss.item(), loss.item())
        norms=model.get_norms()
        normloss=torch.sum(norms)
        sumval=torch.sum(torch.linalg.vector_norm(norms,ord=float('inf'),dim=2,keepdim=True))

        #print(norms)
        # sumval=torch.sum(norms)
        # average=torch.tensor(norms.shape).squeeze().sum()
        # normloss=torch.sum(norms/average)
        #sumval=torch.sum(norms)
        
        

        orthloss=self.compute_orthogonal_loss(model)
        #orthloss=0.0
        #print(bceloss,orthloss)
        #print(bceloss.item(),normloss.item(),orthloss.item(),sumval.item())
        loss=bceloss + 0.0*normloss + self.orthscale*orthloss
        
        
        #loss=bceloss

        losses_dict={
        'bceloss':bceloss,
         'normloss':normloss,
         'orthloss':orthloss
        }


        return losses_dict,loss
        #return loss
