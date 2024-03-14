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

class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=5, gamma_pos=1, clip=0.001, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class GUAPLoss(nn.Module):
    def __init__(self,params):
        super(GUAPLoss,self).__init__()
        self.bceloss=torch.nn.BCEWithLogitsLoss(weight=None, size_average=None, reduce=None, reduction='none')

    def forward(self,outputs,labels,target_class,use_selection_mask=True,model=None):
        with torch.no_grad():
            if use_selection_mask:
                selection_mask=torch.zeros_like(labels).float()
                selection_mask[:,target_class]=1.0
            else:
                selection_mask=torch.ones_like(labels).float()


        bceloss=torch.mul(self.bceloss(outputs,labels),selection_mask).sum()
        
        loss = bceloss
        0/0

        losses_dict={
        'bce_loss':bceloss
        }
        # losses_dict={
        # 'bceloss':bceloss,
        # }


        return losses_dict,loss





class UAPLoss(nn.Module):
    #def __init__(self,loss_hyperparams,metadata,tree,common_parents_tree,common_hierarchy_tree,use_target_indices=False,isattack=False):
    def __init__(self,params):
        super(UAPLoss,self).__init__()
        #self.bceloss=nn.BCEWithLogitsLoss()
        self.bcescale=float(params['bcescale'])
        self.pwscale = float(params['pwscale'])
        self.orthscale=float(params['orthscale'])
        self.usumscale=float(params['usumscale'])
        self.upsumscale=float(params['upsumscale'])
        self.indscale=float(params['indscale'])
        self.weightscale=float(params['weightscale'])
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
    def compute_orthogonal_loss(self,model,target_class,use_selection_mask):
        # get UAPs

        Outputs,F=model.get_UAPs_features()
        # Outputs=O[1]
        # Features=F[1]

        orthloss=0.0

        for Features in F:
        
            with torch.no_grad():
                normval=torch.linalg.vector_norm(Features,ord=2,dim=1,keepdim=True)
            
            k=Features/normval

            dotproduct=torch.matmul(k,k.T).cuda()
            ones=torch.ones_like(dotproduct)-torch.eye(dotproduct.shape[0],dtype=torch.float32).cuda()
            expmat=torch.exp(dotproduct)*ones.cuda()

            # dotproduct=torch.matmul(k_wograd,k.T).cuda()
            # #print('Dot product:',dotproduct.shape)
            # ones=torch.ones_like(dotproduct)-torch.eye(dotproduct.shape[0]).cuda()
            # expmat=torch.exp(dotproduct)*ones.cuda()

            orthloss+= 1.0/torch.sum(expmat)

        orthloss = orthloss/(Outputs[0].shape[0]*Outputs[0].shape[0]*len(F))

        return orthloss,Outputs

        Features=model.get_UAPs_features()
        
        #print('Features length:',len(Features))
        #normalize feature vectors

        # with torch.no_grad():
        #     #U = U.div_(torch.norm(U,dim=1,keepdim=True))
        #     #U = U.div_(torch.linalg.vector_norm(U,ord=2,dim=1,keepdim=True))
        #     U = U.div_(torch.linalg.vector_norm(U,ord=2,dim=1,keepdim=True)+1e-10)

        #U = U.div_(torch.linalg.vector_norm(U,ord=2,dim=1,keepdim=True)+1e-10)
        orthloss=0.0
        
        n_classes=Features[0].shape[0]
        Features=[torch.cat(Features,dim=0)]
        
        for k in Features:
            normval=torch.linalg.vector_norm(k,ord=2,dim=1,keepdim=True)
            #print(normval)
            k=k/normval

            with torch.no_grad():
                k_wograd=torch.clone(k).detach()

            
            #print('k shape:',k.shape)

            #dotproduct=torch.matmul(k,k.T).cuda()
            #print('dotproduct shape:',dotproduct.shape)

            #print(dotproduct)
            #diff=dotproduct*(1.0-torch.eye(dotproduct.shape[0])).cuda()
            #expmat=torch.exp(dotproduct)*(1.0-torch.eye(dotproduct.shape[0])).cuda()

            # if use_selection_mask:
            #     dotproduct=torch.matmul(k_wograd,k[target_class,:]).cuda()
            #     ones=torch.ones_like(dotproduct)
            #     ones[target_class]=0.0
            #     expmat=torch.exp(dotproduct)*ones.cuda()
            # else:
            #     dotproduct=torch.matmul(k_wograd,k.T).cuda()
            #     print('Dot product:',dotproduct.shape)
            #     ones=torch.ones_like(dotproduct)-torch.eye(dotproduct.shape[0]).cuda()
            #     print(ones)
            #     #ones[target_class]=0.0
            #     expmat=torch.exp(dotproduct)*ones.cuda()
            #     print(expmat)
            #     0/0
            dotproduct=torch.matmul(k_wograd,k[target_class,:]).cuda()
            ones=torch.ones_like(dotproduct)
            ones[target_class]=0.0
            expmat=torch.exp(dotproduct)*ones.cuda()

            # dotproduct=torch.matmul(k_wograd,k.T).cuda()
            # #print('Dot product:',dotproduct.shape)
            # ones=torch.ones_like(dotproduct)-torch.eye(dotproduct.shape[0]).cuda()
            # expmat=torch.exp(dotproduct)*ones.cuda()

            orthloss += 1.0/torch.sum(expmat)
            #ones[target_class]=0.0
            
            #print(expmat)
            #print(expmat.shape)
            #0/0
            
            #print(expmat)
            #0/0

            # for tc in [target_class,target_class+n_classes]:
            #     dotproduct=torch.matmul(k_wograd,k[tc,:]).cuda()
            #     ones=torch.ones_like(dotproduct)
            #     ones[target_class]=0.0
            #     expmat=torch.exp(dotproduct)*ones.cuda()
            #     orthloss += 1.0/torch.sum(expmat)

            #print(expmat)
            #print(diff)
            
            #diff=dotproduct - torch.eye(dotproduct.shape[0]).cuda()

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

    
    def forward(self,outputs,labels,target_class,use_selection_mask=True,model=None,selection_mask=None):
        # with torch.no_grad():
        #     if use_selection_mask:
        #         # selection_mask=torch.zeros_like(labels).float()
        #         # selection_mask[:,target_class]=1.0
        #         pass 
        #     else:
        #         selection_mask=torch.ones_like(labels).float()
        with torch.no_grad():
            selection_mask=torch.ones_like(labels).float()


        bceloss=torch.mul(self.bceloss(outputs,labels),selection_mask).sum()
        #bceloss=self.bcescale*(bceloss/outputs.shape[0])

        losses_dict={
        'bce_loss':bceloss,
        
        }

        return losses_dict,bceloss



        U=model.get_params()
        ##############################
        #f(U-i + U_i') = 0
        
        #print(torch.range(0,self.num_classes-1).long().expand(target_label.shape[0],1).shape)
        
        #mul_factor=self.uap_weights[target_label[:,target_class].long(),target_class]
        # Uaps=U*uap_weights[:,:,None]
        Uaps=U

        pw_Uaps=torch.sum(Uaps,dim=0)
        pw_Uaps=rescale_to_image_range(pw_Uaps,dimval=1)
        pw_outputs,_=model(pw_Uaps.view(-1,3,224,224),{'epsilon_norm':0.0})
        pw_selection_mask=torch.eye((labels.shape[1]),dtype=torch.float32).cuda()
        pw_labels=torch.zeros((pw_outputs.shape[0],labels.shape[1]),dtype=torch.float32).cuda()
        pw_bceloss=torch.mul(self.bceloss(pw_outputs,pw_labels),pw_selection_mask).sum()

        pw_bceloss=self.pwscale*(pw_bceloss/pw_outputs.shape[0])
        #print('Pairwise Loss:',pw_bceloss)

        ##############################

        # f(sum U_i = 1)
        U_ones=rescale_to_image_range(torch.sum(Uaps[1,:,:],dim=0),dimval=0)
        U_sum_outputs,_ = model(U_ones.view(-1,3,224,224),{'epsilon_norm':0.0})
        U_sum_labels=torch.ones((U_sum_outputs.shape[0],labels.shape[1]),dtype=torch.float32).cuda()
        U_sum_bceloss=self.bceloss(U_sum_outputs,U_sum_labels).sum()

        U_sum_bceloss=self.usumscale*(U_sum_bceloss/U_sum_labels.shape[0])
        #print('U ones loss',U_sum_bceloss)
        
        
        ##############################  
        # f(sum U_i' = 0)  
        U_zeros=rescale_to_image_range(torch.sum(Uaps[0,:,:],dim=0),dimval=0)
        Up_sum_outputs,_ = model(U_zeros.view(-1,3,224,224),{'epsilon_norm':0.0})
        Up_sum_labels=torch.zeros((Up_sum_outputs.shape[0],labels.shape[1]),dtype=torch.float32).cuda()
        Up_sum_bceloss=self.bceloss(Up_sum_outputs,Up_sum_labels).sum()

        Up_sum_bceloss=self.upsumscale*(Up_sum_bceloss/Up_sum_labels.shape[0])

        #print('U zeros loss:',U_sum_bceloss)

        ##############################  

        #bceloss=self.bceloss(outputs,labels).sum(1)

        
        # #print(self._alpha*bceloss.item(),self._beta*normloss.item(),self._gamma*parent_margin_loss.item(),self._lambda*neg_margin_loss.item(), loss.item())
        norms=model.get_norms()
        normloss=torch.sum(norms)
        sumval=torch.sum(torch.linalg.vector_norm(norms,ord=float('inf'),dim=2,keepdim=True))

        #print(norms)
        # sumval=torch.sum(norms)
        # average=torch.tensor(norms.shape).squeeze().sum()
        # normloss=torch.sum(norms/average)
        #sumval=torch.sum(norms)
        
        orthloss,ind_outputs=self.compute_orthogonal_loss(model,target_class,use_selection_mask)
        orthloss=self.orthscale*orthloss
        #print('Orth loss:',orthloss)

        ind_loss=0.0
        for i_it,i_outputs in enumerate(ind_outputs):
            i_selection_mask=torch.eye((labels.shape[1]),dtype=torch.float32).cuda()
            i_labels=i_it*torch.ones((i_outputs.shape[0],labels.shape[1]),dtype=torch.float32).cuda()
            i_bceloss=torch.mul(self.bceloss(i_outputs,i_labels),i_selection_mask).sum()
            ind_loss+=i_bceloss
            #print('Individual loss:',i_bceloss)
        
        
        ind_loss=self.indscale*(ind_loss/(i_outputs.shape[0]*len(ind_outputs)))

        
        #print('Total ind loss:',ind_loss) 
        
        #orthloss=0.0
        #print(bceloss,orthloss)
        
        #loss=0.1*bceloss + 0.0*normloss + self.orthscale*orthloss
        #loss=bceloss #+ 100*pw_bceloss + U_sum_bceloss + Up_sum_bceloss + 1e4*orthloss + 0.5*ind_loss
        loss=bceloss + pw_bceloss + U_sum_bceloss + Up_sum_bceloss + orthloss + ind_loss
        #print(bceloss.item(),pw_bceloss.item(),U_sum_bceloss.item(),Up_sum_bceloss.item(),orthloss.item(),ind_loss.item(),normloss.item(),sumval.item())
        #print(', '.join(["{:.3f}".format(x) for x in [bceloss.item(),pw_bceloss.item(),U_sum_bceloss.item(),Up_sum_bceloss.item(),orthloss.item(),ind_loss.item(),normloss.item(),sumval.item()]]))

        #print(bceloss.item(),normloss.item(),orthloss.item(),sumval.item(),loss.item())
        
        
        #loss=bceloss

        # losses_dict={
        # 'bceloss':bceloss,
        #  'normloss':normloss,
        #  'orthloss':orthloss
        # }
        losses_dict={
        'bce_loss':bceloss,
        'pw_loss':pw_bceloss,
        'u_sum_loss':U_sum_bceloss,
        'up_sum_loss':Up_sum_bceloss,
        'orth_loss':orthloss,
        'ind_loss':ind_loss,
        'norm_loss':normloss,
        }
        # losses_dict={
        # 'bceloss':bceloss,
        # }


        return losses_dict,loss
        #return loss
