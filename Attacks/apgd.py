
   
# Copyright (c) 2020-present, Francesco Croce
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree
#
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np 
from autoattack.other_utils import L0_norm, L1_norm, L2_norm
from autoattack.checks import check_zero_gradients
from tqdm import tqdm 
import pickle 
import os 
seed=999
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

def L1_projection(x2, y2, eps1):
    '''
    x2: center of the L1 ball (bs x input_dim)
    y2: current perturbation (x2 + y2 is the point to be projected)
    eps1: radius of the L1 ball
    output: delta s.th. ||y2 + delta||_1 <= eps1
    and 0 <= x2 + y2 + delta <= 1
    '''

    x = x2.clone().float().view(x2.shape[0], -1)
    y = y2.clone().float().view(y2.shape[0], -1)
    sigma = y.clone().sign()
    u = torch.min(1 - x - y, x + y)
    #u = torch.min(u, epsinf - torch.clone(y).abs())
    u = torch.min(torch.zeros_like(y), u)
    l = -torch.clone(y).abs()
    d = u.clone()
    
    bs, indbs = torch.sort(-torch.cat((u, l), 1), dim=1)
    bs2 = torch.cat((bs[:, 1:], torch.zeros(bs.shape[0], 1).to(bs.device)), 1)
    
    inu = 2*(indbs < u.shape[1]).float() - 1
    size1 = inu.cumsum(dim=1)
    
    s1 = -u.sum(dim=1)
    
    c = eps1 - y.clone().abs().sum(dim=1)
    c5 = s1 + c < 0
    c2 = c5.nonzero().squeeze(1)
    
    s = s1.unsqueeze(-1) + torch.cumsum((bs2 - bs) * size1, dim=1)
    
    if c2.nelement != 0:
    
      lb = torch.zeros_like(c2).float()
      ub = torch.ones_like(lb) *(bs.shape[1] - 1)
      
      #print(c2.shape, lb.shape)
      
      nitermax = torch.ceil(torch.log2(torch.tensor(bs.shape[1]).float()))
      counter2 = torch.zeros_like(lb).long()
      counter = 0
          
      while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2.)
        counter2 = counter4.type(torch.LongTensor)
        
        c8 = s[c2, counter2] + c[c2] < 0
        ind3 = c8.nonzero().squeeze(1)
        ind32 = (~c8).nonzero().squeeze(1)
        #print(ind3.shape)
        if ind3.nelement != 0:
            lb[ind3] = counter4[ind3]
        if ind32.nelement != 0:
            ub[ind32] = counter4[ind32]
        
        #print(lb, ub)
        counter += 1
        
      lb2 = lb.long()
      alpha = (-s[c2, lb2] -c[c2]) / size1[c2, lb2 + 1] + bs2[c2, lb2]
      d[c2] = -torch.min(torch.max(-u[c2], alpha.unsqueeze(-1)), -l[c2])
    
    return (sigma * d).view(x2.shape)





class APGDAttack():
    """
    AutoPGD
    https://arxiv.org/abs/2003.01690
    :param predict:       forward pass function
    :param norm:          Lp-norm of the attack ('Linf', 'L2', 'L0' supported)
    :param n_restarts:    number of random restarts
    :param n_iter:        number of iterations
    :param eps:           bound on the norm of perturbations
    :param seed:          random seed for the starting point
    :param loss:          loss to optimize ('ce', 'dlr' supported)
    :param eot_iter:      iterations for Expectation over Trasformation
    :param rho:           parameter for decreasing the step size
    """

    def __init__(
            self,
            predict,
            n_iter=100,
            norm='Linf',
            n_restarts=1,
            eps=1e-1,
            seed=0,
            loss='ce',
            eot_iter=1,
            rho=.75,
            topk=None,
            verbose=False,
            device=None,
            use_largereps=False,
            is_tf_model=False,
            logger=None):
        """
        AutoPGD implementation in PyTorch
        """
        
        self.model = predict
        self.n_iter = n_iter
        self.eps = eps
        self.norm = norm
        self.n_restarts = n_restarts
        self.seed = seed
        self.loss = loss
        self.eot_iter = eot_iter
        self.thr_decr = rho
        self.topk = topk
        self.verbose = verbose
        self.device = device
        self.use_rs = True
        #self.init_point = None
        self.use_largereps = use_largereps
        #self.larger_epss = None
        #self.iters = None
        self.n_iter_orig = n_iter + 0
        self.eps_orig = eps + 0.
        self.is_tf_model = is_tf_model
        self.y_target = None
        self.logger = logger
    
    def init_hyperparam(self, x):
        assert self.norm in ['Linf', 'L2', 'L1']
        assert not self.eps is None

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()
        
        ### set parameters for checkpoints
        self.n_iter_2 = max(int(0.22 * self.n_iter), 1)
        self.n_iter_min = max(int(0.06 * self.n_iter), 1)
        self.size_decr = max(int(0.03 * self.n_iter), 1)
    
    def check_oscillation(self, x, j, k, y5, k3=0.75):
        #loss_steps, j(current_iteration), k(self.n_iter2), loss_best
        t = torch.zeros(x.shape[1]).to(self.device)
        for counter5 in range(k):
          t += (x[j - counter5] > x[j - counter5 - 1]).float()

        return (t <= k * k3 * torch.ones_like(t)).float()

    def check_shape(self, x):
        return x if len(x.shape) > 0 else x.unsqueeze(0)

    def normalize(self, x):
        if self.norm == 'Linf':
            t = x.abs().view(x.shape[0], -1).max(1)[0]
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)

        elif self.norm == 'L1':
            try:
                t = x.abs().view(x.shape[0], -1).sum(dim=-1)
            except:
                t = x.abs().reshape([x.shape[0], -1]).sum(dim=-1)
            return x / (t.view(-1, *([1] * self.ndims)) + 1e-12)
    
    def lp_norm(self, x):
        if self.norm == 'L2':
            t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
            return t.view(-1, *([1] * self.ndims))

    def dlr_loss(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
            1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)

    #
    def attackstep(self,attacktype,grad1,norm1,grad2,norm2,orthgrad,orth_indices):

        # MLA_Alpha:
        newgrad1=grad1*norm1
        
        # MLA_Beta
        if 'beta' in attacktype:
            newgrad1+=grad2*norm2
            newgrad1=torch.sign(newgrad1)


        # GMLA
        elif attacktype=='gmla':
            newgrad1+=grad2*norm1

            newgrad1=torch.sign(newgrad1)
            if torch.count_nonzero(orth_indices)==0:    
                return newgrad1
            
            newgrad1[orth_indices]=orthgrad[orth_indices]#*norm1[orth_indices]

        
        return newgrad1
    
    def attack_single_run(self, x, y, targetclasses,nontargetclasses, mycriterion,allotherdata, use_target_indices,selection_mask, x_init=None):
        tau=-0.85
        attacktype=allotherdata['attacktype']
        scale_factor=allotherdata['scale']
        target_selection_mask=allotherdata['target_selection_mask']
        nontarget_selection_mask=allotherdata['nontarget_selection_mask']

        if len(x.shape) < self.ndims:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)


        x_adv=x.detach()
        # if self.norm == 'Linf':
        #     t = 2 * torch.rand(x.shape).to(self.device).detach() - 1
        #     x_adv = x + self.eps * torch.ones_like(x
        #         ).detach() * self.normalize(t)
        allorthindices=[]
        

        

        
        if not x_init is None:
            0/0
            x_adv = x_init.clone()
            
        # would be interesting but out of the focus. Thank you for bringing this to attention. we are oigng to add this to paper. 
        # Please notice that we implemented PGD known to be stronger attack[cite papers] as shown in literature. MIM is related to 
        # PGD. Relative performance is similar. FGSM we observe 
        # Thank you and we will add in explanation. 
        # Explanation
        # may be more like a suggestion. 
        # region proposed
        # we are focussing on these issues. why other context-aware are not but challenges are context-aware in our seeting is this
        # this would be a subject of different paper. We could not address in a single paper. Why not applicable?
        # uyes, we can leverage we thought about it and have ideas about it. How we integrate our method?
        
        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([self.n_iter, x.shape[0]]
            ).to(self.device)
        loss_best_steps = torch.zeros([self.n_iter + 1, x.shape[0]]
            ).to(self.device)
        acc_steps = torch.zeros_like(loss_best_steps)

        allnorms1=[]
        allnorms2=[]


        x_adv.requires_grad_()


        generator = torch.Generator().manual_seed(2147483647)
        
        grad = torch.zeros_like(x)
        for _ in range(self.eot_iter):
            with torch.enable_grad():

                #optimizer.zero_grad()
                logits = self.model(x_adv,{'epsilon_norm':0.0})[0]
                # print(logits.shape)
                
                loss_indiv = -1*mycriterion(logits, y.type(torch.cuda.FloatTensor))

                ####################################
                grad_dot_indices=torch.where(logits[1,:]>0)[0].detach().cpu().numpy().tolist()
                dotgrads=[]
                for s_g in grad_dot_indices:
                    dot_grad_temp=torch.autograd.grad(loss_indiv[1,s_g].sum(),[x_adv],retain_graph=True)[0].detach()[1,:,:,:].flatten().cpu()
                    dotgrads.append(dot_grad_temp)
                
                # tidx=grad_dot_indices.index(targetclasses[0])
                dotgrads=torch.stack(dotgrads,dim=1)
                # normvals=torch.linalg.vector_norm(dotgrads,ord=2,dim=0)[None,:]
                # dotgrads=dotgrads/(normvals+1e-9)
                
                # print(torch.matmul(dotgrads[:,tidx].flatten(),dotgrads))
                # print(normvals)
                ####################################
                

                #loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)
                loss_indiv=torch.mul(loss_indiv,selection_mask)
                #loss_indiv = torch.sum(loss_indiv,1)
                #loss_indiv = criterion_indiv(logits, y)
                
                #loss = loss_indiv.sum()
                loss=loss_indiv

                #grad=torch.autograd.grad(loss.sum(),x_adv,create_graph=True)[0]
                #print(torch.sum(grad),grad.shape)

                #loss.sum().backward(create_graph=True)
                #second_derivative = torch.autograd.grad(grad.sum(), x_adv)[0]
                #print(torch.sum(second_derivative))
                #optimizer.step()
                #0/0
                # print('Grad sum',torch.sum(x_adv.grad))
                #params=[x_adv.cuda()]
                # grads=[p.grad for p in params]
                # x_adv.hess=0.0
                #zs = [torch.randint(0, 2, p.size(), generator=generator).cuda() * 2.0 - 1.0 for p in params]  # Rademacher distribution {-1.0, 1.0}
                #h_zs = torch.autograd.grad(grad, x_adv, grad_outputs=zs, only_inputs=True, retain_graph=False)
                # for h_z, z, p in zip(h_zs, zs, params):
                #     p.hess += h_z * z / self.n_samples

                # print('Done')
                # print('Grad sum',torch.sum(x_adv.grad))

                
                # 0/0

                #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                #hassan * -1
                #loss.backward()

                
                #################################
                # grad1=torch.autograd.grad(loss[:,targetclasses].sum(), [x_adv],retain_graph=True)[0].detach()
                
                # grad1=grad1.view(grad1.shape[0],-1)
                # norm1=torch.linalg.vector_norm(grad1,dim=1,ord=2)[:,None]
                # grad1=grad1/(norm1+1e-9)

                alldotprods=[]
                

                # for nt in tqdm(nontargetclasses[:500]):
                #     tempgrads=torch.autograd.grad(loss[:,nontargetclasses[nt]].sum(), [x_adv],retain_graph=True)[0].detach()
                #     tempgrads=tempgrads.view(tempgrads.shape[0],-1)
                #     norm2=torch.linalg.vector_norm(tempgrads,dim=1,ord=2)[:,None]
                #     tempgrads=tempgrads/(norm2+1e-9)
                #     print('Tempgrad shape:',tempgrads.shape,', norm shape:',norm2.shape)
                #     dotprods=torch.mul(grad1,tempgrads).sum(1)
                #     alldotprods.append(dotprods.flatten().cpu().tolist())
                
                # for k in range(4):
                #     print(np.array(alldotprods)[:,k].tolist())
                # 0/0

                ########################
                # allgrads=[]
                # print(targetclasses)
                # print(target_selection_mask.shape)
                
                # for k in tqdm(range(target_selection_mask.shape[1])):
                #     grad1=torch.autograd.grad(loss[:,k].sum(), [x_adv],retain_graph=True)[0].detach().cpu().numpy()
                #     np.save('/mnt/raptor/hassan/bmvc/supp/1/'+str(k)+'.npy',grad1)


                # 0/0


                ########################
                # print(loss.shape)

                #grad1=torch.autograd.grad(loss[:,targetclasses].sum(), [x_adv],retain_graph=True)[0].detach()
                #grad2=torch.autograd.grad(loss[:,nontargetclasses].sum(), [x_adv])[0].detach()
                #torch.mul(loss,target_selection_mask)
                grad1=torch.autograd.grad(torch.mul(loss,target_selection_mask).sum(), [x_adv],retain_graph=True)[0].detach()
                grad2=torch.autograd.grad(torch.mul(loss,nontarget_selection_mask).sum(), [x_adv])[0].detach()

                #print(grad1.shape,grad2.shape)
                grad1=grad1.view(grad1.shape[0],-1)
                grad2=grad2.view(grad2.shape[0],-1)
                self.model.zero_grad()
                
                
                norm1=torch.linalg.vector_norm(grad1,dim=1,ord=2)[:,None]+1e-9
                norm2=torch.linalg.vector_norm(grad2,dim=1,ord=2)[:,None]+1e-9

                #print(torch.mean(norm1),torch.mean(norm2))
                #allnorms1.append(torch.mean(norm1).item())
                #allnorms2.append(torch.mean(norm2).item())
                #print(norm1.flatten().cpu().tolist(),'\n',norm2.flatten().cpu().tolist())
                grad1=grad1/norm1
                grad2=grad2/norm2

                dotprods=torch.mul(grad1,grad2).sum(1)
                #allnorms1.append(dotprods[0].item())
                #allnorms2+=dotprods.cpu().tolist()
                #allnorms1.append(dotprods[2].item())
                #allnorms2.append(dotprods[3].item())
                allnorms1.append(dotprods.cpu().tolist())
                
                # #newgrad1=newgrad1*norm1
                # #newgrad1=grad1
                # #print(grad1.shape,grad2.shape)
                # combined=torch.cat((grad1.unsqueeze(dim=2),grad2.unsqueeze(dim=2)),dim=2)
                # #print('Combined:',combined.shape)
                # u,s,v=torch.svd(combined,some=True )
                # #print(u.shape,s.shape,v.shape)
                # newvec=torch.matmul(u,v)
                # grad1=newvec[:,:,0].squeeze()
                # grad2=newvec[:,:,0].squeeze()
                # # #print(newvec.shape)
                # #print(torch.matmul(newvec[0,:,:].squeeze().T,newvec[0,:,:].squeeze()))
                # 0/0
                
                #newgrad1=newgrad1*norm1#+grad2*norm1/10
                #newgrad1=grad1*norm1+grad2*norm1
                # ########################
                orth_indices=torch.where(dotprods<=tau)[0]

                print(dotprods)
                ############
                dicttostore={
                'grads':dotgrads,
                'target':targetclasses,
                'dotval':dotprods[1],
                'gradindices':grad_dot_indices
                }
                pickle.dump(dicttostore,open('/mnt/raptor/hassan/cvpr24/rebut/-1','wb'))
                ############

                # print('orth indices:',orth_indices)
                # 0/0
                allorthindices.append(len(orth_indices))
                #######################
                #old version:
                #tempgrad=(grad1.T-torch.mul(grad2.T,torch.mul(grad2,grad1).sum(1))).T
                # normtemp=torch.linalg.vector_norm(tempgrad,dim=1,ord=2)[:,None]+1e-9
                # tempgrad=tempgrad/normtemp
                # orthgrad=torch.clone(tempgrad).detach()
                #######################
                tempgrad=(torch.sign(grad1.T)-torch.mul(grad2.T,torch.mul(grad2,torch.sign(grad1)).sum(1))).T
                normtemp=torch.linalg.vector_norm(tempgrad,dim=1,ord=float('inf'))[:,None]+1e-9
                tempgrad=tempgrad/normtemp
                orthgrad=torch.clone(tempgrad).detach()
                #######################
                
                #tempgrad=grad1
                #grad1=(grad1.T-torch.mul(grad2.T,torch.mul(grad2,grad1).sum(1))).T
                #grad1[orth_indices]=(grad1[orth_indices].T-torch.mul(grad2[orth_indices].T,torch.mul(grad2[orth_indices],grad1[orth_indices]).sum(1))).T
                #grad1[orth_indices]=tempgrad[orth_indices]
                #grad1=(grad1.T-torch.mul(grad2.T,torch.mul(grad2,grad1).sum(1))).T
                #norm1=torch.linalg.vector_norm(grad1,dim=1,ord=2)[:,None]
                # norm1=torch.linalg.vector_norm(grad1,dim=1,ord=float('inf'))[:,None]
                # norm2=torch.linalg.vector_norm(grad2,dim=1,ord=float('inf'))[:,None]
                
                # newgrad1=(grad1/norm1)*0.00004+(grad2/norm2)*0.00004
                #newgrad1=grad1/norm1#+grad2/norm2
                # ########################

                #norm1=norm1/len(targetclasses)
                #norm2=0.002*norm2

                #print('Norms:',norm1,norm2)

                newgrad1=self.attackstep(attacktype,grad1,norm1,grad2,norm2,orthgrad,orth_indices).unsqueeze(-1)
                # grad_sum=self.attackstep(attacktype,grad1,norm1,grad2,norm2,orthgrad,orth_indices)

                #newgrad1=grad1*norm1#+grad2*norm1
                #newgrad1=scale_factor*grad1*norm1+(1-scale_factor)*grad2*norm1
                # newgrad1[orth_indices]=grad1[orth_indices]*norm1[orth_indices]#+ grad2[orth_indices]*norm1[orth_indices]
                #newgrad1[orth_indices]=orthgrad[orth_indices]*norm1[orth_indices]

                # ########################
                # C=torch.cat((grad1.unsqueeze(-1),grad2.unsqueeze(-1)),dim=2)
                # D=torch.cat((torch.transpose(grad1.unsqueeze(-1),1,2),torch.transpose(grad2.unsqueeze(-1),1,2)),dim=1)
                # #print('C shape:',C.shape,', D shape:',D.shape)

                # C_T_C=torch.bmm(torch.transpose(C,1,2),C)

                # D_D_T=torch.bmm(D,torch.transpose(D,1,2))

                # CTC_inv = torch.linalg.pinv(C_T_C)
                # DDT_inv = torch.linalg.pinv(D_D_T)
                # mid = torch.bmm(DDT_inv,CTC_inv)

                # #grad_sum=(0.52*grad1+0.48*grad2).unsqueeze(-1)
                # # grad_sum[orth_indices]=orthgrad[orth_indices].unsqueeze(-1)
                # # newgrad1[orth_indices]=orthgrad[orth_indices].unsqueeze(-1)
                                    

                # out = torch.bmm(torch.transpose(C,1,2),newgrad1)
                # out = torch.bmm(mid,out)
                # out=torch.bmm(torch.transpose(D,1,2),out).squeeze(-1)
                
                # normtemp=torch.linalg.vector_norm(out,dim=1,ord=2)[:,None]+1e-9
                # tempgrad=out/normtemp
                
                # #newgrad1=tempgrad*norm1
                # #newgrad1[orth_indices]=orthgrad[orth_indices]*norm1[orth_indices]
                
                # #newgrad1[orth_indices]=tempgrad[orth_indices]*norm1[orth_indices]
                
                # #newgrad1[orth_indices]=tempgrad[orth_indices]*norm1[orth_indices]

                # #newgrad1=out.squeeze(-1)
                # #print(out.shape)
                # #A_inv=torch.bmm(torch.bmm(torch.transpose(D,1,2),DDT_inv),torch.bmm(CTC_inv,torch.transpose(C,1,2)))
                # #print(A_inv.shape)
                # #torch.matmul(D.T)
                # # ########################

                loss_indiv=loss_indiv.sum(1)
                loss=loss_indiv.sum()

                #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                grad += newgrad1.view(x_adv.shape).detach()

        
        
        grad /= float(self.eot_iter)
        grad_best = grad.clone()

        #acc = logits.detach().max(1)[1] == y 
        #acc = self.compute_accuracy(logits,y,targetclasses, nontargetclasses, use_target_indices,selection_mask)
        acc = self.compute_accuracy(logits,y,target_selection_mask)
        
        finalindices=-1*torch.ones_like(acc)

        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()


        alpha = 2. if self.norm in ['Linf', 'L2'] else 1. if self.norm in ['L1'] else 2e-2
        # step_size = alpha * self.eps * torch.ones([x.shape[0], *(
        #     [1] * self.ndims)]).to(self.device).detach()
        step_size = alpha * 0.006 * torch.ones([x.shape[0], *(
            [1] * self.ndims)]).to(self.device).detach()
        
        x_adv_old = x_adv.clone()
        counter = 0
        k = self.n_iter_2 + 0
        
        counter3 = 0

        loss_best_last_check = loss_best.clone()

        reduced_last_check = torch.ones_like(loss_best)
        n_reduced = 0

        n_fts = x.shape[-3] * x.shape[-2] * x.shape[-1]        
        u = torch.arange(x.shape[0], device=self.device)

        
        breakflag=False
        for i in range(self.n_iter):
            #print('Step',i)
            ### gradient step
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = torch.clone(x_adv).detach()

                a = 0.75 if i > 0 else 1.0

                if self.norm == 'Linf':
                    if attacktype=='gmla':
                        x_adv_1 = x_adv + step_size * grad
                        # print(torch.linalg.vector_norm(grad.flatten(start_dim=1),ord=float('inf'),dim=-1))
                        #x_adv_1 = x_adv + step_size * grad/torch.linalg.vector_norm(grad)
                    else:
                        x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1,
                        x - self.eps), x + self.eps), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(
                        x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a),
                        x - self.eps), x + self.eps), 0.0, 1.0)
                
                elif self.norm == 'L2':
                    0/0
                    x_adv_1 = x_adv + step_size * self.normalize(grad)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv) * a + grad2 * (1 - a)
                    x_adv_1 = torch.clamp(x + self.normalize(x_adv_1 - x
                        ) * torch.min(self.eps * torch.ones_like(x).detach(),
                        L2_norm(x_adv_1 - x, keepdim=True)), 0.0, 1.0)

                x_adv = torch.clone(x_adv_1 + 0.).detach()
                x_adv=x_adv.clamp(0., 1.)
            ### get gradient
            #x_adv.requires_grad_()
            x_adv.requires_grad=True 
            grad = torch.zeros_like(x)
            for _ in range(self.eot_iter):
                
                with torch.enable_grad():
                    # logits = self.model(x_adv,None, None)[0]
                    # loss_indiv = -1*mycriterion(logits, y.float())
                    # loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)
                    
                    # #loss_indiv = torch.sum(loss_indiv,1)
                    # loss = loss_indiv.sum()

                    # grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                    #print('\nsum:',torch.sum(x_adv))
                    logits = self.model(x_adv,{'epsilon_norm':0.0})[0]
                    
                    loss_indiv = -1*mycriterion(logits, y.type(torch.cuda.FloatTensor))
                    
                    ####################################
                    grad_dot_indices=torch.where(logits[1,:]>0)[0].detach().cpu().numpy().tolist()
                    dotgrads=[]
                    for s_g in grad_dot_indices:
                        dot_grad_temp=torch.autograd.grad(loss_indiv[1,s_g].sum(),[x_adv],retain_graph=True)[0].detach()[1,:,:,:].flatten().cpu()
                        dotgrads.append(dot_grad_temp)
                    
                    # tidx=grad_dot_indices.index(targetclasses[0])
                    dotgrads=torch.stack(dotgrads,dim=1)
                    # normvals=torch.linalg.vector_norm(dotgrads,ord=2,dim=0)[None,:]
                    # dotgrads=dotgrads/(normvals+1e-9)
                    
                    # print(torch.matmul(dotgrads[:,tidx].flatten(),dotgrads))
                    # print(normvals)
                    ####################################

                    #loss_indiv=torch.mul(loss_indiv,selection_mask).sum(1)
                    loss_indiv=torch.mul(loss_indiv,selection_mask)

                    #loss_indiv = torch.sum(loss_indiv,1)
                    #loss_indiv = criterion_indiv(logits, y)
                    
                    #loss = loss_indiv.sum()
                    loss=loss_indiv

                    #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                    #hassan * -1
                    #loss.backward()
                    
                    # grad1=torch.autograd.grad(loss[:,targetclasses].sum(), [x_adv],retain_graph=True)[0].detach()
                    # grad2=torch.autograd.grad(loss[:,nontargetclasses].sum(), [x_adv])[0].detach()
                    grad1=torch.autograd.grad(torch.mul(loss,target_selection_mask).sum(), [x_adv],retain_graph=True)[0].detach()
                    grad2=torch.autograd.grad(torch.mul(loss,nontarget_selection_mask).sum(), [x_adv])[0].detach()
                    
                    self.model.zero_grad()
                    #[25, 25, 22, 24, 27, 22, 25, 33, 41, 36, 42]

                    #print(grad1.shape,grad2.shape)
                    grad1=grad1.view(grad1.shape[0],-1)
                    grad2=grad2.view(grad2.shape[0],-1)
                    norm1=torch.linalg.vector_norm(grad1,dim=1,ord=2)[:,None]+1e-9
                    norm2=torch.linalg.vector_norm(grad2,dim=1,ord=2)[:,None]+1e-9

                    #allnorms1.append(torch.mean(norm1).item())
                    #allnorms2.append(torch.mean(norm2).item())

                    
                    grad1=grad1/norm1
                    grad2=grad2/norm2

                    #print(grad1.shape,grad2.shape)
                    #out=grad1*grad2
                    #print('out:',out.shape)

                    
                    dotprods=torch.mul(grad1,grad2).sum(1)
                    print(dotprods)
                    ############
                    dicttostore={
                    'grads':dotgrads,
                    'target':targetclasses,
                    'dotval':dotprods[1],
                    'gradindices':grad_dot_indices
                    }
                    pickle.dump(dicttostore,open('/mnt/raptor/hassan/cvpr24/rebut/'+str(i),'wb'))
                    ############
                    #print(dotprods.shape)
                    #0/0
                    #print(norm1.flatten().cpu().tolist(),'\n',norm2.flatten().cpu().tolist())
                    #print(dotprods.flatten().cpu().tolist())
                    allnorms1.append(dotprods.cpu().tolist())
                    #allnorms1.append(dotprods[2].item())
                    #allnorms2.append(dotprods[3].item())
                    #allnorms2+=dotprods.cpu().tolist()

                    
                    
                    # newgrad1=grad1

                    # # newgrad1=newgrad1*norm1#+grad2*norm1/10
                    # combined=torch.cat((grad1.unsqueeze(dim=2),grad2.unsqueeze(dim=2)),dim=2)
                    # #print('Combined:',combined.shape)
                    # u,s,v=torch.svd(combined,some=True )
                    # #print(u.shape,s.shape,v.shape)
                    # newvec=torch.matmul(u,v)
                    # grad1=newvec[:,:,0].squeeze()
                    # grad2=newvec[:,:,0].squeeze()
                    # # #print(newvec.shape)
                    # # #print(torch.matmul(newvec[0,:,:].squeeze().T,newvec[0,:,:].squeeze()))
                    # # 0/0
                    # ########################
                    # find indices for which dot prod is less than -0.5
                    orth_indices=torch.where(dotprods<=tau)[0]
                    #print('orth indices:',len(orth_indices))
                    allorthindices.append(len(orth_indices))

                    # tempgrad=(grad1.T-torch.mul(grad2.T,torch.mul(grad2,grad1).sum(1))).T
                    # normtemp=torch.linalg.vector_norm(tempgrad,dim=1,ord=2)[:,None]+1e-9
                    # tempgrad=tempgrad/normtemp
                    # orthgrad=torch.clone(tempgrad).detach()


                    tempgrad=(torch.sign(grad1.T)-torch.mul(grad2.T,torch.mul(grad2,torch.sign(grad1)).sum(1))).T
                    normtemp=torch.linalg.vector_norm(tempgrad,dim=1,ord=float('inf'))[:,None]+1e-9
                    tempgrad=tempgrad/normtemp
                    orthgrad=torch.clone(tempgrad).detach()

                    
                    #tempgrad=grad1
                    #grad1[orth_indices]=(grad1[orth_indices].T-torch.mul(grad2[orth_indices].T,torch.mul(grad2[orth_indices],grad1[orth_indices]).sum(1))).T
                    #print('orth indices:',orth_indices)
                    #grad1[orth_indices]=tempgrad[orth_indices]
                    #grad1=(grad1.T-torch.mul(grad2.T,torch.mul(grad2,grad1).sum(1))).T
                    #norm1=torch.linalg.vector_norm(grad1,dim=1,ord=2)[:,None]

                    # norm1=torch.linalg.vector_norm(grad1,dim=1,ord=float('inf'))[:,None]
                    # norm2=torch.linalg.vector_norm(grad2,dim=1,ord=float('inf'))[:,None]
                    #newgrad1=(grad1/norm1)*0.00004+(grad2/norm2)*0.00004
                    #newgrad1=grad1/norm1#+grad2/norm2
                    # ########################
                    
                    #newgrad1=grad1*norm1+grad2*norm2
                    #norm1=norm1/len(targetclasses)
                    #norm2=0.002*norm2

                    #print('Norms:',norm1,norm2)
                    

                    #newgrad1=grad1*norm1+grad2*norm1
                    #newgrad1=0.6*grad1*norm1+0.4*grad2*norm1
                    grad_sum=self.attackstep(attacktype,grad1,norm1,grad2,norm2,orthgrad,orth_indices).unsqueeze(-1)
                    #newgrad1=grad1*norm1#+grad2*norm1
                    #newgrad1=scale_factor*grad1*norm1+(1-scale_factor)*grad2*norm1
                    #newgrad1[orth_indices]=grad1[orth_indices]*norm1[orth_indices]# + grad2[orth_indices]*norm1[orth_indices]
                    #newgrad1[orth_indices]=tempgrad[orth_indices]*norm1[orth_indices]
                    #newgrad1[orth_indices]=orthgrad[orth_indices]*norm1[orth_indices]
                    

                    # ########################
                    # C=torch.cat((grad1.unsqueeze(-1),grad2.unsqueeze(-1)),dim=2)
                    # D=torch.cat((torch.transpose(grad1.unsqueeze(-1),1,2),torch.transpose(grad2.unsqueeze(-1),1,2)),dim=1)
                    # #print('C shape:',C.shape,', D shape:',D.shape)

                    # C_T_C=torch.bmm(torch.transpose(C,1,2),C)

                    # D_D_T=torch.bmm(D,torch.transpose(D,1,2))

                    # CTC_inv = torch.linalg.pinv(C_T_C)
                    # DDT_inv = torch.linalg.pinv(D_D_T)
                    # mid = torch.bmm(DDT_inv,CTC_inv)

                    # #grad_sum=(grad1+grad2).unsqueeze(-1)
                    # #grad_sum=(0.52*grad1+0.48*grad2).unsqueeze(-1)
                    # #grad_sum[orth_indices]=orthgrad[orth_indices].unsqueeze(-1)


                    # out = torch.bmm(torch.transpose(C,1,2),grad_sum)
                    # out = torch.bmm(mid,out)
                    # out=torch.bmm(torch.transpose(D,1,2),out).squeeze(-1)
                    
                    # #newgrad1=out.squeeze(-1)
                    # normtemp=torch.linalg.vector_norm(out,dim=1,ord=2)[:,None]+1e-9
                    # tempgrad=out/normtemp

                    #newgrad1=tempgrad*norm1
                    
                    #newgrad1[orth_indices]=orthgrad[orth_indices]*norm1[orth_indices]
                    
                    #newgrad1[orth_indices]=tempgrad[orth_indices]*norm1[orth_indices]

                    #newgrad1[orth_indices]=tempgrad[orth_indices]*norm1[orth_indices]
                    # grad_sum[orth_indices]=orthgrad[orth_indices].unsqueeze(-1)
                    ########################

                    #newgrad1=newgrad1*norm1#+grad2*norm1/10
                    #newgrad1=grad1*norm1+grad2*norm1

                    loss_indiv=loss_indiv.sum(1)
                    loss=loss_indiv.sum()

                    #grad += torch.autograd.grad(loss, [x_adv])[0].detach()
                    #grad += newgrad1.view(x_adv.shape).detach()
                    grad += grad_sum.view(x_adv.shape).detach()
            
            grad /= float(self.eot_iter)

            #pred = self.compute_accuracy(logits.detach(),y,targetclasses,nontargetclasses,use_target_indices,selection_mask)
            pred = self.compute_accuracy(logits.detach(),y,target_selection_mask)

            # if ((1-pred.float()).sum()==0):
            #     breakflag=True 

            toupdateindices=torch.ones_like(pred).cuda()
            for pvalidx,pval in enumerate(pred.cpu().numpy().tolist()):
                if finalindices[pvalidx]!=-1:
                    toupdateindices[pvalidx]=False 
                    continue
                if pval is True:
                    finalindices[pvalidx]=i

            if i%100==0:
                print(i,'Final indices:',finalindices)
            if(torch.sum(toupdateindices.float())==0):
                breakflag=True
            # print('Loss:',loss_indiv)
            # print(acc)
            #l=[]

            # for i in range(x_adv.shape[0]):
            #     l.append(torch.max(torch.abs(x_adv-x)))

            #print(l)

            #pred = logits.detach().max(1)[1] == y

            acc = torch.max(acc, pred)
            #print(i,'Acc:',acc)
            acc_steps[i + 1] = acc + 0
            #ind_pred = (pred == 0).nonzero().squeeze() hassan
            
            #ind_pred = (pred == 1).nonzero().squeeze()

            ind_pred=torch.logical_and(pred==1,toupdateindices.float()==1).nonzero().squeeze()

            #print('ind_pred:',ind_pred)
            x_best_adv[ind_pred] = x_adv[ind_pred].detach() + 0.
            #print(torch.sum(x_best_adv,dim=(1,2,3)))
            # if self.verbose:
            #     str_stats = ' - step size: {:.5f} - topk: {:.2f}'.format(
            #         step_size.mean(), topk.mean() * n_fts) if self.norm in ['L1'] else ''
            #     print('[m] iteration: {} - best loss: {:.6f} - robust accuracy: {:.2%}{}'.format(
            #         i, loss_best.sum(), acc.float().mean(), str_stats))
            #     #print('pert {}'.format((x - x_best_adv).abs().view(x.shape[0], -1).sum(-1).max()))
            
            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              #print('Loss best:',loss_best)
              #print('y1:',y1)
              loss_steps[i] = y1 + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              #print('Loss best:',loss_best)
              counter3 += 1

              if counter3 == k:
                  if self.norm in ['Linf', 'L2']:
                      fl_oscillation = self.check_oscillation(loss_steps, i, k,
                          loss_best, k3=self.thr_decr)
                      fl_reduce_no_impr = (1. - reduced_last_check) * (
                          loss_best_last_check >= loss_best).float()
                      fl_oscillation = torch.max(fl_oscillation,
                          fl_reduce_no_impr)
                      reduced_last_check = fl_oscillation.clone()
                      loss_best_last_check = loss_best.clone()
    
                      if fl_oscillation.sum() > 0:
                          ind_fl_osc = (fl_oscillation > 0).nonzero().squeeze()
                          step_size[ind_fl_osc] /= 2.0
                          n_reduced = fl_oscillation.sum()
    
                          x_adv[ind_fl_osc] = x_best[ind_fl_osc].clone()
                          grad[ind_fl_osc] = grad_best[ind_fl_osc].clone()

                      k = max(k - self.size_decr, self.n_iter_min)
                  
                  counter3 = 0
                  k = max(k - self.size_decr, self.n_iter_min)
            
            if breakflag:
                break 
        del x_adv

        
        alldots=[]
        for kidx,k in enumerate(np.stack(allnorms1,axis=1).tolist()):
            alldots.append(k[:finalindices[kidx]])
            #break 


        otherdata={
        'dots':alldots
        }
        #print(allnorms2)
        #print('\nLoss steps:\n',loss_steps)
        #print('Loss best final:',loss_best)
        return (x_best, acc, loss_best, x_best_adv,otherdata)


    
    def compute_accuracy(self,y_pred,y,target_selection_mask):
        # make it true or false. If the goal is achieved, it is true. Else false
        
        logits=y_pred.detach()
        logits=torch.where(logits>0,1,0)

        selection_mask=target_selection_mask
        selection_mask=selection_mask.cuda()

        acc=torch.mul(torch.eq(logits,y),selection_mask).sum(1)
        acc=torch.eq(acc,selection_mask.sum(1))
        
        return acc


    def perturb(self, x, y, targetclasses,nontargetclasses,mycriterion, attackdata_dict, selection_mask, best_loss=False, x_init=None):
        """
        :param x:           clean images
        :param y:           clean labels, if None we use the predicted labels
        :param best_loss:   if True the points attaining highest loss
                            are returned, otherwise adversarial examples
        """

        target_selection_mask=attackdata_dict['target_labels_selection_mask']
        nontarget_selection_mask=attackdata_dict['nontarget_labels_selection_mask']

        assert self.loss in ['ce', 'dlr'] #'ce-targeted-cfts'
        if not y is None and len(y.shape) == 0:
            x.unsqueeze_(0)
            y.unsqueeze_(0)
        self.init_hyperparam(x)

        x = x.detach().clone().float().to(self.device)
        
        y_pred = self.model(x,{'epsilon_norm':0.0})[0]
        adv = x.clone()
        #acc = self.compute_accuracy(y_pred,y,targetclasses,nontargetclasses,use_target_indices,selection_mask)
        acc = self.compute_accuracy(y_pred,y,target_selection_mask)
        acc=torch.zeros_like(acc) #hassan
        #loss = -1e10 * torch.ones_like(acc).float() #hassan
        loss = -1e10 * torch.ones_like(acc).float()
        
        if self.verbose:
            print('-------------------------- ',
                'running {}-attack with epsilon {:.5f}'.format(
                self.norm, self.eps),
                '--------------------------')
            print('initial accuracy: {:.2%}'.format(acc.float().mean()))
        
        
        startt = time.time()
        best_loss=False
        if not best_loss:
            #0/0    
            #print('Not best loss')
            #torch.random.manual_seed(self.seed)
            #torch.cuda.random.manual_seed(self.seed)

            for counter in range(self.n_restarts):
                ind_to_fool = (acc==0).nonzero().squeeze()
                #print('Indices to fool:',ind_to_fool)

                if len(ind_to_fool.shape) == 0:
                    ind_to_fool = ind_to_fool.unsqueeze(0)
                if ind_to_fool.numel() != 0:
                    x_to_fool = x[ind_to_fool].clone()
                    y_to_fool = y[ind_to_fool].clone()
                    
                    if not self.use_largereps:
                        res_curr = self.attack_single_run(x_to_fool, y_to_fool,targetclasses,nontargetclasses,mycriterion,allotherdata,use_target_indices,selection_mask)
                    else:
                        res_curr = self.decr_eps_pgd(x_to_fool, y_to_fool, epss, iters)
                    best_curr, acc_curr, loss_curr, adv_curr,otherdata = res_curr

                    # for l in range(adv_curr.shape[0]):
                    #     print(torch.min(torch.abs(x_to_fool[l]-adv_curr[l])),torch.max(torch.abs(x_to_fool[l]-adv_curr[l])))   
                    
                    #print(torch.min(torch.abs(x_to_fool-adv_curr)),torch.max(torch.abs(x_to_fool-adv_curr)))
                    #ind_curr = (acc_curr == 0).nonzero().squeeze() hassan
                    ind_curr = (acc_curr == 1).nonzero().squeeze()

                    acc[ind_to_fool[ind_curr]] = 0
                    adv[ind_to_fool[ind_curr]] = adv_curr[ind_curr].clone()
                    if self.verbose:
                        print('restart {} - robust accuracy: {:.2%}'.format(
                            counter, acc.float().mean()),
                            '- cum. time: {:.1f} s'.format(
                            time.time() - startt))

            return adv,otherdata

        else:
            adv_best = x.detach().clone()
            loss_best = torch.ones([x.shape[0]]).to(
                self.device) * (-float('inf'))
            for counter in range(self.n_restarts):
                best_curr, _, loss_curr, _ = self.attack_single_run(x, y,targetclasses,nontargetclasses,mycriterion,use_target_indices,selection_mask)
                ind_curr = (loss_curr > loss_best).nonzero().squeeze()
                adv_best[ind_curr] = best_curr[ind_curr] + 0.
                loss_best[ind_curr] = loss_curr[ind_curr] + 0.

                # if self.verbose:
                #     print('restart {} - loss: {:.5f}'.format(
                #         counter, loss_best.sum()))

            return adv_best

    def decr_eps_pgd(self, x, y, epss, iters, use_rs=True):
        assert len(epss) == len(iters)
        assert self.norm in ['L1']
        self.use_rs = False
        
        if not use_rs:
            x_init = None
        else:
            x_init = x + torch.randn_like(x)
            x_init += L1_projection(x, x_init - x, 1. * float(epss[0]))
        eps_target = float(epss[-1])
        if self.verbose:
            print('total iter: {}'.format(sum(iters)))
        for eps, niter in zip(epss, iters):
            if self.verbose:
                print('using eps: {:.2f}'.format(eps))
            self.n_iter = niter + 0
            self.eps = eps + 0.
            #
            if not x_init is None:
                x_init += L1_projection(x, x_init - x, 1. * eps)
            x_init, acc, loss, x_adv = self.attack_single_run(x, y, x_init=x_init)

        return (x_init, acc, loss, x_adv)

