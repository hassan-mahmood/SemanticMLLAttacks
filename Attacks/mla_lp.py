# # @Time      :2019/10/26 11:26
# # @Author    :zhounan
# # @FileName  :mla_lp.py
# import numpy as np
# import logging
# import mosek
# import torch
# import gc
# from multiprocessing import Pool


# class MLLP(object):
#     def __init__(self, model):
#         self.model = model

#     def generate_np(self, x, A_m, kwargs):
#         logging.info('prepare attack')
#         self.clip_max = kwargs['clip_max']
#         self.clip_min = kwargs['clip_min']
#         y_target = kwargs['y_target']
#         max_iter = kwargs['max_iter']
#         x_shape = x.shape[1:]
#         y_target[y_target == -1] = 0

#         num_features = x_shape[0] * x_shape[1] * x_shape[2]
#         num_labels = y_target.shape[1]
#         num_instaces = y_target.shape[0]

#         logging.info('get label gradient')

#         # x_t = torch.FloatTensor(x)
#         # y_target_t = torch.FloatTensor(y_target)
#         x_t = x
#         y_target_t = torch.clone(y_target)

#         if torch.cuda.is_available():
#             x_t = x_t.cuda()
#             y_target_t = y_target_t.cuda()

#         # the probability and prediction of original batch x
#         output = self.model(x_t)[0]
#         pred = torch.clone(output)
#         pred[pred >= 0.5] = 1
#         pred[pred < 0.5] = 0

#         # the variabes save best adversarial examples and its output
#         best_adv_x = torch.clone(x)
#         best_pred = torch.clone(pred)
#         adv_pred = pred

#         iteration = 0
#         r_change = torch.zeros_like(x)
#         threshold_value = torch.ones((num_instaces, num_labels)) * 0.5

#         # Probability of attack label output after each attack, attack sample, attack output
#         attack_bool = (A_m == 1)
#         adv_output = torch.clone(output)  # shape (batch, n_label)
#         before_adv_output = torch.clone(adv_output[attack_bool])  # shape size (batch)
#         adv_x = torch.clone(x)  # shape (batch, channels, height, width)

#         error_idx = []
#         while iteration < max_iter:
#             if len(error_idx) == num_instaces:
#                 break
#             for i in range(num_instaces):
#                 if (best_pred[i] == y_target[i]).all() and (i not in error_idx):
#                     log_info = '    {}     iteration， example   {}  found attack sample,  rmsd:   {} , norm:  {} , max:    {}    , mean:    {}  '.format(
#                         iteration, i,
#                         torch.sqrt(torch.mean(torch.square(((best_adv_x[i] / 2 + 0.5) * 255) - ((x[i] / 2 + 0.5) * 255)).flatten())),
#                         torch.linalg.norm(best_adv_x[i] - x[i]),
#                         torch.max(torch.abs(best_adv_x[i] - x[i])),
#                         torch.mean(torch.abs(best_adv_x[i] - x[i])))

#                     error_idx.append(i)
#                     logging.info(log_info)

#             #x_t = torch.FloatTensor(adv_x)
#             x_t=adv_x
#             # if torch.cuda.is_available():
#             #     x_t = x_t.cuda()

#             # get jacobian matrix
#             # gradients shape [n_label, batch, x_features]
#             # output shape [batch, n_label]
            
#             # gradients, output = get_jacobian_loss(self.model, x_t, y_target_t, num_labels)

#             # gradient_samples shape [batch, n_label, x_features]
#             jac_grad, output = get_jacobian_loss(self.model, torch.clone(x_t).detach(), y_target_t, num_labels)
#             jac_grad = np.asarray(jac_grad).reshape((num_labels, num_instaces, num_features))
#             jac_grad = jac_grad.swapaxes(1, 0)
#             print('step 1')
#             # multi-process solving, one sample has one process
#             item_list = []
#             for i in range(num_instaces):
#                 item = []
#                 item.append(np.expand_dims(jac_grad[i], 0))
#                 item.append(np.expand_dims(output[i], 0))
#                 item.append(np.expand_dims(y_target[i].cpu().detach().numpy(), 0))
#                 item.append(threshold_value[i].numpy())
#                 item.append(torch.unsqueeze(r_change[i], dim=0).cpu().numpy())
#                 item_list.append(item)
#             with Pool(processes=num_instaces) as pool:
#                 result = pool.starmap(mosek_inner_point_solver, item_list)
            
#             # result=mosek_inner_point_solver(np.expand_dims(jac_grad[0], 0), np.expand_dims(output[0], 0), np.expand_dims(y_target[0].cpu().detach().numpy(), 0), np.expand_dims(threshold_value[0],0), r_change[0])
            
#             output=torch.from_numpy(output)
#             print('step 2')
#             for i in range(num_instaces):
#                 if i in error_idx:
#                     result[i] = torch.zeros((1, num_features))
#             result = torch.from_numpy(np.asarray(result))
#             result = result.reshape((num_instaces, x_shape[0], x_shape[1], x_shape[2]))
#             #temp_adv_x = np.clip(adv_x + result, a_min=self.clip_min, a_max=self.clip_max)
#             temp_adv_x = torch.clip(adv_x + torch.FloatTensor(result).cuda(), min=self.clip_min, max=self.clip_max)

#             # x_t = torch.FloatTensor(temp_adv_x)
#             x_t=temp_adv_x
#             if torch.cuda.is_available():
#                 x_t = x_t.cuda()

#             temp_adv_output = self.model(x_t)[0].detach()
#             print('step 3')
#             for i in range(num_instaces):
#                 if i in error_idx:
#                     continue

#                 if temp_adv_output[attack_bool][i] == output[attack_bool][i] and iteration >= 5:
#                     msg = 'example    {}  failed to solve'.format(i)
#                     logging.info(msg)
#                     error_idx.append(i)
#                     continue
#                 if torch.all(result[i] == 0):
#                     msg = 'example    {}  failed to solve'.format(i)
#                     logging.info(msg)
#                     error_idx.append(i)
#                     continue

#                 if (before_adv_output[i] <= temp_adv_output[attack_bool][i] or temp_adv_output[attack_bool][
#                     i] < 0.6) and threshold_value[attack_bool][i] > 0.1:
#                     temp = threshold_value[attack_bool]
#                     temp[i] = temp[i] - 0.05
#                     threshold_value[attack_bool] = temp[i]
#             print('step 4')
#             adv_x = temp_adv_x
#             # r_change = adv_x[i] - x[i]
#             adv_output = temp_adv_output
#             before_adv_output = temp_adv_output[attack_bool]

#             pred_temp = torch.clone(adv_output)
#             pred_temp[pred_temp >= 0.5] = 1
#             pred_temp[pred_temp < 0.5] = 0
#             adv_pred = pred_temp


#             # print('Prediction shape:',adv_pred.shape,y_target.shape)

#             print('step 5')
#             for i in range(num_instaces):
#                 if i in error_idx:
#                     continue
#                 # print('Prediction shape:',adv_pred.shape,y_target.shape)
#                 print('value of i:',i)

#                 # print(adv_pred[i,:])
#                 # print(y_target[i,:].cpu().numpy())
#                 print(adv_pred[i,:]==y_target[i,:])

#                 eq_value_1=(adv_pred[i,:]==y_target[i,:]).float()
#                 eq_value_1=torch.sum(eq_value_1)

#                 eq_value_2=(best_pred[i,:].detach()==y_target[i,:]).float()
#                 eq_value_2=torch.sum(eq_value_2)


#                 # eq_value_1 = np.sum((adv_pred[i,:]==y_target[i,:].cpu().numpy()).astype(np.float32))
#                 # eq_value_2 = np.sum((best_pred[i,:]==y_target[i,:].cpu().numpy()).astype(np.float32))

#                 # eq_value_1 = np.sum((adv_pred[i,:] == y_target[i,:].cpu().numpy()).float())
#                 # eq_value_2 = np.sum((best_pred[i,:] == y_target[i,:].cpu().numpy()).float())
#                 if eq_value_1 > eq_value_2:
#                     best_adv_x[i] = adv_x[i]
#                     best_pred[i,:] = adv_pred[i,:]
#                 elif eq_value_1 == eq_value_2 and torch.linalg.vector_norm(adv_x[i] - x[i],2) < torch.linalg.vector_norm(best_adv_x[i] - x[i],2):
#                     best_adv_x[i] = adv_x[i]
#                     best_pred[i,:] = adv_pred[i,:]
#                 elif eq_value_1 < eq_value_2:
#                     adv_x[i] = x[i]
#                     best_adv_x[i] = x[i]
#                     best_pred[i,:] = pred[i,:]
#             iteration = iteration + 1

#         return best_adv_x

# def get_jacobian_loss(model, x, y_target_t, noutputs):
#     num_instaces = x.size()[0]
#     v = torch.eye(noutputs).cuda()
#     jac = []

#     if torch.cuda.is_available():
#         x = x.cuda()
#     x.requires_grad = True
#     y = model(x)[0]
#     y = torch.clamp(y, 1e-6, 1 - 1e-6)
#     loss = -(y_target_t * torch.log(y) + (1 - y_target_t) * torch.log(1 - y))

#     retain_graph = True
#     for i in range(noutputs):
#         if i == noutputs - 1:
#             retain_graph = False
#         loss.backward(torch.unsqueeze(v[i], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
#         g = x.grad.cpu().detach().numpy()
#         # 梯度清零
#         x.grad.zero_()
#         jac.append(g)
#     jac = np.asarray(jac)
#     y = y.cpu().detach().numpy()
#     return jac, y


# def mosek_inner_point_solver(A, output, y_target, threshold_value, r_change):
#     """
#     Solving Linear Programming on a Sample use Mosek
#     the tutorial of Mosek on (https://docs.mosek.com/9.1/pythonapi/tutorial-lo-shared.html)
#     :param A: jac matrix
#         shape: [1, num_labels, input_dimension]
#     :param output: label confidence
#         shape: [1, num_labels]
#     :param y_target: target label
#         shape: [1, num_labels]
#         value: {0, 1}
#     :param threshold_value: target label confidence, default is threshold(0.5)
#         shape: [1, num_labels]
#         value: 0.5
#     :param r_change:
#     :return:
#     """
#     num_labels = y_target.shape[-1]
#     num_instances = A.shape[0]
#     d = A.shape[-1]
#     inf = 0.0

#     output = np.clip(output, 1e-6, 1.0 - (1e-6))
#     loss_current = -(y_target * np.log(output) + (1 - y_target) * np.log(1 - output))
#     loss_target = -(y_target * np.log(np.ones((num_labels)) * threshold_value) + (1 - y_target) * np.log(
#         1 - np.ones((num_labels)) * threshold_value))
#     delta_loss = loss_target - loss_current
#     # r_change = r_change.reshape(num_instances, d)
#     print('step 11')
#     tasks = []
#     with mosek.Env() as env:
#         for i in range(num_instances):
#             task = mosek.Task(env, 0, 0)
#             # task.set_Stream(mosek.streamtype.log, streamprinter)
#             # task.putintparam(mosek.iparam.intpnt_multi_thread, mosek.onoffkey.off)
#             A = A[i]

#             delta_loss = delta_loss[i]

#             # delete the full zero rows in jac matrix
#             delete_idx = []
#             # for j in range(num_labels):
#             #     if (A[j] == 0).all():
#             #         delete_idx.append(j)
#             # if len(delete_idx) != 0:
#             #     A = np.delete(A, delete_idx, axis=0)
#             #     delta_loss = np.delete(delta_loss, delete_idx, axis=0)
#             rows = A.shape[0]
#             # Set the boundry of all variables
#             # boundkey.fr [-inf, +inf]
#             # boundkey.lo [value, +inf]
#             # boundry for r1, r2,..., rd, z
#             # bkx: boundry type
#             # blx: low boundry
#             # blx: up boundry
#             print('step 12')
#             bkx = [mosek.boundkey.ra for i in range(d)] + [mosek.boundkey.lo]
#             blx = [-1 for i in range(d)] + [0.0]
#             bux = [1 for i in range(d)] + [1.]

#             # Set the boundry of all constraints
#             # boundkey.up [-inf, value]
#             # bkc: boundry type
#             # blc: low boundry
#             # blc: up boundry
#             bkc = [mosek.boundkey.up for i in range(rows + d * 2)]
#             blc = [-inf for i in range(rows + d * 2)]
#             buc = delta_loss.tolist() + [0.0 for i in range(d * 2)]

#             #  sparse matrix ordinal value, matrix stored by column.
#             # shape: n_x_feature, n_label
#             # [[0, 1, 2, ..., 19],
#             #   ...
#             # [0, 1, 2, ..., 19]]
#             zeros_to_rows = np.tile(np.arange(rows).reshape(1, -1), (d, 1))

#             # [20, 22, 24, ...,]
#             rows_to_rows_add_d = np.arange(start=rows, stop=rows + d * 2, step=2).reshape(-1, 1)
#             # [21, 23, 25, ...,]
#             rows_add1_to_rows_add_d = (rows_to_rows_add_d + 1).reshape(-1, 1)
#             B = np.c_[zeros_to_rows, rows_to_rows_add_d, rows_add1_to_rows_add_d]
#             asub = B.tolist()

#             ones_d = np.ones(d).reshape(-1, 1)
#             A = np.c_[A.T, ones_d, -ones_d]
#             aval = A.tolist()
#             print('step 13')
#             asub.extend([[i for i in range(rows, rows + d * 2)]])
#             aval.extend([[-1 for i in range(rows, rows + d * 2)]])
#             c = [0.0 for i in range(d)] + [1]
#             numvar = len(bkx)
#             numcon = len(bkc)
#             task.appendcons(numcon)
#             task.appendvars(numvar)

#             #garbage collection
#             del A
#             del B
#             del output
#             del zeros_to_rows
#             del rows_to_rows_add_d
#             del rows_add1_to_rows_add_d
#             gc.collect()
#             print('step 14')
#             for j in range(numvar):
#                 # Set the linear term c_j in the objective.
#                 task.putcj(j, c[j])
#                 task.putvarbound(j, bkx[j], blx[j], bux[j])
#                 task.putacol(j,  # Variable (column) index.
#                              asub[j],  # Row index of non-zeros in column j.
#                              aval[j])  # Non-zero Values of column j.
#             for j in range(numcon):
#                 task.putconbound(j, bkc[j], blc[j], buc[j])
            
#             task.putobjsense(mosek.objsense.minimize)
            
#             task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            
#             task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            

#             tasks.append(task)
#         print('step 15')
#         # if the solution fails, then result is full zero
#         result = np.zeros((num_instances, d))
#         for i in range(num_instances):
            
#             tasks[i].optimize()
#             # tasks[i].solutionsummary(mosek.streamtype.msg)
#             # Get status information about the solution
#             solsta = tasks[i].getsolsta(mosek.soltype.itr)
#             if (solsta == mosek.solsta.optimal):
#                 xx = [0.] * numvar
#                 tasks[i].getxx(mosek.soltype.itr,  # Request the basic solution.
#                                xx)
#                 xx = np.asarray(xx)
#                 result[i] = xx[:-1]
#             elif (solsta == mosek.solsta.dual_infeas_cer or
#                   solsta == mosek.solsta.prim_infeas_cer):
#                 pass 
#                 # print("Primal or dual infeasibility certificate found.\n")
#             elif solsta == mosek.solsta.unknown:
#                 print("Unknown solution status")
#             else:
#                 print("Other solution status")
#         print('step 16')
#     return torch.from_numpy(result)

# @Time      :2019/10/26 11:26
# @Author    :zhounan
# @FileName  :mla_lp.py
import numpy as np
import logging
import mosek
import torch
import gc
from multiprocessing import Pool


class MLLP(object):
    def __init__(self, model):
        self.model = model

    def generate_np(self, x, A_m, kwargs):
        logging.info('prepare attack')
        self.clip_max = kwargs['clip_max']
        self.clip_min = kwargs['clip_min']
        y_target = kwargs['y_target']
        max_iter = kwargs['max_iter']
        x_shape = x.shape[1:]
        y_target[y_target == -1] = 0

        num_features = x_shape[0] * x_shape[1] * x_shape[2]
        num_labels = y_target.shape[1]
        num_instaces = y_target.shape[0]

        logging.info('get label gradient')

        # x_t = torch.FloatTensor(x)
        # y_target_t = torch.FloatTensor(y_target)
        x_t = x
        y_target_t = y_target

        if torch.cuda.is_available():
            x_t = x_t.cuda()
            y_target_t = y_target_t.cuda()

        # the probability and prediction of original batch x
        output = torch.sigmoid(self.model(x_t)[0])
        pred = torch.clone(output)
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        # the variabes save best adversarial examples and its output
        best_adv_x = torch.clone(x)
        best_pred = torch.clone(pred).detach().cpu()
        adv_pred = pred

        iteration = 0
        r_change = torch.zeros_like(x)
        threshold_value = np.ones((num_instaces, num_labels)) * 0.5

        # Probability of attack label output after each attack, attack sample, attack output
        attack_bool = (A_m == 1)

        adv_output = torch.clone(output)  # shape (batch, n_label)
        before_adv_output=torch.clone(adv_output).detach().cpu().numpy()
        # before_adv_output = torch.clone(adv_output[attack_bool])  # shape size (batch)

        adv_x = torch.clone(x)  # shape (batch, channels, height, width)

        error_idx = []
        while iteration < max_iter:
            if len(error_idx) == num_instaces:
                break
            for i in range(num_instaces):
                if (best_pred[i].cpu() == y_target[i].cpu()).all() and (i not in error_idx):
                    log_info = '    {}     iteration， example   {}  found attack sample,  rmsd:   {} , norm:  {} , max:    {}    , mean:    {}  '.format(
                        iteration, i,
                        torch.sqrt(torch.mean(torch.square(((best_adv_x[i] / 2 + 0.5) * 255) - ((x[i] / 2 + 0.5) * 255)).flatten())),
                        torch.linalg.norm(best_adv_x[i] - x[i]),
                        torch.max(torch.abs(best_adv_x[i] - x[i])),
                        torch.mean(torch.abs(best_adv_x[i] - x[i])))

                    error_idx.append(i)
                    logging.info(log_info)

            #x_t = torch.FloatTensor(adv_x)
            x_t=adv_x
            # if torch.cuda.is_available():
            #     x_t = x_t.cuda()

            # get jacobian matrix
            # gradients shape [n_label, batch, x_features]
            # output shape [batch, n_label]
            
            # gradients, output = get_jacobian_loss(self.model, x_t, y_target_t, num_labels)

            # gradient_samples shape [batch, n_label, x_features]
            jac_grad, output = get_jacobian_loss(self.model, torch.clone(x_t).detach(), y_target_t, num_labels)
            jac_grad = np.asarray(jac_grad).reshape((num_labels, num_instaces, num_features))
            jac_grad = jac_grad.swapaxes(1, 0)

            # multi-process solving, one sample has one process
            item_list = []
            for i in range(num_instaces):
                item = []
                item.append(np.expand_dims(jac_grad[i], 0))
                item.append(np.expand_dims(output[i], 0))
                item.append(np.expand_dims(y_target[i].cpu().detach().numpy(), 0))
                item.append(threshold_value[i])
                item.append(torch.unsqueeze(r_change[i], dim=0).cpu().numpy())
                item_list.append(item)
            with Pool(processes=num_instaces) as pool:
                result = pool.starmap(mosek_inner_point_solver, item_list)
            
            # result=mosek_inner_point_solver(np.expand_dims(jac_grad[0], 0), np.expand_dims(output[0], 0), np.expand_dims(y_target[0].cpu().detach().numpy(), 0), np.expand_dims(threshold_value[0],0), r_change[0])
            

            for i in range(num_instaces):
                if i in error_idx:
                    result[i] = np.zeros((1, num_features))
            result = np.asarray(result)
            result = result.reshape((num_instaces, x_shape[0], x_shape[1], x_shape[2]))
            #temp_adv_x = np.clip(adv_x + result, a_min=self.clip_min, a_max=self.clip_max)
            temp_adv_x = torch.clip(adv_x + torch.FloatTensor(result).cuda(), min=self.clip_min, max=self.clip_max)

            # x_t = torch.FloatTensor(temp_adv_x)
            x_t=temp_adv_x
            if torch.cuda.is_available():
                x_t = x_t.cuda()

            temp_adv_output = torch.sigmoid(self.model(x_t)[0]).cpu().detach().numpy()

            for i in range(num_instaces):
                if i in error_idx:
                    continue

                # Temp attack bool for specific instance i
                temp_attack_bool=np.where(attack_bool[i,:].cpu().numpy()==1.0)[0]

                if np.all(temp_adv_output[i,:][temp_attack_bool] == output[i,:][temp_attack_bool]) and iteration >= 5:
                    msg = 'example    {}  failed to solve'.format(i)
                    # print(msg)
                    # logging.info(msg)
                    error_idx.append(i)
                    continue
                if np.all(result[i] == 0):
                    msg = 'example    {}  f ailed to solve'.format(i)
                    # print(msg)
                    # logging.info(msg)
                    error_idx.append(i)
                    continue

                if np.all(before_adv_output[i,:][temp_attack_bool] <= temp_adv_output[i,:][temp_attack_bool]) or (np.all(temp_adv_output[i,:][temp_attack_bool] < 0.6) and np.all(threshold_value[i,:][temp_attack_bool] > 0.1)):
                    temp = threshold_value[i,:][temp_attack_bool]
                    temp = temp - 0.05
                    threshold_value[i,:][temp_attack_bool] = temp

            adv_x = temp_adv_x
            # r_change = adv_x[i] - x[i]
            adv_output = temp_adv_output
            # before_adv_output = temp_adv_output[attack_bool.cpu().numpy()]

            pred_temp = np.copy(adv_output)
            pred_temp[pred_temp >= 0.5] = 1
            pred_temp[pred_temp < 0.5] = 0
            adv_pred = pred_temp


            # print('Prediction shape:',adv_pred.shape,y_target.shape)


            for i in range(num_instaces):
                if i in error_idx:
                    continue
                # print('Prediction shape:',adv_pred.shape,y_target.shape)
                # print('value of i:',i)

                # print(adv_pred[i,:])
                # print(y_target[i,:].cpu().numpy())
                # print(adv_pred[i,:]==y_target[i,:].cpu().numpy())

                eq_value_1=(adv_pred[i,:]==y_target[i,:].cpu().numpy()).astype(np.float32)
                eq_value_1=np.sum(eq_value_1)

                eq_value_2=(best_pred[i,:].detach().cpu().numpy()==y_target[i,:].cpu().numpy()).astype(np.float32)
                eq_value_2=np.sum(eq_value_2)


                # eq_value_1 = np.sum((adv_pred[i,:]==y_target[i,:].cpu().numpy()).astype(np.float32))
                # eq_value_2 = np.sum((best_pred[i,:]==y_target[i,:].cpu().numpy()).astype(np.float32))

                # eq_value_1 = np.sum((adv_pred[i,:] == y_target[i,:].cpu().numpy()).float())
                # eq_value_2 = np.sum((best_pred[i,:] == y_target[i,:].cpu().numpy()).float())
                if eq_value_1 > eq_value_2:
                    best_adv_x[i] = adv_x[i]
                    best_pred[i,:] = torch.from_numpy(adv_pred[i,:])
                elif eq_value_1 == eq_value_2 and torch.linalg.vector_norm(adv_x[i] - x[i],2) < torch.linalg.vector_norm(best_adv_x[i] - x[i],2):
                    best_adv_x[i] = adv_x[i]
                    best_pred[i,:] = torch.from_numpy(adv_pred[i,:])
                elif eq_value_1 < eq_value_2:
                    adv_x[i] = x[i]
                    best_adv_x[i] = x[i]
                    best_pred[i,:] = pred[i,:]
            iteration = iteration + 1

        return best_adv_x

def get_jacobian_loss(model, x, y_target_t, noutputs):
    num_instaces = x.size()[0]
    v = torch.eye(noutputs).cuda()
    jac = []

    if torch.cuda.is_available():
        x = x.cuda()
    x.requires_grad = True
    y = torch.sigmoid(model(x)[0])
    y = torch.clamp(y, 1e-6, 1 - 1e-6)
    loss = -(y_target_t * torch.log(y) + (1 - y_target_t) * torch.log(1 - y))

    retain_graph = True
    for i in range(noutputs):
        if i == noutputs - 1:
            retain_graph = False
        loss.backward(torch.unsqueeze(v[i], 0).repeat(num_instaces, 1), retain_graph=retain_graph)
        g = x.grad.cpu().detach().numpy()
        # 梯度清零
        x.grad.zero_()
        jac.append(g)
    jac = np.asarray(jac)
    y = y.cpu().detach().numpy()
    return jac, y


def mosek_inner_point_solver(A, output, y_target, threshold_value, r_change):
    """
    Solving Linear Programming on a Sample use Mosek
    the tutorial of Mosek on (https://docs.mosek.com/9.1/pythonapi/tutorial-lo-shared.html)
    :param A: jac matrix
        shape: [1, num_labels, input_dimension]
    :param output: label confidence
        shape: [1, num_labels]
    :param y_target: target label
        shape: [1, num_labels]
        value: {0, 1}
    :param threshold_value: target label confidence, default is threshold(0.5)
        shape: [1, num_labels]
        value: 0.5
    :param r_change:
    :return:
    """
    num_labels = y_target.shape[-1]
    num_instances = A.shape[0]
    d = A.shape[-1]
    inf = 0.0

    output = np.clip(output, 1e-6, 1.0 - (1e-6))
    loss_current = -(y_target * np.log(output) + (1 - y_target) * np.log(1 - output))
    loss_target = -(y_target * np.log(np.ones((num_labels)) * threshold_value) + (1 - y_target) * np.log(
        1 - np.ones((num_labels)) * threshold_value))
    delta_loss = loss_target - loss_current
    # r_change = r_change.reshape(num_instances, d)

    tasks = []
    with mosek.Env() as env:
        for i in range(num_instances):
            task = mosek.Task(env, 0, 0)
            # task.set_Stream(mosek.streamtype.log, streamprinter)
            # task.putintparam(mosek.iparam.intpnt_multi_thread, mosek.onoffkey.off)
            A = A[i]

            delta_loss = delta_loss[i]

            # delete the full zero rows in jac matrix
            delete_idx = []
            # for j in range(num_labels):
            #     if (A[j] == 0).all():
            #         delete_idx.append(j)
            # if len(delete_idx) != 0:
            #     A = np.delete(A, delete_idx, axis=0)
            #     delta_loss = np.delete(delta_loss, delete_idx, axis=0)
            rows = A.shape[0]
            # Set the boundry of all variables
            # boundkey.fr [-inf, +inf]
            # boundkey.lo [value, +inf]
            # boundry for r1, r2,..., rd, z
            # bkx: boundry type
            # blx: low boundry
            # blx: up boundry
            
            bkx = [mosek.boundkey.ra for i in range(d)] + [mosek.boundkey.lo]
            blx = [-1 for i in range(d)] + [0.0]
            bux = [1 for i in range(d)] + [1.]

            # Set the boundry of all constraints
            # boundkey.up [-inf, value]
            # bkc: boundry type
            # blc: low boundry
            # blc: up boundry
            bkc = [mosek.boundkey.up for i in range(rows + d * 2)]
            blc = [-inf for i in range(rows + d * 2)]
            buc = delta_loss.tolist() + [0.0 for i in range(d * 2)]

            #  sparse matrix ordinal value, matrix stored by column.
            # shape: n_x_feature, n_label
            # [[0, 1, 2, ..., 19],
            #   ...
            # [0, 1, 2, ..., 19]]
            zeros_to_rows = np.tile(np.arange(rows).reshape(1, -1), (d, 1))

            # [20, 22, 24, ...,]
            rows_to_rows_add_d = np.arange(start=rows, stop=rows + d * 2, step=2).reshape(-1, 1)
            # [21, 23, 25, ...,]
            rows_add1_to_rows_add_d = (rows_to_rows_add_d + 1).reshape(-1, 1)
            B = np.c_[zeros_to_rows, rows_to_rows_add_d, rows_add1_to_rows_add_d]
            asub = B.tolist()

            ones_d = np.ones(d).reshape(-1, 1)
            A = np.c_[A.T, ones_d, -ones_d]
            aval = A.tolist()

            asub.extend([[i for i in range(rows, rows + d * 2)]])
            aval.extend([[-1 for i in range(rows, rows + d * 2)]])
            c = [0.0 for i in range(d)] + [1]
            numvar = len(bkx)
            numcon = len(bkc)
            task.appendcons(numcon)
            task.appendvars(numvar)

            #garbage collection
            del A
            del B
            del output
            del zeros_to_rows
            del rows_to_rows_add_d
            del rows_add1_to_rows_add_d
            gc.collect()

            for j in range(numvar):
                # Set the linear term c_j in the objective.
                task.putcj(j, c[j])
                task.putvarbound(j, bkx[j], blx[j], bux[j])
                task.putacol(j,  # Variable (column) index.
                             asub[j],  # Row index of non-zeros in column j.
                             aval[j])  # Non-zero Values of column j.
            for j in range(numcon):
                task.putconbound(j, bkc[j], blc[j], buc[j])
            
            task.putobjsense(mosek.objsense.minimize)
            
            task.putintparam(mosek.iparam.optimizer, mosek.optimizertype.intpnt)
            
            task.putintparam(mosek.iparam.intpnt_basis, mosek.basindtype.never)
            

            tasks.append(task)

        # if the solution fails, then result is full zero
        result = np.zeros((num_instances, d))
        for i in range(num_instances):
            
            tasks[i].optimize()
            # tasks[i].solutionsummary(mosek.streamtype.msg)
            # Get status information about the solution
            solsta = tasks[i].getsolsta(mosek.soltype.itr)
            if (solsta == mosek.solsta.optimal):
                xx = [0.] * numvar
                tasks[i].getxx(mosek.soltype.itr,  # Request the basic solution.
                               xx)
                xx = np.asarray(xx)
                result[i] = xx[:-1]
            elif (solsta == mosek.solsta.dual_infeas_cer or
                  solsta == mosek.solsta.prim_infeas_cer):
                pass 
                # print("Primal or dual infeasibility certificate found.\n")
            elif solsta == mosek.solsta.unknown:
                print("Unknown solution status")
            else:
                print("Other solution status")
    return result
