from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import os
import shutil
import math
import time
import random
#import skimage
import numpy as np
#from skimage import io
#from skimage.transform import resize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.optimize import linear_sum_assignment

import cv2
from datetime import datetime
######################################
#       measurement functions        #
######################################
       
def get_model_parm_nums(model):
    total = sum([param.numel() for param in model.parameters()])
    total = float(total) / 1e6
    return total



######################################
#         basic functions            #
######################################

def load_checkpoint(args, running_file):

    model_dir = os.path.join(args.savedir, 'save_models')
    latest_filename = os.path.join(model_dir, 'latest.txt')
    model_filename = ''

    if args.evaluate is not None:
        model_filename = args.evaluate
    else:
        if os.path.exists(latest_filename):
            with open(latest_filename, 'r') as fin:
                model_filename = fin.readlines()[0].strip()
    loadinfo = "=> loading checkpoint from '{}'".format(model_filename)
    print(loadinfo)

    state = None
    if os.path.exists(model_filename):
        state = torch.load(model_filename, map_location='cpu')
        loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    else:
        loadinfo2 = "no checkpoint loaded"
    print(loadinfo2)
    running_file.write('%s\n%s\n' % (loadinfo, loadinfo2))
    running_file.flush()

    return state


def save_checkpoint(state, epoch, root, saveID, keep_freq=10):

    filename = 'checkpoint_%03d.pth' % epoch
    model_dir = os.path.join(root, 'save_models')
    model_filename = os.path.join(model_dir, filename)
    latest_filename = os.path.join(model_dir, 'latest.txt')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # write new checkpoint 
    torch.save(state, model_filename)
    with open(latest_filename, 'w') as fout:
        fout.write(model_filename)
    print("=> saved checkpoint '{}'".format(model_filename))

    # remove old model
    if saveID is not None and (saveID + 1) % keep_freq != 0:
        filename = 'checkpoint_%03d.pth' % saveID
        model_filename = os.path.join(model_dir, filename)
        if os.path.exists(model_filename):
            os.remove(model_filename)
            print('=> removed checkpoint %s' % model_filename)

    print('##########Time##########', time.strftime('%Y-%m-%d %H:%M:%S'))
    return epoch


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        #self.sum += val * n
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch, args):
    method = args.lr_type
    if method == 'cosine':
        T_total = float(args.epochs)
        T_cur = float(epoch)
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * T_cur / T_total))
    elif method == 'multistep':
        lr = args.lr
        for epoch_step in args.lr_steps:
            if epoch >= epoch_step:
                lr = lr * 0.1

    #optimizer.param_groups[0]['lr'] = 2.5e-4
    #print("HERE ", len(optimizer.param_groups))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    str_lr = '%.6f' % lr
    return str_lr


######################################
#     edge specific functions        #
######################################

def cross_entropy_loss_HED(prediction, labelf):

    total_loss = 0
    batch, channel_num, imh, imw = labelf.size()
    
    for b_i in range(batch):
        target_new = labelf[b_i, :, :, :].unsqueeze(1).clone()
        p = prediction[b_i, :, :, :].unsqueeze(1)
        t = target_new
        mask = (t > 0.5).float()
        b, c, h, w = mask.shape
        num_pos = torch.sum(mask, dim=[1, 2, 3]).float()  # Shape: [b,].
        num_neg = c * h * w - num_pos  # Shape: [b,].
        class_weight = torch.zeros_like(mask)
        class_weight[t > 0.5] = num_neg / (num_pos + num_neg)
        class_weight[t <= 0.5] = num_pos / (num_pos + num_neg)
        # weighted element-wise losses
        loss = F.binary_cross_entropy(p, t.float(), weight=class_weight, reduction='sum')
        # do the reduction for the weighted loss
        #loss = weight_reduce_loss(loss, weight, reduction=reduction, avg_factor=avg_factor)
        total_loss = total_loss + loss
    return total_loss
  
  
def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    
    temp_ind = torch.argwhere(prediction >= 0.5).cuda()
    
    #print("num pos :", num_positive ," ", temp_ind.size(),flush=True)
    cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask, reduction='sum')

    return cost
    
def create_1to1_Label(pred, labelf,delta = 0.1,max_toleration = 9,img=None,delta_aux = -1):

    
    max_toleration = 8 #MODIFY this line for NYUD (8),BSDS (4) ,Multicue (11), and BIPED (11) in training. Does not affect inference.
    weight = 20        #MODIFY this line for NYUD (20),BSDS (5) ,Multicue (5), and BIPED (55) in training. Does not affect inference.
 
    gt = labelf.clone()
    B,C,W,H = pred.size()
    edge_pix_ind = (gt > 0)
    valid_gt_ind = torch.argwhere(gt == 1).cuda()
    valid_gt_ind_w = valid_gt_ind[:,2]
    valid_gt_ind_h = valid_gt_ind[:,3]
    
    valid_pred_ind = torch.argwhere(pred >= delta).cuda()    
    valid_pred_ind_w = valid_pred_ind[:,2]
    valid_pred_ind_h = valid_pred_ind[:,3]
    
    dist_mat_w = torch.abs(valid_pred_ind_w[:,None] - valid_gt_ind_w)#.cuda()
    dist_mat_h = torch.abs(valid_pred_ind_h[:,None] - valid_gt_ind_h)#.cuda()
        
    if len(valid_gt_ind) > 1 and len(valid_pred_ind) > 1:
        dist_mat = dist_mat_w + dist_mat_h - weight*pred.detach()[:,:,valid_pred_ind_w.ravel(),valid_pred_ind_h.ravel()].squeeze()[:,None]
                                           
    else:
        dist_mat = dist_mat_w + dist_mat_h
    
    dist_mat_spatial = dist_mat_w + dist_mat_h
    dist_mat[dist_mat_spatial > max_toleration] = 10000
    rids, cids = linear_sum_assignment(dist_mat.cpu())
    rids=torch.from_numpy(rids).cuda()
    cids=torch.from_numpy(cids).cuda()
    matching_cost = dist_mat[rids,cids]
    
    #max_toleration
    false_match = torch.argwhere(matching_cost>max_toleration).ravel().cuda()    
    false_match_index_rids = rids[false_match]
    false_match_pred = valid_pred_ind[false_match_index_rids]
    false_match_pred = false_match_pred.view(-1,false_match_pred.size()[-1])
    
    matched_pred = valid_pred_ind[rids]     #find matched pixel on prediction
    matched_gt = valid_gt_ind[cids]         #find matched pixel on gt
    new_gt = torch.zeros_like(gt).cuda()   #rids for pred, cids for gt
    
    new_gt[:,:,matched_pred[:,2], matched_pred[:,3]] = 1.0      #recreate gt based on matching pred pixels
    new_gt[:,:,false_match_pred[:,2], false_match_pred[:,3]] = 0.0      #remove false matching from a new gt
        
    
    false_match_index_cids = cids[false_match]
    false_match_gt = valid_gt_ind[false_match_index_cids]
    false_match_gt = false_match_gt.view(-1,false_match_gt.size()[-1])
    gt[:,:,matched_gt[:,2],matched_gt[:,3]]= 0
    gt[:,:,false_match_gt[:,2],false_match_gt[:,3]]= 1
    new_gt[gt>0] = 1.0
    
    return new_gt

def cross_entropy_loss_RCF_1to1(prediction, labelf, beta, img= None, max_toleration=0):

    prediction,grad_coef = prediction
    orig_label = labelf.clone()
    labelf = create_1to1_Label(prediction,labelf,max_toleration=max_toleration)
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label>0).float()
    num_negative = torch.sum(label==0).float()

    mask[label >0 ] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    
    cost = F.binary_cross_entropy(
            prediction, labelf, weight=mask, reduction='sum')

    return cost


def bce_entropy(inp, target,img, oneToOne=False,delta = 0.8):

    total_loss = 0
    ##delta = 0.1
    distance_thresh = 8
    batch, channel_num, imh, imw = target.size()
    
    for b_i in range(batch):
    
        target_new = target[b_i, :, :, :].unsqueeze(1).clone()
        if oneToOne:
            _,_,w,h = target.size()            
            
            h_q  = w // 4
            w_q  = h // 4    
            
            p = inp[b_i, :, :, :].unsqueeze(1)
            t = target[b_i, :, :, :].unsqueeze(1)            
            target_new[:, :, 0:w_q, 0:h_q] = create_1to1_Label(p[:, :, 0:w_q, 0:h_q], t[:, :, 0:w_q, 0:h_q], delta, distance_thresh, img)
            target_new[:, :, w_q:2*w_q, 0:h_q] = create_1to1_Label(p[:, :, w_q:2*w_q, 0:h_q], t[:, :, w_q:2*w_q, 0:h_q], delta, distance_thresh, img)
            target_new[:, :, 2*w_q:3*w_q, 0:h_q] = create_1to1_Label(p[:, :, 2*w_q:3*w_q, 0:h_q], t[:, :, 2*w_q:3*w_q, 0:h_q], delta, distance_thresh, img)
            target_new[:, :, 3*w_q:, 0:h_q] = create_1to1_Label(p[:, :, 3*w_q:, 0:h_q], t[:, :, 3*w_q:, 0:h_q], delta, distance_thresh, img)
            #
            ## Row 1
            target_new[:, :, 0:w_q, h_q:2*h_q] = create_1to1_Label(p[:, :, 0:w_q, h_q:2*h_q], t[:, :, 0:w_q, h_q:2*h_q], delta, distance_thresh, img)
            target_new[:, :, w_q:2*w_q, h_q:2*h_q] = create_1to1_Label(p[:, :, w_q:2*w_q, h_q:2*h_q], t[:, :, w_q:2*w_q, h_q:2*h_q], delta, distance_thresh, img)
            target_new[:, :, 2*w_q:3*w_q, h_q:2*h_q] = create_1to1_Label(p[:, :, 2*w_q:3*w_q, h_q:2*h_q], t[:, :, 2*w_q:3*w_q, h_q:2*h_q], delta, distance_thresh, img)
            target_new[:, :, 3*w_q:, h_q:2*h_q] = create_1to1_Label(p[:, :, 3*w_q:, h_q:2*h_q], t[:, :, 3*w_q:, h_q:2*h_q], delta, distance_thresh, img)
            
            # Row 2
            target_new[:, :, 0:w_q, 2*h_q:3*h_q] = create_1to1_Label(p[:, :, 0:w_q, 2*h_q:3*h_q], t[:, :, 0:w_q, 2*h_q:3*h_q], delta, distance_thresh, img)
            target_new[:, :, w_q:2*w_q, 2*h_q:3*h_q] = create_1to1_Label(p[:, :, w_q:2*w_q, 2*h_q:3*h_q], t[:, :, w_q:2*w_q, 2*h_q:3*h_q], delta, distance_thresh, img)
            target_new[:, :, 2*w_q:3*w_q, 2*h_q:3*h_q] = create_1to1_Label(p[:, :, 2*w_q:3*w_q, 2*h_q:3*h_q], t[:, :, 2*w_q:3*w_q, 2*h_q:3*h_q], delta, distance_thresh, img)
            target_new[:, :, 3*w_q:, 2*h_q:3*h_q] = create_1to1_Label(p[:, :, 3*w_q:, 2*h_q:3*h_q], t[:, :, 3*w_q:, 2*h_q:3*h_q], delta, distance_thresh, img)
            
            # Row 3
            target_new[:, :, 0:w_q, 3*h_q:] = create_1to1_Label(p[:, :, 0:w_q, 3*h_q:], t[:, :, 0:w_q, 3*h_q:], delta, distance_thresh, img)
            target_new[:, :, w_q:2*w_q, 3*h_q:] = create_1to1_Label(p[:, :, w_q:2*w_q, 3*h_q:], t[:, :, w_q:2*w_q, 3*h_q:], delta, distance_thresh, img)
            target_new[:, :, 2*w_q:3*w_q, 3*h_q:] = create_1to1_Label(p[:, :, 2*w_q:3*w_q, 3*h_q:], t[:, :, 2*w_q:3*w_q, 3*h_q:], delta, distance_thresh, img)
            target_new[:, :, 3*w_q:, 3*h_q:] = create_1to1_Label(p[:, :, 3*w_q:, 3*h_q:], t[:, :, 3*w_q:, 3*h_q:], delta, distance_thresh, img) 
            
        p = inp[b_i, :, :, :].unsqueeze(1)

        t = target_new
        # weighted element-wise losses
        loss = F.binary_cross_entropy(p, t.float(),reduction='sum')
        total_loss = total_loss + loss
    return total_loss
    
######################################
#         debug functions            #
######################################

# no function currently
