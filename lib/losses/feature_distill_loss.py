import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from torch import einsum
from lib.datasets.utils import gaussian_radius
from lib.datasets.utils import draw_umich_gaussian_torch
import torchvision.ops.roi_align as roi_align

def imitation_loss(input, target,choice='L1_loss'):
    if choice == 'L1_loss':
        diff = input - target
        loss = diff.abs()
    elif choice == 'L2_loss':
        diff = input - target
        loss = 0.5 * diff ** 2
    elif choice == 'smooth_l1_loss':
        loss = F.smooth_l1_loss(input, target, reduction='none', beta=0.05)
    else:
        raise NotImplementedError
    return loss

def compute_imitation_loss(input, target, weights):
    target = torch.where(torch.isnan(target), input, target)  # ignore nan targets
    loss = imitation_loss(input, target,choice='L2_loss')
    assert weights.shape == loss.shape[:-1]
    weights = weights.unsqueeze(-1)
    assert len(loss.shape) == len(weights.shape)
    loss = loss * weights
    return loss

def compute_backbone_l1_loss(features_preds, features_targets, target):
    feature_ditill_loss = 0.0
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):
            downsample_ratio = 2 ** (i + 3)

            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            mask = calculate_box_mask(feature_pred, target, downsample_ratio)  # mask [B, H, W]

            feature_pred = feature_pred.permute(0, *range(2, len(feature_pred.shape)), 1)
            feature_target = feature_target.permute(0, *range(2, len(feature_target.shape)), 1)
            batch_size = int(feature_pred.shape[0])
            positives = feature_pred.new_ones(*feature_pred.shape[:3])
            positives = positives * torch.any(feature_target != 0, dim=-1).float()
            positives = positives * mask.cuda()

            reg_weights = positives.float()
            pos_normalizer = positives.sum().float()
            reg_weights /= pos_normalizer

            pos_inds = reg_weights > 0
            pos_feature_preds = feature_pred[pos_inds]
            pos_feature_targets = feature_target[pos_inds]

            imitation_loss_src = compute_imitation_loss(pos_feature_preds,
                                                          pos_feature_targets,
                                                          weights=reg_weights[pos_inds])  # [N, M]

            imitation_loss = imitation_loss_src.mean(-1)
            imitation_loss = imitation_loss.sum() / 10
            feature_ditill_loss = feature_ditill_loss + imitation_loss

    else:
        raise NotImplementedError

    return feature_ditill_loss

def compute_backbone_weight_l1_loss(features_preds, features_targets, target,teacher_confidence,student_confidence,epoch,weight_mode='student_weight'):
    student_confidence = torch.clamp(student_confidence, min=0, max=1)
    teacher_confidence = torch.clamp(teacher_confidence, min=0, max=1)
    feature_ditill_loss = 0.0
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):
            downsample_ratio = 2 ** (i + 3)

            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            weight,mask = calculate_box_weight(feature_pred, target, downsample_ratio,teacher_confidence,student_confidence,epoch,weight_mode)

            feature_pred = feature_pred.permute(0, *range(2, len(feature_pred.shape)), 1)
            feature_target = feature_target.permute(0, *range(2, len(feature_target.shape)), 1)
            batch_size = int(feature_pred.shape[0])
            positives = feature_pred.new_ones(*feature_pred.shape[:3])
            positives = positives * torch.any(feature_target != 0, dim=-1).float()
            positives = positives * weight.cuda()

            nums = feature_pred.new_ones(*feature_pred.shape[:3])
            nums = nums * torch.any(feature_target != 0, dim=-1).float()
            nums = nums * mask.cuda()

            reg_weights = positives.float()
            pos_normalizer = nums.sum().float()
            reg_weights /= pos_normalizer

            pos_inds = reg_weights > 0
            pos_feature_preds = feature_pred[pos_inds]
            pos_feature_targets = feature_target[pos_inds]

            imitation_loss_src = compute_imitation_loss(pos_feature_preds,
                                                          pos_feature_targets,
                                                          weights=reg_weights[pos_inds])  # [N, M]

            imitation_loss = imitation_loss_src.mean(-1)
            # imitation_loss = imitation_loss.sum()
            imitation_loss = imitation_loss.sum() / 10
            feature_ditill_loss = feature_ditill_loss + imitation_loss

    else:
        raise NotImplementedError

    return feature_ditill_loss

def compute_backbone_roi_affinity_loss(features_preds, features_targets, target):
    feature_ditill_loss = 0.0
    resize_shape = [7,7]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)): # 1/8   1/16   1/32
            downsample_ratio = 2 ** (i + 3)
            
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]
            box2d_masked = calculate_box_masks(feature_pred, target, downsample_ratio)  # box2d_masked [Batch_num,x1,y1,x2,y2]
            
            # input  : feature_pred [B,C,H,W] box2d_masked [K,5] output_size  [28,28] aligned True 
            # return : [K,C,output_size[0],output_size[1]]
            roi_feature_target = roi_align(feature_target,box2d_masked,resize_shape,aligned=True)
            roi_feature_pred = roi_align(feature_pred,box2d_masked,resize_shape,aligned=True)
            
            K,_,_,_ = roi_feature_pred.shape
            
            roi_feature_target = roi_feature_target.reshape(K,-1)
            depth_affinity = torch.mm(roi_feature_target,roi_feature_target.permute(1,0))
            
            roi_feature_pred = roi_feature_pred.reshape(K,-1)
            rgb_affinity = torch.mm(roi_feature_pred,roi_feature_pred.permute(1,0))
            
            feature_ditill_loss = feature_ditill_loss + F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / (K*K)
    else:
        raise NotImplementedError

    return feature_ditill_loss


def compute_backbone_roi_uncertainty_affinity_loss(features_preds, features_targets, target,teacher_confidence,student_confidence):
    
    teacher_uncertainty = 1 / teacher_confidence[target['indices_all']!=0].detach()
    nums = teacher_uncertainty.shape[0]
    teacher_uncertainty_pow2 = torch.pow(teacher_uncertainty,2)
    teacher_uncertainty_pow2 = teacher_uncertainty_pow2.unsqueeze(0).repeat([nums,1])
    teacher_uncertainty_pow2 = teacher_uncertainty_pow2 + teacher_uncertainty_pow2.T
    
    student_uncertainty = 1 / student_confidence[target['indices_all']!=0].detach()
    nums = student_uncertainty.shape[0]
    student_uncertainty_pow2 = torch.pow(student_uncertainty,2)
    student_uncertainty_pow2 = student_uncertainty_pow2.unsqueeze(0).repeat([nums,1])
    student_uncertainty_pow2 = student_uncertainty_pow2 + student_uncertainty_pow2.T
    
    
    feature_ditill_loss = 0.0
    resize_shape = [7,7]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)): # 1/8   1/16   1/32
            downsample_ratio = 2 ** (i + 3)
            
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]
            box2d_masked = calculate_box_masks(feature_pred, target, downsample_ratio)  # box2d_masked [Batch_num,x1,y1,x2,y2]
            
            # input  : feature_pred [B,C,H,W] box2d_masked [K,5] output_size  [28,28] aligned True 
            # return : [K,C,output_size[0],output_size[1]]
            roi_feature_target = roi_align(feature_target,box2d_masked,resize_shape,aligned=True)
            roi_feature_pred = roi_align(feature_pred,box2d_masked,resize_shape,aligned=True)
            
            K,_,_,_ = roi_feature_pred.shape
            
            roi_feature_target = roi_feature_target.reshape(K,-1)
            depth_affinity = torch.mm(roi_feature_target,roi_feature_target.permute(1,0))
            depth_affinity = depth_affinity / teacher_uncertainty_pow2 + torch.log(teacher_uncertainty_pow2)
            
            roi_feature_pred = roi_feature_pred.reshape(K,-1)
            rgb_affinity = torch.mm(roi_feature_pred,roi_feature_pred.permute(1,0))
            # rgb_affinity = rgb_affinity / student_uncertainty_pow2 + torch.log(student_uncertainty_pow2)
            rgb_affinity = rgb_affinity / teacher_uncertainty_pow2 + torch.log(teacher_uncertainty_pow2)
            
            feature_ditill_loss = feature_ditill_loss + 0.5*F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / (K*K)
    else:
        raise NotImplementedError

    return feature_ditill_loss


def compute_backbone_resize_affinity_loss(features_preds, features_targets):
    feature_ditill_loss = 0.0
    resize_shape = features_preds[-1].shape[-2:]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):  # 1/8   1/16   1/32
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            B, C, H, W = feature_pred.shape
            feature_pred_down = F.interpolate(feature_pred, size=resize_shape, mode="bilinear")
            feature_target_down = F.interpolate(feature_target, size=resize_shape, mode="bilinear")

            feature_target_down = feature_target_down.reshape(B, C, -1)
            depth_affinity = torch.bmm(feature_target_down.permute(0, 2, 1), feature_target_down)

            feature_pred_down = feature_pred_down.reshape(B, C, -1)
            rgb_affinity = torch.bmm(feature_pred_down.permute(0, 2, 1), feature_pred_down)

            feature_ditill_loss = feature_ditill_loss + F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / B

    else:
        raise NotImplementedError

    return feature_ditill_loss


def compute_backbone_local_affinity_loss(features_preds, features_targets):
    feature_ditill_loss = 0.0
    local_shape = features_preds[-1].shape[-2:]
    if isinstance(features_preds, list):
        for i in range(len(features_preds)):  # 1/8   1/16   1/32
            feature_target = features_targets[i].detach()
            feature_pred = features_preds[i]

            B, _, H, W = feature_pred.shape
            feature_pred_q = rearrange(feature_pred, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
            feature_pred_k = rearrange(feature_pred, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])

            rgb_affinity = einsum('b i d, b j d -> b i j', feature_pred_q, feature_pred_k)

            feature_target_q = rearrange(feature_target, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])
            feature_target_k = rearrange(feature_target, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=local_shape[0], p2=local_shape[1])

            depth_affinity = einsum('b i d, b j d -> b i j', feature_target_q, feature_target_k)

            feature_ditill_loss = feature_ditill_loss + F.l1_loss(rgb_affinity, depth_affinity, reduction='mean') / B

    else:
        raise NotImplementedError

    return feature_ditill_loss

def calculate_box_masks(features_preds, target, downsample_ratio):
    B, C, H, W = features_preds.shape
    box2d_masked = torch.cat([torch.arange(B).unsqueeze(-1).unsqueeze(-1).repeat([1,50,1]).type(torch.float).cuda(),target['box2d_gt']/ downsample_ratio],-1)
    box2d_masked = box2d_masked[target['indices_all']!=0] 
    return box2d_masked

def calculate_box_mask(features_preds, target, downsample_ratio):
    B, C, H, W = features_preds.shape
    gt_mask = torch.zeros((B, H, W))

    for i in range(B):
        for j in range(target['obj_num'][i]):
            bbox2d_gt = target['box2d_gt'][i, j] / downsample_ratio

            left_top = bbox2d_gt[:2]
            right_bottom = bbox2d_gt[2:]

            left_top_x = int(left_top[0].item())
            left_top_y = int(left_top[1].item())

            right_bottom_x = int(right_bottom[0].item())
            right_bottom_y = int(right_bottom[1].item())

            gt_mask[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1
    return gt_mask

def calculate_box_weight(features_preds, target, downsample_ratio,teacher_confidence,student_confidence,epoch,weight_mode='student_weight'):  
    B, C, H, W = features_preds.shape
    weight = torch.zeros((B, H, W))
    mask = torch.zeros((B, H, W))
    
    for i in range(B):
        for j in range(target['obj_num'][i]):
            bbox2d_gt = target['box2d_gt'][i, j] / downsample_ratio
            # bbox2d_gt = target['box2d_gt_head'][i, j] / downsample_ratio

            left_top = bbox2d_gt[:2]
            right_bottom = bbox2d_gt[2:]

            left_top_x = int(left_top[0].item())
            left_top_y = int(left_top[1].item())

            right_bottom_x = int(right_bottom[0].item())
            right_bottom_y = int(right_bottom[1].item())
            
            if target['indices_all'][i,j] != 0:
                if weight_mode == 'teacher_weight':
                    weight[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = teacher_confidence[i,j].detach()
                elif weight_mode == 'student_weight':
                    weight[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1 - student_confidence[i,j].detach()
                else:
                    print('error')
            else:
                weight[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 0.01
                
            mask[i, left_top_y:right_bottom_y, left_top_x:right_bottom_x] = 1
    return weight,mask