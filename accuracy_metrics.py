import torch
from sklearn.metrics import classification_report
import numpy as np

def transform_pred(y_pred, threshold=0.65):
    # first separate between atmos and ground
    # for ground: construct a new y_pred with 1's where y_pred >= <threshold> and 0 otherwise
    # re-concatenate
    # we take a high threshold because the MultiLabelLoss favorizes high numbers for the prediction
    pred_ground = y_pred[:, 4:].clone().detach()
    pred_ground = torch.where(pred_ground>threshold, 1, 0)
    pred_atmos = y_pred[:, :4].clone().detach()
    pred_atmos = torch.where(pred_atmos>=pred_atmos.max(dim=1)[0].view(pred_atmos.shape[0], -1), 1, 0)
    new_pred = torch.cat((pred_atmos, pred_ground), dim=1)
    return new_pred


def Hamming_distance(y_pred, y, threshold=0.65):
    # number of "bits" we need to chane to get the correct prediction
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    res = (torch.abs(new_pred-y)==1).float().mean()
    return res

def overall_acc(y_pred, y, threshold=0.65):
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    return (new_pred==y).float().mean()

def count_false(y_pred, y, threshold=0.65):
    y_pred = transform_pred(y_pred, threshold)
    false_positive = torch.zeros(y_pred.shape)
    false_negative = torch.zeros(y_pred.shape)
    false_positive += torch.where((y_pred-y)==1, 1, 0) # predict 1 when you should predict 0
    false_negative += torch.where((y_pred-y)==-1, 1, 0) # predict 0 but should have predicted 1
    return false_positive.numpy(), false_negative.numpy()

