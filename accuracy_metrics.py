import torch
from sklearn.metrics import classification_report
import numpy as np

def transform_pred(y_pred, threshold=3):
    # first separate between atmos and ground
    # for ground: construct a new y_pred with 1's where y_pred >= <threshold> and 0 otherwise
    # re-concatenate
    # we take a high threshold because the MultiLabelLoss favorizes high numbers for the prediction
    pred_ground = y_pred[:, 4:].clone().detach()
    pred_ground = torch.where(pred_ground>threshold, 1, 0)
    new_pred = torch.cat((y_pred[:, :4].clone().detach(), pred_ground), dim=1)
    return new_pred


def Hamming_distance(y_pred, y, threshold=3):
    # number of "bits" we need to chane to get the correct prediction
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    res = (torch.abs(new_pred-y)==1).float().mean()
    return res

def overall_acc(y_pred, y, threshold=3):
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    return (new_pred==y).float().mean()

def f1_score(y_pred, y, threshold=3):
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    res_atmos = classification_report(y_pred=new_pred[:, :4].detach().cpu(), y_true=y[:, :4].detach().cpu(), output_dict=True)
    res_ground = classification_report(y_pred=new_pred[:, 4:].detach().cpu(), y_true=y[:, 4:].detach().cpu(), output_dict=True)
    f1 = np.mean(res_atmos["f1-score"])+np.mean(res_ground["f1-score"]) # mean f1-score
    return f1

def Jaccard_index(y_pred, y, threshold=3):
    # "intersection over union" i.e. number of correct pred over number of true+pred labels
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    intersection = (new_pred==y).count_nonzero()
    union = (new_pred+y>0).count_nonzero()
    return intersection/union


