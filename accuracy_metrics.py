import torch

def transform_pred(y_pred, threshold=3):
    # first separate between atmos and ground
    # for ground: construct a new y_pred with 1's where y_pred >= <threshold> and 0 otherwise
    # re-concatenate
    # we take a high threshold because the MultiLabelLoss favorizes high numbers for the prediction
    pred_ground = y_pred[:, 4:].clone().detach()
    pred_ground = torch.where(pred_ground>threshold, 1, 0)
    new_pred = torch.cat((y_pred[:, :4].clone().detach(), pred_ground), dim=1)
    return new_pred


def Hamming(y_pred, y, threshold=3):
    # number of correct labels over the total number of labels
    # first tranform the pred then we just count y_pred==y and take the mean
    new_pred = transform_pred(y_pred, threshold=threshold)
    return (new_pred==y).float().mean()

def Jaccard_index(y_pred, y, threshold=3):
    # "intersection over union" i.e. number of correct pred over number of true+pred labels
    new_pred = transform_pred(y_pred=y_pred, threshold=threshold)
    intersection = (new_pred==y).count_nonzero()
    union = (new_pred+y>0).count_nonzero()
    return intersection/union


