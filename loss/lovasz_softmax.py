import torch
import torch.nn.functional as F
from itertools import filterfalse
from torch.autograd import Variable


def lovasz_grad(gt_sorted):


    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    p = len(gt_sorted)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_softmax(probas, labels, classes = 'present', per_image = False, ignore = None):
    """
      probas: [B, C, H, W] Variable, class probabilities at each prediction (softmax output of logits)
      labels: [B, H, W] Tensor, ground truth labels (0 <= labels < C)
      classes: 'present' for classes present in labels, 'all' for all classes, or a list of classes to average.
      per_image: compute the loss per image instead of per batch
      ignore: void class labels
    """

    if per_image:
        loss = mean(lovasz_softmax_flat(*flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore))
                    for prob, lab in zip(probas, labels))
    else:
        loss = lovasz_softmax_flat(*flatten_probas(probas, labels, ignore))
    return loss

def flatten_probas(probas, labels, ignore=None):


    if ignore is None:
        return probas.view(-1, probas.size(1)), labels.view(-1)
    valid = labels != ignore
    vprobas = probas.view(-1, probas.size(1))
    vlabels = labels.view(-1)
    return vprobas[valid], vlabels[valid]

def lovasz_softmax_flat(probas, labels):


    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if fg.sum() == 0:
            continue
        if C == 1:
            if len(probas.shape) == 1:
                probas = probas.unsqueeze(0)
            fg = fg.unsqueeze(1)
        prob_fg = probas[:, c]
        errors = (Variable(fg) - prob_fg).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, Variable(lovasz_grad(fg_sorted))))
    return mean(losses)

def mean(l, ignore_nan = False, empty = 0):


    l = iter(l)
    if ignore_nan:
        l = filterfalse(torch.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n