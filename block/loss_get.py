import torch


def loss_get(args):
    loss = loss_class(args)
    return loss


class loss_class:
    def __init__(self, args):
        self.loss = torch.nn.CrossEntropyLoss()

    def __call__(self, pred, true):
        loss = self.loss(pred, true)
        return loss
