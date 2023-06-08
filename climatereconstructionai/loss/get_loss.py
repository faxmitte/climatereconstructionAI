import torch

from .fft_loss import FTLoss
from .gauss_loss import GaussLoss
from .hole_loss import HoleLoss
from .valid_loss import ValidLoss
from .feature_loss import FeatureLoss
from .total_variation_loss import TotalVariationLoss
from .var_loss import VarLoss
from ..utils.featurizer import VGG16FeatureExtractor
from .. import config as cfg

class ModularizedFunction(torch.nn.Module):
    def __init__(self, forward_op):
        super().__init__()
        self.forward_op = forward_op

    def forward(self, *args, **kwargs):
        return self.forward_op(*args, **kwargs)


class CriterionParallel(torch.nn.Module):
    def __init__(self, criterion):
        super().__init__()
        if not isinstance(criterion, torch.nn.Module):
            criterion = ModularizedFunction(criterion)
        self.criterion = torch.nn.DataParallel(criterion)

    def forward(self, *args, **kwargs):
        multi_dict = self.criterion(*args, **kwargs)
        for key in multi_dict.keys():
            multi_dict[key] = multi_dict[key].mean()
        return multi_dict

class loss_criterion(torch.nn.Module):
    def __init__(self):
        super().__init__()
  
        self.criterions = torch.nn.ModuleDict()
        
        for loss, lambda_ in cfg.lambda_dict.items():
            if lambda_ > 0:
                if loss == 'style' or loss == 'prc':
                    criterion = FeatureLoss(VGG16FeatureExtractor()).to(cfg.device)
                elif loss == 'valid':
                    criterion = ValidLoss().to(cfg.device)
                elif loss == 'hole':
                    criterion = HoleLoss().to(cfg.device)
                elif loss == 'tv':
                    criterion = TotalVariationLoss().to(cfg.device)
                elif loss == 'var':
                    criterion = VarLoss().to(cfg.device)
                elif loss == 'ft':
                    criterion = FTLoss().to(cfg.device)
                elif loss == 'gauss':
                    criterion = GaussLoss().to(cfg.device)
                
                if not criterion in self.criterions.values():
                    self.criterions[loss] = criterion

    def forward(self, mask, output, gt):
       
        loss_dict = {}
        for loss, criterion in self.criterions.items():
            if loss=='gauss':
                l = criterion(mask[:, cfg.recurrent_steps, 0, :, :], output[:, cfg.recurrent_steps, :, :, :], gt[:, cfg.recurrent_steps, 0, :, :])
            else:
                if output.shape[2] != mask.shape[2]:
                    l = criterion(mask[:, cfg.recurrent_steps, :, :, :], output[:, cfg.recurrent_steps, [0], :, :], gt[:, cfg.recurrent_steps, :, :, :])
                else:
                    l = criterion(mask[:, cfg.recurrent_steps, :, :, :], output[:, cfg.recurrent_steps, :, :, :], gt[:, cfg.recurrent_steps, :, :, :])    
            loss_dict.update(l)

        loss_dict["total"] = 0
        for loss, lambda_value in cfg.lambda_dict.items():
            if lambda_value>0:
                loss_w_lambda = loss_dict[loss]*lambda_value
                loss_dict["total"] += loss_w_lambda
                loss_dict[loss] = loss_w_lambda.item()
        
        return loss_dict




class LossComputation():
    def __init__(self):
        super().__init__()
        if cfg.multi_gpus:
            self.criterion = CriterionParallel(loss_criterion())
        else:
            self.criterion = loss_criterion()


    def get_loss(self, img_mask, loss_mask, output, gt):

        mask = img_mask
        if loss_mask is not None:
            mask += loss_mask
            assert ((mask == 0) | (mask == 1)).all(), "Not all values in mask are zeros or ones!"

        loss_dict = self.criterion(mask, output ,gt)

        return loss_dict
