import torch
import torch.nn.functional as F
from torch import nn
from .utils import conv_variance


class FTLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()

        
    def ftt2_t(self, tensor, mask=None):

        threshold = 0.1
        iter_thresh = 2

        ft2 = torch.fft.fft2(tensor)
        ft2_abs = torch.abs(torch.fft.fftshift(ft2))
        dims = ft2_abs.shape
        
        if mask is None:
            mask = torch.zeros_like(ft2_abs)
        
            for k in range(iter_thresh):
                ft2_abs = ft2_abs.reshape((ft2_abs.shape[0],ft2_abs.shape[1]*ft2_abs.shape[2]))
                values, _ = ft2_abs.max(dim=1)
                mean_value = values.mean()
                ft2_abs = ft2_abs.reshape(dims)
                
                mask[ft2_abs>(threshold*mean_value)] = 1
                ft2_abs[ft2_abs>(threshold*mean_value)] = 0#(threshold*mean_value)
        else:
            ft2_abs[mask.bool()]=0

        return ft2_abs, mask


    def ftt2(self, tensor):

        ft2 = torch.fft.fft2(tensor)
        ft2_abs = (torch.fft.fftshift(ft2))
        
        return ft2_abs

    def forward(self, mask, output, gt):
        loss_dict = {
            'ft': 0.0
        } 

        # calculate loss for all channels
        for channel in range(output.shape[1]):
            # only select first channel

          #  mask_ch = torch.unsqueeze(mask[:, channel, :, :], dim=1)
          #  gt_ch = torch.unsqueeze(gt[:, channel, :, :], dim=1)
          #  output_ch = torch.unsqueeze(output[:, channel, :, :], dim=1)
            mask_ch = mask[:, channel, :, :]
            gt_ch = gt[:, channel, :, :]
            output_ch = output[:, channel, :, :]

            #loss_dict['ft'] += self.l1(self.ftt2(output_ch), self.ftt2(gt_ch))

            ft_output = self.ftt2(output_ch)
            ft_gt = self.ftt2(gt_ch)

            loss_dict['ft'] += self.l1(torch.imag(ft_output), torch.imag(ft_gt))
            loss_dict['ft'] += self.l1(torch.real(ft_output), torch.real(ft_gt))
           
            if loss_dict['ft'].isnan().sum()>0:
                pass
        return loss_dict