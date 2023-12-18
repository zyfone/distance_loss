# The following code snippet is from https://github.com/SkyKuang/DGCAT/tree/main
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class Geometry_loss(nn.Module):
    def __init__(self, w_d=1, w_a=2):
        super(Geometry_loss, self).__init__()
        self.w_d = w_d
        self.w_a = w_a

    def forward(self, f_s, f_t):
        student = f_s.view(f_s.shape[0], -1)
        teacher = f_t.view(f_t.shape[0], -1)
        # pdb.set_trace()
        #Distance loss
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d > 0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d > 0].mean()
        d = d / mean_d

        loss_d = F.smooth_l1_loss(d, t_d)
     
        #Angle loss
        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss_a = F.smooth_l1_loss(s_angle, t_angle)
 
        loss = self.w_d * loss_d + self.w_a * loss_a

        return loss

    @staticmethod
    def pdist(e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        # prod = e @ e.t()
        prod = torch.matmul(e,e.t())
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res





# RPN MASK foreground 
mask_t = torch.sum(rpn_cls_map_t[:, 9:, :, :], dim=1)
mask_t[mask_t > 1.0] = 1.0
mask_t = (mask_t + 1.0) / 2
mask_t = mask_t.unsqueeze(0).detach()
mask_low_t = F.interpolate(mask_t, size=(low_features_t.shape[2], low_features_t.shape[3])).detach()
mask_mid_t = F.interpolate(mask_t, size=(mid_features_t.shape[2], mid_features_t.shape[3])).detach()
