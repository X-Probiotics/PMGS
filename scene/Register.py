import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from utils.spring_utils.logger import logger
from utils.spring_utils.builder import MODEL
from utils.spring_utils.misc import param_size
from utils.spring_utils.transform import rot6d_to_rotmat, euler_to_quat, quat_to_rot6d, quat_to_rotmat
from scene.gaussian_model import get_expon_lr_func


@MODEL.register_module()
class Register(nn.Module):

    def __init__(self,INIT_R,INIT_T,INIT_S):
        super().__init__()
        self.name = type(self).__name__
        # self.cfg = cfg

        euler = torch.tensor(INIT_R, dtype=torch.float32) * torch.pi / 180
        quat = euler_to_quat(euler)
        rot6d = quat_to_rot6d(quat)

        self.r = nn.Parameter(rot6d, requires_grad=True)
        self.t = nn.Parameter(torch.tensor(INIT_T, dtype=torch.float32), requires_grad=True)
        self.s = nn.Parameter(torch.tensor(INIT_S, dtype=torch.float32), requires_grad=True)

        logger.info(f"{self.name} has {param_size(self)}M parameters")

    def training_setup(self,R_LR,T_LR,S_LR):
        l = [{
            'params': [self.r],
            'lr': R_LR,
            "name": "r"
        }, {
            'params': [self.t],
            'lr': T_LR,
            "name": "t"
        }, {
            'params': [self.s],
            'lr': S_LR,
            "name": "s"
        }]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.R_scheduler = get_expon_lr_func(lr_init = R_LR, lr_final = 0.007*5, max_steps = 10)
        self.T_scheduler = get_expon_lr_func(lr_init = T_LR, lr_final = 5e-4*3, max_steps = 10)

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "r":
                lr = self.R_scheduler(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "t":
                lr = self.T_scheduler(iteration)
                param_group['lr'] = lr

    @property
    def get_scale(self):
        return deepcopy(self.s).detach()

    def forward(self, xyz):
        R = rot6d_to_rotmat(self.r)

        # origin = torch.mean(xyz, dim=0, keepdim=True)
        # xyz = self.s * (xyz - origin)  #+ origin
        xyz = self.s * xyz #+ origin

        xyz = (R @ xyz.transpose(0, 1)).transpose(0, 1) + self.t.unsqueeze(0)

        # print('[R,T.S]:',self.r,self.t,self.s)
        # R_new = self.r
        # T_new = self.s
        # S_new = self.t
        # if iteration % 1000 == 0:
        #  print(R_new,T_new,S_new)

        return xyz
    
