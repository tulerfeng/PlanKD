import math
import copy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import numpy as np
import logging
from torchvision import transforms
from timm.models.resnet2 import resnet6e, resnet10_2, resnet6d
from torch.autograd import Variable
import torch.nn.init as init
import cv2

import matplotlib.cm as cm

import os

import matplotlib.pyplot as plt
from PIL import Image

torch.autograd.set_detect_anomaly(True)




class WP_Attention(nn.Module):
    def __init__(
        self,
        device,
        args
    ):
        super().__init__()
        self.device=device
        self.sigma = 3.0 
        self.bev_encoder = resnet6e(pretrained = False)
        self.args = args
        self.beta_r = self.args.beta_r
        
        self.wp_encoder = nn.Sequential(
            nn.Linear(2, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            
        )
        
        self.query_encoder = nn.Sequential(
            nn.Linear(7 + 256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
        )
        
        self.W_q = nn.Linear(256, 128).to(self.device)
        self.W_k = nn.Linear(64, 128).to(self.device)
        
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)

    def forward(self, wp, bev, measurement, actor_pos, actor_cnt):
        
        x = torch.linspace(-18, 18, 180)
        y = torch.linspace(-18, 18, 180)
        xx, yy = torch.meshgrid(x, y)
        coords = torch.stack([xx, yy]).to(self.device)
        coords = coords.repeat((bev.shape[0], 1, 1, 1))
        bev =  torch.cat((bev, coords), dim=1)
        
        bev, _ = self.bev_encoder(bev) 
        
        query = torch.cat((bev, measurement), dim = 1)
        query = self.query_encoder(query).unsqueeze(1)
        query = self.W_q(query)
            
        key = self.wp_encoder(wp)
        key = self.W_k(key)
        
        score = torch.matmul(query, key.transpose(1, 2))  
        score = score.squeeze(1) / torch.sqrt(torch.tensor(128)) 
        
        # waypoints: (batch_size, num_waypoints, 2)
        # reference_points: (batch_size, max_num_reference_points, 2)
        
        pairwise_dists = torch.cdist(wp.float(), actor_pos.float(), p=2) # shape: (batch_size, num_waypoints, max_num_reference_points)
        kernel_vals = torch.exp(- pairwise_dists ** 2 / (2 * self.sigma ** 2)) # shape: (batch_size, num_waypoints, max_num_reference_points)
        mask = (actor_pos.sum(dim=-1) != 0).unsqueeze(1)  # shape: (batch_size, 1, max_num_reference_points)
        kernel_vals *= mask.float()
        norm_vals = torch.sum(kernel_vals, dim=-1)  # shape: (batch_size, num_waypoints)
        
        score = torch.softmax(score, dim = 1)
        
        diff = norm_vals.unsqueeze(dim=2) - norm_vals.unsqueeze(dim=1)
        labels = torch.sign(diff)

        labels_flat = labels.flatten()
        
        score_diff = score.unsqueeze(dim=2) - score.unsqueeze(dim=1)
        score_flat = score_diff.flatten()

        
        rank_loss = torch.max(torch.zeros(len(score_flat)).to(self.device), -labels_flat * score_flat + 0.1).mean()
        
        
        entropy = torch.mean(torch.sum(score * torch.log(score + 1e-8), dim=1))
        
        
        
        return score, entropy, rank_loss
    

    
    
    
class IB_Discriminator(nn.Module):
    def __init__(
        self,
        device,
    ):
        super().__init__()
        self.device=device
        
        self.main = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 2),
        )
        
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        

    def forward(self, z):
        
        z = self.main(z)

        return z
    
    
class VIB(nn.Module):
    def __init__(
        self,
        device,
        base_model,
    ):
        super().__init__()
        self.device = device
        self.base_model = base_model 
        self.beta = 1e-3
        
        if self.base_model == 'interfuser':
            self.init_dim = 1296
        
        self.encoder = nn.Sequential(
            nn.Linear(self.init_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 3 + 2 * 7), 
        )
        
        
        self.pred_list = ['light_state', 'junction_state', 'stop_sign', 'is_vehicle_present', 
                          'is_pedestrian_present', 'brake', 'steer', 'throttle']
        
        self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        
    def reparametrize_n(self, mu, std, n=1):

        eps = Variable(std.data.new(std.size()).normal_().to(self.device))
        return mu + eps * std

    def forward_encoder(self, feature_list):
        
        all_feature = None
        for i in range(len(feature_list)):
            feature = torch.mean(feature_list[i], dim = 1)
            feature = torch.flatten(feature, 1)
            if all_feature is None:
                all_feature = feature
            else:
                all_feature = torch.cat((all_feature, feature), dim=1)
        
        statistics  = self.encoder(all_feature)
        mu = statistics[:, :256]
        std = F.softplus(statistics[:, 256:] - 5, beta = 1)
        
        z = self.reparametrize_n(mu, std)
        
        return mu, std, z   
    
    def get_gussian_prob(self, mu, std, z):

        pi = torch.tensor([3.141592653589793]).to(self.device)
        coef = 1 / (torch.sqrt(2 * pi * torch.clamp(std,1e-15,100)**2) + 1e-15)
        exp = torch.exp(-((z - mu)**2) / ((2 * std**2) + 1e-15))
        pdf = coef * exp
        pdf = torch.clamp(pdf, 1e-15, 1)
        if (pdf <= 0).any():
            import pdb
            pdb.set_trace()
        log_pdf = torch.log(pdf)
        log_pdf = log_pdf.mean()
        return log_pdf
    
    def get_loss(self, mu, std, logit, x):
        
        class_loss = 0
        st = 0
        ed = 0
        for p in self.pred_list:
            if p == 'light_state':
                ed += 3
            else:
                ed += 2
            
            class_loss += F.cross_entropy(logit[:, st:ed], torch.tensor(x[p])).div(math.log(2))    
            
            if p == 'light_state':
                st += 3
            else:
                st += 2
        
        
        info_loss = -0.5*(1+2*(std + 1e-15).log()-mu.pow(2)-std.pow(2)).sum(1).mean().div(math.log(2))              
        
        total_loss = class_loss + self.beta*info_loss
        
        return total_loss
    
    def forward(self, feature_list, x):
        
        mu, std, z = self.forward_encoder(feature_list)
        
        logit = self.decoder(z)
        
        total_loss = self.get_loss(mu, std, logit, x)

        return total_loss

    


class PlanKD(nn.Module):
    def __init__(
        self,
        device,
        teacher_model, 
        student_model,
        base_model,
        args,

    ):
        super().__init__()
        self.device=device
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.args = args
        
        if base_model[:10] == 'interfuser':
            self.base_model = 'interfuser'
            
        for name, param in self.teacher_model.named_parameters():
            param.requires_grad_(False)
        
        self.IB_discriminator =  IB_Discriminator(device = self.device)
        self.IB = VIB(device = self.device, base_model = self.base_model)
        self.wp_attention = WP_Attention(device = self.device, args = self.args)
    
        self.IB_optimizer = torch.optim.Adam(self.IB.parameters(), lr = 1e-4)
        
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.bce_loss = nn.BCELoss(reduction='mean').to(self.device)
        
        self.cam_exits = 0
        

    
    def train_IB(self, loss):
        
        self.IB_optimizer.zero_grad()
        loss.backward()
        self.IB_optimizer.step()
        
        return
    
    def get_cnn_features(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        if 'act1' in dir(backbone):
            x = backbone.act1(x)
        else:
            x = backbone.relu(x)
        x = backbone.maxpool(x)

        x_layer1 = backbone.layer1(x)
        x_layer2 = backbone.layer2(x_layer1)
        # x_layer3 = backbone.layer3(x_layer2)
        # x_layer4 = backbone.layer4(x_layer3)
        return x_layer2
    
    def get_cnn_features_last(self, backbone, x):
        x = backbone.conv1(x)
        x = backbone.bn1(x)
        if 'act1' in dir(backbone):
            x = backbone.act1(x)
        else:
            x = backbone.relu(x)
        x = backbone.maxpool(x)

        x_layer1 = backbone.layer1(x)
        x_layer2 = backbone.layer2(x_layer1)
        x_layer3 = backbone.layer3(x_layer2)
        x_layer4 = backbone.layer4(x_layer3)
        return x_layer3
    
    

    


    def forward(self, x):
        
        
        output_s = self.student_model(x)
        
        wp_g_loss = torch.tensor(0)
        

        if self.base_model == 'interfuser':
            feature_t_front = self.get_cnn_features(self.teacher_model.rgb_backbone, x['rgb'])
            feature_t_left = self.get_cnn_features(self.teacher_model.rgb_backbone, x['rgb_left'])
            feature_t_right = self.get_cnn_features(self.teacher_model.rgb_backbone, x['rgb_right'])
            feature_list_t = [feature_t_left, feature_t_front, feature_t_right]
            
            feature_s_front = self.get_cnn_features(self.student_model.rgb_backbone, x['rgb'])
            feature_s_left = self.get_cnn_features(self.student_model.rgb_backbone, x['rgb_left'])
            feature_s_right = self.get_cnn_features(self.student_model.rgb_backbone, x['rgb_right'])
            feature_list_s = [feature_s_left, feature_s_front, feature_s_right]
            
        
        IB_loss = 0
        if self.training:
            IB_loss_t = self.IB.forward([cur_f.detach() for cur_f in  feature_list_t], x)
            IB_loss_s = self.IB.forward([cur_f.detach() for cur_f in  feature_list_s], x)
            IB_loss = IB_loss_t + IB_loss_s

            self.train_IB(IB_loss)
        
        
        
        _, _, z_real = self.IB.forward_encoder(feature_list_t)
        _, _, z_fake = self.IB.forward_encoder(feature_list_s)
        
        z_g_loss = torch.abs(z_real - z_fake).mean()
        
        
        
        if self.args.loss_kind == 'att':
            if self.base_model == 'interfuser':
                att, entropy, rank_loss = self.wp_attention(x['gt'], x['bev'], x['measurements'], x['actor_pos'], x['actor_cnt'])
                
                return output_s, (wp_g_loss, z_g_loss, att, entropy, rank_loss, IB_loss)
