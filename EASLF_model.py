import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from networks.cma import CMA
from networks.depth_decoder import DepthDecoder
from networks.seg_decoder import SegDecoder
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
from utils.depth_utils import BackprojectDepth, Project3D, disp_to_depth, SSIM, get_smooth_loss, ConvBlock,Conv3x3,upsample,\
    transformation_from_parameters
import numpy as np
import math
import matplotlib.pyplot as plt

class Net(nn.Module):
    def __init__(self, options):
        super(Net, self).__init__()
        self.opt = options
        self.num_cascade = 2
        mindisp = -0.47
        maxdisp = 1.55
        self.angRes = self.opt.angRes
        self.maxdisp = maxdisp
        self.mindisp = mindisp
        self.models = nn.ModuleDict({
            'encoder': ResnetEncoder(num_layers=self.opt.num_layers, pretrained=self.opt.pretrained,)})
        if not self.opt.no_cma:
            self.models.update({
                'decoder': CMA(self.models['encoder'].num_ch_enc, opt=self.opt)
            })

        else:
            self.models.update({
                'depth': DepthDecoder(self.models['encoder'].num_ch_enc,
                                      scales=self.opt.scales, opt=self.opt),
            })

            if self.opt.semantic_distil is not None:
                self.models['seg'] = SegDecoder(self.models['encoder'].num_ch_enc, scales=[0])

        self.ssim = SSIM()
        self.parameters_to_train = []
        for model in self.models:
            self.parameters_to_train += list(self.models[model].parameters())
        self.loss_functions = {}
        self.masking_functions = []

        self.loss_functions = {self.compute_Rec_Loss: self.opt.Rec_Loss}

        if self.opt.disparity_smoothness:
            self.loss_functions[self.compute_smoothness] = self.opt.disparity_smoothness

        if self.opt.semantic_distil:
            self.loss_functions[self.compute_semantic_distil] = self.opt.semantic_distil

        if self.opt.sgt:
            self.loss_functions[self.compute_sgt_loss] = self.opt.sgt



    def forward(self, inputs):
        losses = {}
        loss = 0
        outputs = self.compute_outputs(inputs)

        for loss_function, loss_weight in self.loss_functions.items():
            loss_type = loss_function.__name__
            losses[loss_type] = loss_function(inputs, outputs) * loss_weight

        for loss_type, value in losses.items():
            to_optim = value.mean()
            loss += to_optim

        losses["loss"] = loss
        for key, value in outputs.items():
            if key != 'loss':
                outputs[key] = value.data
        return losses, outputs

    def compute_outputs(self, inputs):
        outputs = {}
        features = {}
        features_enc = {}
        features_forward = []

        #lf_forward_list = [10, 13, 16, 20, 22, 24, 30, 31, 32, 37, 38, 39, 40, 41, 42, 43, 48, 49, 50, 56, 58, 60, 64, 67, 70]
        #num_features = 25
        lf_forward_list = [30, 31, 32, 39, 40, 41, 48, 49, 50]
        num_features = 9

        for i in lf_forward_list:
            # 加载图像并进行必要的预处理操作，例如转换为张量格式
            features_enc = self.models["encoder"](inputs[("lf_forward", i)])
            features_forward.append(features_enc)

        attention_fusion = AttentionFusion(num_features)
        features[0] = attention_fusion(features_forward)

        #features_forward[0] = self.models["encoder"](fused_feature)

        #features[0] = self.models["encoder"](inputs[("lf_forward", 40)])

        if not self.opt.no_cma:
            disp, seg = self.models['decoder'](features[0])
            outputs.update(disp)
            for s in self.opt.scales:
                if s > 0:
                    disp = F.interpolate(outputs[("disp", s)], (self.opt.height, self.opt.width), mode='bilinear', align_corners=False)
                else:
                    disp = outputs[("disp", s)]
                _, depth = disp_to_depth(disp, self.mindisp, self.maxdisp)
                outputs[("depth", 0, s)] = depth
            outputs.update(seg)
        else:
            if self.opt.semantic_distil is not None:
                seg = self.models["seg"](features[0])
                outputs.update(seg)

            outputs.update(self.models["depth"](features[0]))
            _, depth = disp_to_depth(outputs[("disp", 0)], self.mindisp, self.maxdisp)
            for s in self.opt.scales:
                if s > 0:
                    disp = F.interpolate(outputs[("disp", s)], (self.opt.height, self.opt.width), mode='bilinear',
                                         align_corners=False)
                else:
                    disp = outputs[("disp", s)]
                _, depth = disp_to_depth(disp, self.mindisp, self.maxdisp)
                outputs[("depth", 0, s)] = depth

        return outputs

    def compute_affinity(self, feature, kernel_size):
        pad = kernel_size // 2
        feature = F.normalize(feature, dim=1)
        unfolded = F.pad(feature, [pad] * 4).unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)
        feature = feature.unsqueeze(-1).unsqueeze(-1)
        similarity = (feature * unfolded).sum(dim=1, keepdim=True)
        eps = torch.zeros(similarity.shape).to(similarity.device) + 1e-9
        affinity = torch.max(eps, 2 - 2 * similarity).sqrt()
        return affinity

    def compute_sgt_loss(self, inputs, outputs):

        assert len(self.opt.sgt_layers) == len(self.opt.sgt_kernel_size)
        seg_target = inputs["seg_gt"]
        _, _, h, w = seg_target.shape
        total_loss = 0

        for s, kernel_size in zip(self.opt.sgt_layers, self.opt.sgt_kernel_size):
            pad = kernel_size // 2
            h = self.opt.height // 2 ** s
            w = self.opt.width // 2 ** s
            seg = F.interpolate(seg_target, size=(h, w), mode='nearest')
            center = seg
            padded = F.pad(center, [pad] * 4, value=-1)
            aggregated_label = torch.zeros(*(center.shape + (kernel_size, kernel_size))).to(center.device)
            for i in range(kernel_size):
                for j in range(kernel_size):
                    shifted = padded[:, :, 0 + i: h + i, 0 + j:w + j]
                    label = center == shifted
                    aggregated_label[:, :, :, :, i, j] = label
            aggregated_label = aggregated_label.float()
            pos_idx = (aggregated_label == 1).float()
            neg_idx = (aggregated_label == 0).float()
            pos_idx_num = pos_idx.sum(dim=-1).sum(dim=-1)
            neg_idx_num = neg_idx.sum(dim=-1).sum(dim=-1)

            boundary_region = (pos_idx_num >= kernel_size - 1) & (
                    neg_idx_num >= kernel_size - 1)
            non_boundary_region = (pos_idx_num != 0) & (neg_idx_num == 0)

            if s == min(self.opt.sgt_layers):
                outputs[('boundary', s)] = boundary_region.data
                outputs[('non_boundary', s)] = non_boundary_region.data

            feature = outputs[('d_feature', s)]
            affinity = self.compute_affinity(feature, kernel_size=kernel_size)
            pos_dist = (pos_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                       pos_idx.sum(dim=-1).sum(dim=-1)[
                           boundary_region]
            neg_dist = (neg_idx * affinity).sum(dim=-1).sum(dim=-1)[boundary_region] / \
                       neg_idx.sum(dim=-1).sum(dim=-1)[
                           boundary_region]
            zeros = torch.zeros(pos_dist.shape).to(pos_dist.device)
            loss = torch.max(zeros, pos_dist - neg_dist + self.opt.sgt_margin)

            total_loss += loss.mean() / (2 ** s)

        return total_loss

    def reprojection_loss(self, pred, target):
        '''if pred.size(2) != target.size(2):
            pred = pred.unsqueeze(0)  # 假设批量大小为1
            target = target.unsqueeze(0)  # 假设批量大小为1
            target = F.interpolate(target, size=(pred.size(2), pred.size(3)), mode='bilinear', align_corners=False)
            pred = pred.squeeze(0)
            target = target.squeeze(0)'''
        abs_diff = torch.abs(target - pred)
        ssim_loss = torch.mean(self.ssim(pred, target))
        l1_loss = torch.mean(abs_diff)

        loss = 0.85 * ssim_loss + 0.15 * l1_loss
        return loss

    def compute_semantic_distil(self, inputs, outputs):

        total_loss = 0
        # for s in self.opt.scales:

        scales = [0]

        for s in scales:
            seg_target = inputs["seg_gt"].long().squeeze(1)
            seg_pred = outputs[("seg_logits", s)]
            weights = seg_target.sum(1, keepdim=True).float()
            ignore_mask = (weights == 0)
            weights[ignore_mask] = 1
            seg_loss = F.cross_entropy(seg_pred, seg_target, reduction='none')
            total_loss += seg_loss.mean() / (2 ** s)

        return total_loss

    def compute_smoothness(self, inputs, outputs):
        total_loss = 0
        for s in self.opt.scales:
            disp = outputs[("disp", s)]
            lf_forward = F.interpolate(inputs[("lf_forward",40)], size=(disp.size(2), disp.size(3)), mode='bilinear', align_corners=False)
            mean_disp = disp.mean(2, True).mean(3, True)
            norm_disp = disp / (mean_disp + 1e-7)
            smooth_loss = get_smooth_loss(norm_disp, lf_forward)
            total_loss += smooth_loss / (2 ** s)

        return total_loss


    def compute_Rec_Loss(self, inputs, outputs):
        total_loss = 0
        target = inputs[("lf_forward", 40)]

        #lf_warp_list = [10, 13, 16, 20, 22, 24, 30, 31, 32, 37, 38, 39, 41, 42, 43, 48, 49, 50, 56, 58, 60, 64, 67, 70]
        lf_warp_list = [30, 31, 32, 39, 40, 41, 48, 49, 50]

        scales = [0]

        #for s in self.opt.scales:
        for s in scales:
            disp = outputs[("disp", s)]
            losses = []

            for k_s in lf_warp_list:
                pred = self.warping(disp, k_s, 40, inputs[("lf_forward", k_s)], self.opt.angRes)
                pred_resized = F.interpolate(pred, size=(target.size(2), target.size(3)), mode='bilinear', align_corners=True)

                ssim_loss = self.reprojection_loss(pred_resized, target)

                losses.append(ssim_loss)

            total_loss += torch.sum(torch.stack(losses)) / len(losses) / (2 ** s)

        return total_loss

    def warping(self, disp, ind_source, ind_target, img_source, an):
        '''warping one source image/map to the target'''
        # an angular number
        # disparity: int or [N,h,w]
        # ind_souce
        # ind_target
        # img_source [N,c,h,w]

        # ==> out [N,c,h,w]
        # print('img source ', img_source.shape)

        N, c, h, w = img_source.shape
        disp = disp.type_as(img_source)  # 将disp的数据类型转换为img_source的数据类型
        # ind_source = ind_source.type_as(disp)
        # ind_target = ind_target.type_as(disp)
        # print(img_source.shape)
        # coordinate for source and target
        # ind_souce = torch.tensor([0,an-1,an2-an,an2-1])[ind_source]
        ind_h_source = math.floor(ind_source / an)
        ind_w_source = ind_source % an

        ind_h_target = math.floor(ind_target / an)
        ind_w_target = ind_target % an

        # generate grid
        XX = torch.arange(0, w).view(1, 1, w).expand(N, h, w).type_as(img_source)  # [N,h,w]
        YY = torch.arange(0, h).view(1, h, 1).expand(N, h, w).type_as(img_source)

        disp = disp.squeeze(1)
        disp_height = disp.size(1)
        disp_width = disp.size(2)
        XX_resized = torch.unsqueeze(XX, dim=1)
        YY_resized = torch.unsqueeze(YY, dim=1)
        XX_resized = F.interpolate(XX_resized, size=(disp_height, disp_width), mode='bilinear', align_corners=False)
        YY_resized = F.interpolate(YY_resized, size=(disp_height, disp_width), mode='bilinear', align_corners=False)
        XX = XX_resized.squeeze(1)
        YY = YY_resized.squeeze(1)
        grid_w = XX + disp * (ind_w_target - ind_w_source)
        grid_h = YY + disp * (ind_h_target - ind_h_source)

        grid_w_norm = 2.0 * grid_w / (w - 1) - 1.0
        grid_h_norm = 2.0 * grid_h / (h - 1) - 1.0

        grid = torch.stack((grid_w_norm, grid_h_norm), dim=3)  # [N,h,w,2]

        # img_source = torch.unsqueeze(img_source, 1)
        # print(img_source.shape)
        # print(grid.shape)
        # print(tt)
        img_target = F.grid_sample(img_source, grid, padding_mode='border', align_corners=False)  # [N,3,h,w]
        # img_target = torch.squeeze(img_target, 1)  # [N,h,w]
        return img_target


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """

    def __init__(self, num_layers=18, pretrained=True, num_input_images=1,):
        super(ResnetEncoder, self).__init__()

        if pretrained:
            print('load pretrained from imagenet, resnet', num_layers)
        else:
            print('train starts from the scratch')
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            num_layers = min(50, num_layers)
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:

            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
        if not pretrained:
            self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, input_image):
        features = []
        x = (input_image - 0.45) / 0.225 #对输入图像进行归一化
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))

        return features

class AttentionFusion(nn.Module):
    def __init__(self, num_features):
        super(AttentionFusion, self).__init__()
        self.num_features = num_features
        self.attention_weights = nn.Parameter(torch.ones(num_features))  # 初始化注意力权重

    def forward(self, features_list):
        # 将特征列表转置，使得每个维度的特征在列表中是连续的
        transposed_features = [torch.stack([features_list[j][i] for j in range(self.num_features)], dim=0) for i in range(5)]

        # 计算注意力权重
        attention_scores = F.softmax(self.attention_weights, dim=0)

        # 使用注意力权重对每个维度上的特征进行加权融合
        fused_features = [torch.sum(attention_scores[i] * transposed_features[i], dim=0) for i in range(5)]

        return fused_features



