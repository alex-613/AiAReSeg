import torch
import torch.nn as nn
import torch.nn.functional as F

from external.PreciseRoIPooling.pytorch.prroi_pool import PrRoIPool2D
from lib.models.aiareseg.backbone import FrozenBatchNorm2d


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
         freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))


class Corner_Predictor(nn.Module):
    """
    Corner predictor module.
    """

    def __init__(self, inplanes=64, channel=256, feat_sz=20, stride=16, freeze_bn=False):
        super(Corner_Predictor, self).__init__()
        self.feat_sz = feat_sz
        self.stride = stride
        self.img_sz = self.feat_sz * self.stride
        # Top-left corner
        self.conv1_tl = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_tl = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_tl = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_tl = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=(1, 1))

        # Bottom-right corner
        self.conv1_br = conv(inplanes, channel, freeze_bn=freeze_bn)
        self.conv2_br = conv(channel, channel // 2, freeze_bn=freeze_bn)
        self.conv3_br = conv(channel // 2, channel // 4, freeze_bn=freeze_bn)
        self.conv4_br = conv(channel // 4, channel // 8, freeze_bn=freeze_bn)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=(1, 1))

        # About coordinates and indexes
        with torch.no_grad():
            self.indice = torch.arange(0, self.feat_sz).view(-1, 1) * self.stride
            # Generate mesh-grid
            self.coord_x = self.indice.repeat((self.feat_sz, 1)).view((self.feat_sz * self.feat_sz,)).float().cuda()
            self.coord_y = self.indice.repeat((1, self.feat_sz)).view((self.feat_sz * self.feat_sz,)).float().cuda()

    def forward(self, x):
        """
        Forward pass with input x.
        """

        score_map_tl, score_map_br = self.get_score_map(x)
        coorx_tl, coory_tl = self.soft_argmax(score_map_tl)
        coorx_br, coory_br = self.soft_argmax(score_map_br)
        return torch.stack((coorx_tl, coory_tl, coorx_br, coory_br), dim=1) / self.img_sz

    def get_score_map(self, x):
        # Top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        score_map_tl = self.conv5_tl(x_tl4)

        # Bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        score_map_br = self.conv5_br(x_br4)
        return score_map_tl, score_map_br

    def soft_argmax(self, score_map):
        """
        Get soft-argmax coordinate for a given heatmap.
        """

        prob_vec = nn.functional.softmax(
            score_map.view((-1, self.feat_sz * self.feat_sz)), dim=1)  # (batch, feat_sz * feat_sz)
        exp_x = torch.sum((self.coord_x * prob_vec), dim=1)
        exp_y = torch.sum((self.coord_y * prob_vec), dim=1)
        return exp_x, exp_y


class MLP(nn.Module):
    """
    Very simple multi-layer perceptron (also called FFN).
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_planes, out_planes, input_sz, bias=True, batch_norm=True, relu=True):
        super().__init__()
        self.linear = nn.Linear(in_planes * input_sz * input_sz, out_planes, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes) if batch_norm else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.linear(x.reshape(x.shape[0], -1))
        if self.bn is not None:
            x = self.bn(x.reshape(x.shape[0], x.shape[1], 1, 1))
        if self.relu is not None:
            x = self.relu(x)
        return x.reshape(x.shape[0], -1)


class IoUNet(nn.Module):
    """
    Network module for IoU prediction.

    Args:
        dim: Feature dimensionality of the encoded features.
    """

    def __init__(self, dim=256):
        super().__init__()
        self.conv1 = conv(dim, dim, kernel_size=3, stride=1)
        self.conv2 = conv(dim, dim, kernel_size=3, stride=1)
        self.conv3 = conv(dim, dim, kernel_size=3, stride=1)

        self.prroi_pool = PrRoIPool2D(4, 4, 20.0)

        self.fc = LinearBlock(dim, dim, 4)

        self.iou_predictor = nn.Linear(dim, 1, bias=True)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization
                # Which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1)
                # So we use the same initialization here
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()

    def forward(self, feat, proposals):
        """
        Runs the IoUNet during training operation.
        This forward pass is mainly used for training. Call the individual functions during tracking instead.

        Args:
            feat: Features from the test frames (4 or 5 dims).
            proposals: Proposal boxes for which the IoU will be predicted (images, sequences, num_proposals, 4).
        """

        assert proposals.dim() == 4

        num_images = proposals.shape[0]
        num_sequences = proposals.shape[1]

        iou_feat = self.get_iou_feat(feat)

        proposals = proposals.reshape(num_sequences * num_images, -1, 4)
        pred_iou = self.predict_iou(iou_feat, proposals)
        return pred_iou.reshape(num_images, num_sequences, -1)

    def predict_iou(self, feat, proposals):
        """
        Predicts IoU for the give proposals.

        Args:
            feat: IoU features (from get_iou_feat) for test images. Dims (batch, feature_dim, H, W).
            proposals: Proposal boxes for which the IoU will be predicted (batch, num_proposals, 4).
        """

        batch_size = feat.shape[0]

        # Add batch_index to rois
        batch_index = torch.arange(batch_size, dtype=torch.float32).reshape(-1, 1).to(feat.device)

        # Push the different rois for the same image along the batch dimension
        num_proposals_per_batch = proposals.shape[1]

        # Input proposals is in format xywh, convert it to x0y0x1y1 format
        proposals_xyxy = torch.cat((proposals[:, :, 0:2], proposals[:, :, 0:2] + proposals[:, :, 2:4]), dim=2)

        # Add batch index
        roi = torch.cat((batch_index.reshape(batch_size, -1, 1).expand(-1, num_proposals_per_batch, -1),
                         proposals_xyxy), dim=2)
        roi = roi.reshape(-1, 5).to(proposals_xyxy.device)

        roi_feat = self.prroi_pool(feat, roi)

        fc_feat = self.fc(roi_feat)

        iou_pred = self.iou_predictor(fc_feat).reshape(batch_size, num_proposals_per_batch)

        return iou_pred

    def get_iou_feat(self, feat):
        """
        Get IoU prediction features from a 4 or 5 dimensional backbone input.
        """

        feat = feat.reshape(-1, *feat.shape[-3:]) if feat.dim() == 5 else feat
        iou_feat = self.conv3(self.conv2(self.conv1(feat)))

        return iou_feat

def deconv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, freeze_bn=False):
    if freeze_bn:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding,dilation=dilation,bias=True),
            FrozenBatchNorm2d(out_planes),
            nn.ReLU(inplace=True))
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                               padding=padding,dilation=dilation,bias=True),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True))

class DoubleConv2d(nn.Module):
    def __init__(self,in_channels,features):
        super(DoubleConv2d, self).__init__()
        self.block = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                #nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True),
                nn.Conv2d(features, features, kernel_size=3, padding=1),
                nn.BatchNorm2d(features),
                #nn.GroupNorm(num_groups=int(features/16), num_channels=features),
                nn.ReLU(inplace=True)
                )

    def forward(self,x):
        return self.block(x)

class deconv3d(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(deconv3d, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels,out_channels, kernel_size=(2,2,2), stride=(1,2,2), padding=(1,0,0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.block(x)

class deconv3d_4C(nn.Module):
    # This module will reduce the number of channels from 4 to 1
    def __init__(self,in_channels,out_channels):
        super(deconv3d_4C, self).__init__()
        self.block = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=(4,2,2), stride=(1,2,2), padding=(3,0,0)),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        return self.block(x)

class AiAReSeg(nn.Module):

    def __init__(self, dim=256, embed_size=400, feature_size=20):
        super().__init__()

        self.in_dim = dim
        self.embed_size = embed_size
        self.feature_size = feature_size

        self.seg160dim = 64
        self.seg80dim = 256
        self.seg40dim = 512
        self.seg20dim = 1024

        self.d11 = DoubleConv2d(self.in_dim, self.in_dim//2)          # 256  -> 128
        self.r11 = DoubleConv2d(self.seg20dim, self.seg20dim//2)      # 1024 -> 512
        self.r12 = DoubleConv2d(self.seg20dim//2, self.seg20dim//4)   # 512  -> 256
        self.r13 = DoubleConv2d(self.seg20dim//4, self.seg20dim//8)   # 256  -> 128

        self.d21 = deconv3d(self.in_dim//2, self.in_dim//4)           # 128 -> 64, 20 -> 40
        self.r21 = DoubleConv2d(self.seg40dim, self.seg40dim//2)      # 512 -> 256
        self.r22 = DoubleConv2d(self.seg40dim//2, self.seg40dim//4)   # 256 -> 128
        self.r23 = DoubleConv2d(self.seg40dim//4, self.seg40dim//8)   # 128 -> 64

        self.d31 = deconv3d(self.in_dim//4, self.in_dim//8)           # 64  -> 32, 40 -> 80
        self.r31 = DoubleConv2d(self.seg80dim, self.seg80dim//2)      # 256 -> 128
        self.r32 = DoubleConv2d(self.seg80dim//2, self.seg80dim//4)   # 128 -> 64
        self.r33 = DoubleConv2d(self.seg80dim//4, self.seg80dim//8)   # 64  -> 32

        self.d41 = deconv3d(self.in_dim//8, self.in_dim//16)          # 32 -> 16, 80 -> 160
        self.d42 = DoubleConv2d(self.in_dim//16, self.in_dim//32)     # 16 -> 8,
        self.d43 = DoubleConv2d(self.in_dim//32, self.in_dim//64)     # 8  -> 4
        self.r41 = DoubleConv2d(self.seg160dim, self.seg160dim//2)    # 64 -> 32
        self.r42 = DoubleConv2d(self.seg160dim//2, self.seg160dim//4) # 32 -> 16
        self.r43 = DoubleConv2d(self.seg160dim//4, self.seg160dim//16)# 16 -> 4

        self.d51 = deconv3d(self.in_dim//64, self.in_dim//128)        # 4 -> 2, 160 -> 320
        self.d52 = DoubleConv2d(self.in_dim//128, self.in_dim//256)   # 2 -> 1
        self.d53 = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=1,stride=1,padding=0) # 1 x 1 conv

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization
                # Which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1)
                # So we use the same initialization here
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()


    def forward(self,embedding, seg20=None, seg40=None, seg80=None, seg160=None, pos_emb=None):

        opt = embedding.permute(2,0,3,1).contiguous() # (B,1,C,HW)
        # We need to use the positonal embedding information during the reconstruction in order for it to be meaningful
        pos_emb = pos_emb.permute(1,2,0).contiguous()

        # Perform position reconstruction
        bs, Nq, C, HW = opt.size()
        pos_emb = pos_emb.view(-1, C, self.feature_size, self.feature_size)
        opt_feat = opt.view(-1, C, self.feature_size, self.feature_size)
        opt_feat = opt_feat * pos_emb # This may be incorrect, this may be addition instead of multiplication

        dout1 = self.d11(opt_feat)
        rout1 = self.r13(self.r12(self.r11(seg20)))
        layer1_out = torch.stack([dout1,rout1],dim=2)

        dout2 = self.d21(layer1_out).squeeze(2)
        rout2 = self.r23(self.r22(self.r21(seg40)))
        layer2_out = torch.stack([dout2,rout2],dim=2)

        dout3 = self.d31(layer2_out).squeeze(2)
        rout3 = self.r33(self.r32(self.r31(seg80)))
        layer3_out = torch.stack([dout3,rout3],dim=2)

        dout4 = self.d41(layer3_out).squeeze(2)
        dout4 = self.d43(self.d42(dout4))
        rout4 = self.r43(self.r42(self.r41(seg160)))
        layer4_out = torch.stack([dout4,rout4],dim=2)

        dout5 = self.d51(layer4_out).squeeze(2)
        layer5_out = self.d53(self.d52(dout5))

        output = torch.sigmoid(layer5_out)

        return output

class AiAReSeg_w_temp(nn.Module):

    def __init__(self, dim=256, embed_size=400, feature_size=20):
        super().__init__()

        self.in_dim = dim
        self.embed_size = embed_size
        self.feature_size = feature_size

        self.seg160dim = 64
        self.seg80dim = 256
        self.seg40dim = 512
        self.seg20dim = 1024

        self.d11 = DoubleConv2d(self.in_dim, self.in_dim//2)          # 256  -> 128
        self.r11 = DoubleConv2d(self.seg20dim, self.seg20dim//2)      # 1024 -> 512
        self.r12 = DoubleConv2d(self.seg20dim//2, self.seg20dim//4)   # 512  -> 256
        self.r13 = DoubleConv2d(self.seg20dim//4, self.seg20dim//8)   # 256  -> 128

        self.d21 = deconv3d_4C(self.in_dim//2, self.in_dim//4)        # 128 -> 64, 20 -> 40
        self.r21 = DoubleConv2d(self.seg40dim, self.seg40dim//2)      # 512 -> 256
        self.r22 = DoubleConv2d(self.seg40dim//2, self.seg40dim//4)   # 256 -> 128
        self.r23 = DoubleConv2d(self.seg40dim//4, self.seg40dim//8)   # 128 -> 64

        self.d31 = deconv3d_4C(self.in_dim//4, self.in_dim//8)        # 64  -> 32, 40 -> 80
        self.r31 = DoubleConv2d(self.seg80dim, self.seg80dim//2)      # 256 -> 128
        self.r32 = DoubleConv2d(self.seg80dim//2, self.seg80dim//4)   # 128 -> 64
        self.r33 = DoubleConv2d(self.seg80dim//4, self.seg80dim//8)   # 64  -> 32

        self.d41 = deconv3d_4C(self.in_dim//8, self.in_dim//16)       # 32 -> 16, 80 -> 160
        self.d42 = DoubleConv2d(self.in_dim//16, self.in_dim//32)     # 16 -> 8,
        self.d43 = DoubleConv2d(self.in_dim//32, self.in_dim//64)     # 8  -> 4
        self.r41 = DoubleConv2d(self.seg160dim, self.seg160dim//2)    # 64 -> 32
        self.r42 = DoubleConv2d(self.seg160dim//2, self.seg160dim//4) # 32 -> 16
        self.r43 = DoubleConv2d(self.seg160dim//4, self.seg160dim//16)# 16 -> 4

        self.d51 = deconv3d_4C(self.in_dim//64, self.in_dim//128)        # 4 -> 2, 160 -> 320
        self.d52 = DoubleConv2d(self.in_dim//128, self.in_dim//256)   # 2 -> 1
        self.d53 = nn.Conv2d(in_channels=1,out_channels=1, kernel_size=1,stride=1,padding=0) # 1 x 1 conv

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight.data, mode='fan_in')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                # In earlier versions batch norm parameters was initialized with default initialization
                # Which changed in pytorch 1.2. In 1.1 and earlier the weight was set to U(0,1)
                # So we use the same initialization here
                # m.weight.data.fill_(1)
                m.weight.data.uniform_()
                m.bias.data.zero_()


    def forward(self,embedding, search_outputs, reference_outputs):

        opt = embedding.permute(2,0,3,1).contiguous() # (B,1,C,HW)
        # We need to use the positonal embedding information during the reconstruction in order for it to be meaningful
        pos_emb = search_outputs['pos'].permute(1,2,0).contiguous()

        # Perform position reconstruction
        bs, Nq, C, HW = opt.size()
        pos_emb = pos_emb.view(-1, C, self.feature_size, self.feature_size)
        opt_feat = opt.view(-1, C, self.feature_size, self.feature_size)
        opt_feat = opt_feat * pos_emb

        dout1 = self.d11(opt_feat)
        rsout1 = self.r13(self.r12(self.r11(search_outputs['Channel20'])))
        riout1 = self.r13(self.r12(self.r11(reference_outputs['0']['Channel20'])))
        rintout1 = self.r13(self.r12(self.r11(reference_outputs['1']['Channel20'])))
        layer1_out = torch.stack([dout1, rsout1, riout1, rintout1], dim=2)

        dout2 = self.d21(layer1_out).squeeze(2)
        rsout2 = self.r23(self.r22(self.r21(search_outputs['Channel40'])))
        riout2 = self.r23(self.r22(self.r21(reference_outputs['0']['Channel40'])))
        rintout2 = self.r23(self.r22(self.r21(reference_outputs['1']['Channel40'])))
        layer2_out = torch.stack([dout2, rsout2, riout2, rintout2], dim=2)

        dout3 = self.d31(layer2_out).squeeze(2)
        rsout3 = self.r33(self.r32(self.r31(search_outputs['Channel80'])))
        riout3 = self.r33(self.r32(self.r31(reference_outputs['0']['Channel80'])))
        rintout3 = self.r33(self.r32(self.r31(reference_outputs['1']['Channel80'])))
        layer3_out = torch.stack([dout3, rsout3, riout3, rintout3], dim=2)

        dout4 = self.d41(layer3_out).squeeze(2)
        dout4 = self.d43(self.d42(dout4))
        rsout4 = self.r43(self.r42(self.r41(search_outputs['Channel160'])))
        riout4 = self.r43(self.r42(self.r41(reference_outputs['0']['Channel160'])))
        rintout3 = self.r43(self.r42(self.r41(reference_outputs['1']['Channel160'])))
        layer4_out = torch.stack([dout4,rsout4,riout4,rintout3],dim=2)

        dout5 = self.d51(layer4_out).squeeze(2)
        layer5_out = self.d53(self.d52(dout5))

        output = torch.sigmoid(layer5_out)



        # dout1 = self.d11(opt_feat)
        # rout1 = self.r13(self.r12(self.r11(seg20)))
        # layer1_out = torch.stack([dout1,rout1],dim=2)
        #
        # dout2 = self.d21(layer1_out).squeeze(2)
        # rout2 = self.r23(self.r22(self.r21(seg40)))
        # layer2_out = torch.stack([dout2,rout2],dim=2)
        #
        # dout3 = self.d31(layer2_out).squeeze(2)
        # rout3 = self.r33(self.r32(self.r31(seg80)))
        # layer3_out = torch.stack([dout3,rout3],dim=2)
        #
        # dout4 = self.d41(layer3_out).squeeze(2)
        # dout4 = self.d43(self.d42(dout4))
        # rout4 = self.r43(self.r42(self.r41(seg160)))
        # layer4_out = torch.stack([dout4,rout4],dim=2)
        #
        # dout5 = self.d51(layer4_out).squeeze(2)
        # layer5_out = self.d53(self.d52(dout5))
        #
        # output = torch.sigmoid(layer5_out)

        return output



def build_box_head(cfg):
    if cfg.MODEL.HEAD_TYPE == 'MLP':
        hidden_dim = cfg.MODEL.HIDDEN_DIM
        mlp_head = MLP(hidden_dim, hidden_dim, 4, 3)  # dim_in, dim_hidden, dim_out, 3 layers
        return mlp_head
    elif cfg.MODEL.HEAD_TYPE == 'CORNER':
        if cfg.MODEL.BACKBONE.DILATION is False:
            stride = 16
        else:
            stride = 8
        feat_sz = int(cfg.DATA.SEARCH.SIZE / stride)
        corner_head = Corner_Predictor(inplanes=cfg.MODEL.HIDDEN_DIM, channel=256,
                                       feat_sz=feat_sz, stride=stride)
        return corner_head
    else:
        raise ValueError('ERROR: head type %s is not supported' % cfg.MODEL.HEAD_TYPE)


def build_iou_head(cfg):
    hidden_dim = cfg.MODEL.HIDDEN_DIM
    return IoUNet(hidden_dim)

def build_segmentation_head(cfg):
    hidden_dim = cfg.MODEL.HIDDEN_DIM
    embed_size = cfg.MODEL.AIA.FEAT_SIZE
    feature_size = 20
    #seg_head = Segmentation_head(dim=hidden_dim, embed_size=embed_size, feature_size=feature_size)
    #seg_head = AiAReSeg(dim=hidden_dim, embed_size=embed_size, feature_size=feature_size)
    seg_head = AiAReSeg_w_temp(dim=hidden_dim, embed_size=embed_size, feature_size=feature_size)

    return seg_head
