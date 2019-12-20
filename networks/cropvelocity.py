import torch
import torch.nn as nn

from .PWCNet import *
from .roialign.roi_align.roi_align import RoIAlign


class CropVelocity(PWCDCNet):
    def __init__(self, args):
        super(CropVelocity, self).__init__(args.md)

        self.args = args
        self.batch_size = args.batch_size

        self.car_roi = RoIAlign(7, 7, transform_fpcoor=False)

        self.relu = nn.ReLU(inplace=True)

        # for p in self.parameters():
        #     p.requires_grad = False

        self.roi_conv1 = nn.Conv2d(196, 196, 7, stride=1, padding=0)
        self.roi_conv2 = nn.Conv2d(196, 196, 1, stride=1, padding=0)
        self.mv_fc1 = nn.Linear(6 + 196 + 7*7*5*2, 512)
        self.mv_fc2 = nn.Linear(512, 256)
        self.mv_fc3 = nn.Linear(256, 128)
        self.mv_fc4 = nn.Linear(128, 64)
        self.mv_fc5 = nn.Linear(64, 3)
        # self.dropout = nn.Dropout(0.2)

    def forward(self, input):
        n, b, c, h, w = input['second_image'].shape
        im1 = input['second_image'].view(n * b, c, h, w)
        im2 = input['first_image'].view(n * b, c, h, w)
        rois = input['rois'].view(n * b, 4)

        roi_ids = torch.arange(n * b, dtype=torch.int32, device=rois.device)

        area = input['rois_size'].view(n * b, 6)
        bias = input['rescale'].view(n * b, 2)

        c11 = self.conv1b(self.conv1aa(self.conv1a(im1)))
        c21 = self.conv1b(self.conv1aa(self.conv1a(im2)))
        c12 = self.conv2b(self.conv2aa(self.conv2a(c11)))
        c22 = self.conv2b(self.conv2aa(self.conv2a(c21)))
        c13 = self.conv3b(self.conv3aa(self.conv3a(c12)))
        c23 = self.conv3b(self.conv3aa(self.conv3a(c22)))
        c14 = self.conv4b(self.conv4aa(self.conv4a(c13)))
        c24 = self.conv4b(self.conv4aa(self.conv4a(c23)))
        c15 = self.conv5b(self.conv5aa(self.conv5a(c14)))
        c25 = self.conv5b(self.conv5aa(self.conv5a(c24)))
        c16 = self.conv6b(self.conv6a(self.conv6aa(c15)))
        c26 = self.conv6b(self.conv6a(self.conv6aa(c25)))

        corr6 = self.corr(c16, c26)
        corr6 = self.leakyRELU(corr6)

        x = torch.cat((self.conv6_0(corr6), corr6), 1)
        x = torch.cat((self.conv6_1(x), x), 1)
        x = torch.cat((self.conv6_2(x), x), 1)
        x = torch.cat((self.conv6_3(x), x), 1)
        x = torch.cat((self.conv6_4(x), x), 1)
        flow6 = self.predict_flow6(x)
        up_flow6 = self.deconv6(flow6)
        up_feat6 = self.upfeat6(x)

        warp5 = self.warp(c25, up_flow6 * 0.625)
        corr5 = self.corr(c15, warp5)
        corr5 = self.leakyRELU(corr5)
        x = torch.cat((corr5, c15, up_flow6, up_feat6), 1)
        x = torch.cat((self.conv5_0(x), x), 1)
        x = torch.cat((self.conv5_1(x), x), 1)
        x = torch.cat((self.conv5_2(x), x), 1)
        x = torch.cat((self.conv5_3(x), x), 1)
        x = torch.cat((self.conv5_4(x), x), 1)
        flow5 = self.predict_flow5(x)
        up_flow5 = self.deconv5(flow5)
        up_feat5 = self.upfeat5(x)

        warp4 = self.warp(c24, up_flow5 * 1.25)
        corr4 = self.corr(c14, warp4)
        corr4 = self.leakyRELU(corr4)
        x = torch.cat((corr4, c14, up_flow5, up_feat5), 1)
        x = torch.cat((self.conv4_0(x), x), 1)
        x = torch.cat((self.conv4_1(x), x), 1)
        x = torch.cat((self.conv4_2(x), x), 1)
        x = torch.cat((self.conv4_3(x), x), 1)
        x = torch.cat((self.conv4_4(x), x), 1)
        flow4 = self.predict_flow4(x)
        up_flow4 = self.deconv4(flow4)
        up_feat4 = self.upfeat4(x)

        warp3 = self.warp(c23, up_flow4 * 2.5)
        corr3 = self.corr(c13, warp3)
        corr3 = self.leakyRELU(corr3)

        x = torch.cat((corr3, c13, up_flow4, up_feat4), 1)
        x = torch.cat((self.conv3_0(x), x), 1)
        x = torch.cat((self.conv3_1(x), x), 1)
        x = torch.cat((self.conv3_2(x), x), 1)
        x = torch.cat((self.conv3_3(x), x), 1)
        x = torch.cat((self.conv3_4(x), x), 1)
        flow3 = self.predict_flow3(x)
        up_flow3 = self.deconv3(flow3)
        up_feat3 = self.upfeat3(x)

        warp2 = self.warp(c22, up_flow3 * 5.0)
        corr2 = self.corr(c12, warp2)
        corr2 = self.leakyRELU(corr2)
        x = torch.cat((corr2, c12, up_flow3, up_feat3), 1)
        x = torch.cat((self.conv2_0(x), x), 1)
        x = torch.cat((self.conv2_1(x), x), 1)
        x = torch.cat((self.conv2_2(x), x), 1)
        x = torch.cat((self.conv2_3(x), x), 1)
        x = torch.cat((self.conv2_4(x), x), 1)
        flow2 = self.predict_flow2(x)

        x = self.dc_conv4(self.dc_conv3(self.dc_conv2(self.dc_conv1(x))))
        flow2 += self.dc_conv7(self.dc_conv6(self.dc_conv5(x)))

        z = self.car_roi(c16, rois / 64, roi_ids)
        z = self.roi_conv1(z)
        z = self.relu(z)
        z = self.roi_conv2(z)
        z = self.relu(z)
        z = z.view(-1, 196)
        z = torch.cat((area, z), 1)

        roi_flow6 = self.car_roi(flow6*0.3125, rois/64, roi_ids)
        roi_flow5 = self.car_roi(flow5*0.625, rois/32, roi_ids)
        roi_flow4 = self.car_roi(flow4*1.25, rois/16, roi_ids)
        roi_flow3 = self.car_roi(flow3*2.5, rois/8, roi_ids)
        roi_flow2 = self.car_roi(flow2*5, rois/4, roi_ids)

        bias = bias.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 7, 7) * 4.0
        roi_flow6 = roi_flow6 * bias
        roi_flow5 = roi_flow5 * bias
        roi_flow4 = roi_flow4 * bias
        roi_flow3 = roi_flow3 * bias
        roi_flow2 = roi_flow2 * bias

        flow = torch.cat((roi_flow2, roi_flow3, roi_flow4, roi_flow5, roi_flow6), 1)
        flow = flow.view(-1, 7*7*5*2)

        z_and_flow = torch.cat((z, flow), dim=1)
        z_and_flow = self.relu(self.mv_fc1(z_and_flow))
        z_and_flow = self.relu(self.mv_fc2(z_and_flow))
        z_and_flow = self.relu(self.mv_fc3(z_and_flow))
        z_and_flow = self.relu(self.mv_fc4(z_and_flow))
        z_v = self.mv_fc5(z_and_flow)

        return z_v


def crop_velocity_net(args):
    model = CropVelocity(args)
    if args.pre_trained:
        pretrained_dict = torch.load(args.trained_model)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                           k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


class CropLoss(nn.Module):
    def __init__(self, args):
        super(CropLoss, self).__init__()

        self.args = args

    def forward(self, output, target):
        z_v_out = output
        n, b, c, h, w = target['second_image'].shape
        velocity_true = target['velocity'].view(n * b, 2)
        position_true = target['location'].view(n * b, 2)
        z_v_true = torch.cat((self.args.div_distance * position_true[:, 0:1], velocity_true), dim=1)
        z_v_loss = torch.norm(z_v_true - z_v_out, p=2, dim=1).mean()

        # z_true = self.args.div_distance * position_true[:, 0:1]
        # z_loss = torch.norm(z_true-z_v_out[:, 0:1], p=2, dim=1).mean()
        # return z_loss
        return z_v_loss


