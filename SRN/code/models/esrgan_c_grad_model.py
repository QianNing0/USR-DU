import torch
import torch.nn.functional as F
import torch.nn as nn
from collections import OrderedDict

from code.utils.registry import MODEL_REGISTRY
from .srgan_model import SRGANModel
from code.losses import build_loss
from pytorch_wavelets import DWTForward


class GaussianFilter(nn.Module):
    def __init__(self, kernel_size=5, stride=1, padding=4):
        super(GaussianFilter, self).__init__()
        # initialize guassian kernel
        mean = (kernel_size - 1) / 2.0
        variance = (kernel_size / 6.0) ** 2.0
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_coord = torch.arange(kernel_size)
        x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

        # Calculate the 2-dimensional gaussian kernel
        gaussian_kernel = torch.exp(-torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)

        # create gaussian filter as convolutional layer
        self.gaussian_filter = nn.Conv2d(3, 3, kernel_size, stride=stride, padding=padding, groups=3, bias=False)
        self.gaussian_filter.weight.data = gaussian_kernel
        self.gaussian_filter.weight.requires_grad = False

    def forward(self, x):
        return self.gaussian_filter(x)


class FilterLow(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, padding=True, include_pad=True, gaussian=False):
        super(FilterLow, self).__init__()
        if padding:
            pad = int((kernel_size - 1) / 2)
        else:
            pad = 0
        if gaussian:
            self.filter = GaussianFilter(kernel_size=kernel_size, stride=stride, padding=pad)
        else:
            self.filter = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=pad, count_include_pad=include_pad)
        self.recursions = recursions

    def forward(self, img):
        for i in range(self.recursions):
            img = self.filter(img)
        return img

def color_gradient(x):
    # tf.image.image_gradients(image)
    # gradient step=1
    l = x
    r = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]
    t = x
    b = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = torch.abs(r - l), torch.abs(b - t)
    # dx will always have zeros in the last column, r-l
    # dy will always have zeros in the last row,    b-t
    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx+dy


@MODEL_REGISTRY.register()
class ESRGANModel_C_Grad(SRGANModel):
    """ESRGAN model for single image super-resolution."""
    def __init__(self, opt):
        super(ESRGANModel_C_Grad, self).__init__(opt=opt)
        train_opt = opt['FS']
        self.norm = train_opt['norm']
        if train_opt['type'] == 'wavelet':
            # Wavelet
            self.DWT2 = DWTForward(J=1, mode='reflect', wave='haar').to(self.device)
            self.fs = self.wavelet_s
        elif train_opt['type'] == 'gau':
            # Gaussian
            self.filter_low = FilterLow(kernel_size=train_opt['kernel_size'], gaussian=True).to(self.device)
            self.fs = self.filter_low
        elif train_opt['type'] == 'avgpool':
            # avgpool
            self.filter_low = FilterLow(kernel_size=train_opt['kernel_size']).to(self.device)
            self.fs = self.filter_low
        else:
            raise NotImplementedError('FS type [{:s}] not recognized.'.format(train_opt['type']))
        if self.opt['weighted_loss']:
            self.UncertaintyLoss = build_loss(opt['train']['UncertaintyLoss_opt']).to(self.device)

    def optimize_parameters(self, current_iter):
        if self.opt['use_USM']:
            l1_gt = self.gt_usm
            percep_gt = self.gt_usm
        else:
            l1_gt = self.gt
            percep_gt = self.gt

        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()
        if self.opt['need_HR']:
            self.output = self.net_g([self.lq, self.gt])
        else:
            self.output = self.net_g(self.lq)

        if self.opt['weighted_loss']:
            self.var = self.output[1]
            self.output = self.output[0]
        else:
            self.var = None

        self.output_grad = color_gradient(self.output)
        self.gt_grad = color_gradient(self.gt)

        self.output_low_f = self.fs(self.output)
        self.gt_low_f = self.fs(self.gt)

        # self.output = self.net_g(self.lq)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.opt['weighted_loss']:
                l_g_pix = self.UncertaintyLoss(self.output, l1_gt, self.var / 255)
                l_g_total += l_g_pix
                loss_dict['l_c_pix'] = l_g_pix
            else:
                l_g_pix = self.cri_pix(self.output, l1_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.opt['low_f_loss']:
                l_l_pix = self.cri_pix(self.output_low_f, self.gt_low_f) * self.opt['low_f_loss_weight']
                l_g_total += l_l_pix
                loss_dict['l_l_loss'] = l_l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.output, percep_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # # gan loss (relativistic gan)
            # real_d_pred = self.net_d(self.gt_grad).detach()
            # fake_g_pred = self.net_d(self.output_grad)
            # l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            # l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            # l_g_gan = (l_g_real + l_g_fake) / 2

            # gan loss
            fake_g_pred = self.net_d(self.output_grad)
            l_g_gan = self.cri_gan(fake_g_pred, True, is_disc=False)

            l_g_total += l_g_gan
            loss_dict['l_g_gan'] = l_g_gan

            l_g_total.backward()
            self.optimizer_g.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # # gan loss (relativistic gan)
        #
        # # In order to avoid the error in distributed training:
        # # "Error detected in CudnnBatchNormBackward: RuntimeError: one of
        # # the variables needed for gradient computation has been modified by
        # # an inplace operation",
        # # we separate the backwards for real and fake, and also detach the
        # # tensor for calculating mean.
        #
        # # real
        # fake_d_pred = self.net_d(self.output_grad).detach()
        # real_d_pred = self.net_d(self.gt_grad)
        # l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        # l_d_real.backward()
        # # fake
        # fake_d_pred = self.net_d(self.output_grad.detach())
        # l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        # l_d_fake.backward()
        # self.optimizer_d.step()

        # real
        real_d_pred = self.net_d(self.gt_grad)
        l_d_real = self.cri_gan(real_d_pred, True, is_disc=True)
        l_d_real.backward()
        # fake
        fake_d_pred = self.net_d(self.output_grad.detach().clone())  # clone for pt1.9
        l_d_fake = self.cri_gan(fake_d_pred, False, is_disc=True)
        l_d_fake.backward()
        self.optimizer_d.step()

        loss_dict['l_d_real'] = l_d_real
        loss_dict['l_d_fake'] = l_d_fake
        loss_dict['out_d_real'] = torch.mean(real_d_pred.detach())
        loss_dict['out_d_fake'] = torch.mean(fake_d_pred.detach())

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)


    def wavelet_s(self, x, norm=False):
        LL, Hc = self.DWT2(x)
        Hc = Hc[0]
        if norm:
            LL, Hc = LL * 0.5, Hc * 0.5 + 0.5  # norm [0, 1]

        LH, HL, HH = Hc[:, :, 0, :, :], \
                     Hc[:, :, 1, :, :], \
                     Hc[:, :, 2, :, :]
        Hc = torch.cat((LH, HL, HH), dim=1)
        return LL, Hc

