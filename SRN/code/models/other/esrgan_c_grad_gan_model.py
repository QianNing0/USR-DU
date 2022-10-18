import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from code.archs import build_network
from code.utils import get_root_logger
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


class FilterHigh(nn.Module):
    def __init__(self, recursions=1, kernel_size=5, stride=1, include_pad=True, normalize=True, gaussian=False):
        super(FilterHigh, self).__init__()
        self.filter_low = FilterLow(recursions=1, kernel_size=kernel_size, stride=stride, include_pad=include_pad,
                                    gaussian=gaussian)
        self.recursions = recursions
        self.normalize = normalize

    def forward(self, img):
        if self.recursions > 1:
            for i in range(self.recursions - 1):
                img = self.filter_low(img)
        img = img - self.filter_low(img)
        if self.normalize:
            return 0.5 + img * 0.5
        else:
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
class ESRGANModel_C_Grad_GAN(SRGANModel):
    """ESRGAN model for single image super-resolution."""
    def __init__(self, opt):
        super(ESRGANModel_C_Grad_GAN, self).__init__(opt=opt)
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

    def init_training_settings(self):
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define network net_d
        self.net_d = build_network(self.opt['network_d'])
        self.net_d = self.model_to_device(self.net_d)
        self.print_network(self.net_d)

        # define network net_d2
        self.net_d2 = build_network(self.opt['network_d2'])
        self.net_d2 = self.model_to_device(self.net_d2)
        self.print_network(self.net_d2)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d', 'params')
            self.load_network(self.net_d, load_path, self.opt['path'].get('strict_load_d', True), param_key)

        load_path = self.opt['path'].get('pretrain_network_d2', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_d2', 'params')
            self.load_network(self.net_d2, load_path, self.opt['path'].get('strict_load_d2', True), param_key)

        self.net_g.train()
        self.net_d.train()
        self.net_d2.train()

        # define losses
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if train_opt.get('gan_opt'):
            self.cri_gan = build_loss(train_opt['gan_opt']).to(self.device)

        self.net_d_iters = train_opt.get('net_d_iters', 1)
        self.net_d_init_iters = train_opt.get('net_d_init_iters', 0)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()


    def setup_optimizers(self):
        train_opt = self.opt['train']
        # optimizer g
        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, self.net_g.parameters(), **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)
        # optimizer d
        optim_type = train_opt['optim_d'].pop('type')
        self.optimizer_d = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d'])
        self.optimizers.append(self.optimizer_d)
        # optimizer d2
        optim_type = train_opt['optim_d2'].pop('type')
        self.optimizer_d2 = self.get_optimizer(optim_type, self.net_d.parameters(), **train_opt['optim_d2'])
        self.optimizers.append(self.optimizer_d2)

    def feed_data(self, data):
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'lq' in data:
            self.lq = data['lq'].to(self.device)
        else:
            fake_lq = data['fake_lq']
            real_gt = data['real_gt']
            real_lq = data['real_lq']
            unpair_real_gt = data['unpair_real_gt']

            if self.opt['variance'] > 0:
                lq_var = torch.var(real_lq, dim=(1, 2, 3)) * 255 * 255
                lq_var_np = lq_var.numpy()
                unpair_gt_var = torch.var(unpair_real_gt, dim=(1, 2, 3)) * 255 * 255
                unpair_gt_var_np = unpair_gt_var.numpy()
                lq_unpair_gt_pair_list = []
                for i in range(len(lq_var_np)):
                    for j in range(len(unpair_gt_var_np)):
                        if abs(lq_var_np[i] - unpair_gt_var_np[j]) < self.opt['variance']:
                            lq_unpair_gt_pair_list.append([i, j])
                if len(lq_unpair_gt_pair_list) < self.opt['variance_batch_size']:
                    print(len(lq_unpair_gt_pair_list))
                while len(lq_unpair_gt_pair_list) < self.opt['variance_batch_size']:
                    i = np.random.randint(low=0, high=self.opt['datasets']['train']['batch_size_per_gpu'] - 1)
                    j = np.random.randint(low=0, high=self.opt['datasets']['train']['batch_size_per_gpu'] - 1)
                    lq_unpair_gt_pair_list.append([i, j])
                lq_unpair_gt_pair_idx = np.random.randint(low=0, high=len(lq_unpair_gt_pair_list) - 1,
                                                   size=self.opt['variance_batch_size'])
                real_lq_list, unpair_gt_list = [], []
                for idx in lq_unpair_gt_pair_idx:
                    lq_unpair_gt_pair = lq_unpair_gt_pair_list[idx]
                    real_lq_np, unpair_gt_np = real_lq[lq_unpair_gt_pair[0], :, :, :].numpy(), unpair_real_gt[lq_unpair_gt_pair[1], :, :, :].numpy()
                    real_lq_list.append(real_lq_np), unpair_gt_list.append(unpair_gt_np)
                real_lq = torch.tensor(real_lq_list)
                unpair_real_gt = torch.tensor(unpair_gt_list)

                lq_gt_pair_idx = np.random.randint(low=0, high=self.opt['datasets']['train']['batch_size_per_gpu'] - 1,
                                                          size=self.opt['variance_batch_size'])
                fake_lq_list, real_gt_list = [], []
                for idx in lq_gt_pair_idx:
                    fake_lq_np, real_gt_np = fake_lq[idx, :, :, :].numpy(), real_gt[idx, :, :, :].numpy()
                    fake_lq_list.append(fake_lq_np), real_gt_list.append(real_gt_np)
                fake_lq = torch.tensor(fake_lq_list)
                real_gt = torch.tensor(real_gt_list)

            self.fake_lq = fake_lq.to(self.device)
            self.real_gt = real_gt.to(self.device)
            self.real_lq = real_lq.to(self.device)
            self.unpair_real_gt = unpair_real_gt.to(self.device)

    def optimize_parameters(self, current_iter):
        # optimize net_g
        for p in self.net_d.parameters():
            p.requires_grad = False

        self.optimizer_g.zero_grad()

        if self.opt['need_HR']:
            self.fake_lq_output = self.net_g([self.fake_lq, self.real_gt])
            self.real_lq_output = self.net_g([self.real_lq, self.real_gt])
        else:
            self.fake_lq_output = self.net_g(self.fake_lq)
            self.real_lq_output = self.net_g(self.real_lq)

        if self.opt['weighted_loss']:
            self.var = self.fake_lq_output[1]
            self.fake_lq_output = self.fake_lq_output[0]
            self.real_lq_output = self.real_lq_output[0]
        else:
            self.var = None

        self.fake_lq_output_grad = color_gradient(self.fake_lq_output)
        self.real_gt_grad = color_gradient(self.real_gt)

        self.real_lq_output_grad = color_gradient(self.real_lq_output)
        self.unpair_real_gt_grad = color_gradient(self.unpair_real_gt)

        self.fake_lq_output_low_f = self.fs(self.fake_lq_output)
        self.real_gt_low_f = self.fs(self.real_gt)

        l_g_total = 0
        loss_dict = OrderedDict()
        if (current_iter % self.net_d_iters == 0 and current_iter > self.net_d_init_iters):
            # pixel loss
            if self.opt['weighted_loss']:
                l_g_pix = self.UncertaintyLoss(self.fake_lq_output, self.real_gt, self.var / 255)
                l_g_total += l_g_pix
                loss_dict['l_c_pix'] = l_g_pix
            else:
                l_g_pix = self.cri_pix(self.fake_lq_output, self.real_gt)
                l_g_total += l_g_pix
                loss_dict['l_g_pix'] = l_g_pix

            if self.opt['low_f_loss']:
                l_l_pix = self.cri_pix(self.fake_lq_output_low_f, self.real_gt_low_f) * self.opt['low_f_loss_weight']
                l_g_total += l_l_pix
                loss_dict['l_l_loss'] = l_l_pix

            # perceptual loss
            if self.cri_perceptual:
                l_g_percep, l_g_style = self.cri_perceptual(self.fake_lq_output, self.real_gt)
                if l_g_percep is not None:
                    l_g_total += l_g_percep
                    loss_dict['l_g_percep'] = l_g_percep
                if l_g_style is not None:
                    l_g_total += l_g_style
                    loss_dict['l_g_style'] = l_g_style

            # # gan loss (relativistic gan)
            # real_d_pred = self.net_d(self.unpair_real_gt_high_f).detach()
            # fake_g_pred = self.net_d(self.real_lq_output_high_f)
            # l_g_real = self.cri_gan(real_d_pred - torch.mean(fake_g_pred), False, is_disc=False)
            # l_g_fake = self.cri_gan(fake_g_pred - torch.mean(real_d_pred), True, is_disc=False)
            # l_g_gan = (l_g_real + l_g_fake) / 2

            # gan loss
            fake_fake_lq_g_pred = self.net_d(self.fake_lq_output_grad)
            l_g_fake_lq_gan = self.cri_gan(fake_fake_lq_g_pred, True, is_disc=False)
            l_g_total += l_g_fake_lq_gan
            loss_dict['l_g_fake_lq_gan'] = l_g_fake_lq_gan

            fake_real_lq_g_pred = self.net_d2(self.real_lq_output_grad)
            l_g_real_lq_gan = self.cri_gan(fake_real_lq_g_pred, True, is_disc=False)
            l_g_total += l_g_real_lq_gan
            loss_dict['l_g_real_lq_gan'] = l_g_real_lq_gan

            l_g_total.backward()
            self.optimizer_g.step()


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
        # fake_d_pred = self.net_d(self.real_lq_output_high_f).detach()
        # real_d_pred = self.net_d(self.unpair_real_gt_high_f)
        # l_d_real = self.cri_gan(real_d_pred - torch.mean(fake_d_pred), True, is_disc=True) * 0.5
        # l_d_real.backward()
        # # fake
        # fake_d_pred = self.net_d(self.real_lq_output_high_f.detach())
        # l_d_fake = self.cri_gan(fake_d_pred - torch.mean(real_d_pred.detach()), False, is_disc=True) * 0.5
        # l_d_fake.backward()
        # self.optimizer_d.step()

        # optimize net_d
        for p in self.net_d.parameters():
            p.requires_grad = True

        self.optimizer_d.zero_grad()
        # real
        real_real_gt_d_pred = self.net_d(self.real_gt_grad)
        l_d_real_gt_real = self.cri_gan(real_real_gt_d_pred, True, is_disc=True)
        l_d_real_gt_real.backward()
        # fake
        fake_fake_lq_d_pred = self.net_d(self.fake_lq_output_grad.detach().clone())  # clone for pt1.9
        l_d_fake_lq_fake = self.cri_gan(fake_fake_lq_d_pred, False, is_disc=True)
        l_d_fake_lq_fake.backward()
        self.optimizer_d.step()



        # optimize net_d2
        for p in self.net_d2.parameters():
            p.requires_grad = True

        self.optimizer_d2.zero_grad()
        # real
        real_unpair_real_gt_d_pred = self.net_d2(self.unpair_real_gt_grad)
        l_d_unpair_real_gt_real = self.cri_gan(real_unpair_real_gt_d_pred, True, is_disc=True)
        l_d_unpair_real_gt_real.backward()
        # fake
        fake_real_lq_d_pred = self.net_d2(self.real_lq_output_grad.detach().clone())  # clone for pt1.9
        l_d_real_lq_fake = self.cri_gan(fake_real_lq_d_pred, False, is_disc=True)
        l_d_real_lq_fake.backward()
        self.optimizer_d2.step()

        loss_dict['l_d_real_gt_real'] = l_d_real_gt_real
        loss_dict['l_d_fake_lq_fake'] = l_d_fake_lq_fake
        loss_dict['l_d_unpair_real_gt_real'] = l_d_unpair_real_gt_real
        loss_dict['l_d_real_lq_fake'] = l_d_real_lq_fake

        loss_dict['out_d_real_gt_real'] = torch.mean(real_real_gt_d_pred.detach())
        loss_dict['out_d_fake_lq_fake'] = torch.mean(fake_fake_lq_d_pred.detach())
        loss_dict['out_d_unpair_real_gt_real'] = torch.mean(real_unpair_real_gt_d_pred.detach())
        loss_dict['out_d_real_lq_fake'] = torch.mean(fake_real_lq_d_pred.detach())


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

    def filter_func(self, x, norm=False):
        low_f, high_f = self.filter_low(x), self.filter_high(x)
        if norm:
            high_f = high_f * 0.5 + 0.5
        return low_f, high_f
