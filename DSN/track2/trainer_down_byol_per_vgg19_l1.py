import time
import torch
import numpy as np
import torch.nn as nn
from importlib import import_module
from bicubic_pytorch import core
import random
import torch.nn.functional as F
import torchvision

from track2.data_loss import get_data_loss, FilterHigh

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class AdaptiveDownsamplingModel(nn.Module):
    def __init__(self, args):
        super(AdaptiveDownsamplingModel, self).__init__()

        self.args = args
        self.gpu = args.gpu
        self.data_loss_type = args.data_loss_type  # data loss option

        gen_module = import_module('generator.' + args.gen_model.lower())
        self.gen = gen_module.make_model(args)
        self.gen.cuda(args.gpu)

        if self.args.perceptual_loss == 'VGG19':
            self.l1_vgg19 = PerceptualLossVGG19().cuda(self.args.gpu)
        elif self.args.perceptual_loss == 'VGG19_Per':
            self.l1_vgg19_per = PerceptualLossVGG19_Per().cuda(self.args.gpu)

        if self.args.phase == 'train':
            dis_module = import_module('discriminator.' + args.dis_model.lower())
            self.dis = dis_module.make_model(args)  # discriminators
            self.dis.cuda(args.gpu)

            self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=args.lr_down, betas=(0.9, 0.999))
            self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=args.lr_down, betas=(0.9, 0.999))

            self.gen_sch = torch.optim.lr_scheduler.MultiStepLR(self.gen_opt, args.milestones, gamma=0.5)
            self.dis_sch = torch.optim.lr_scheduler.MultiStepLR(self.dis_opt, args.milestones, gamma=0.5)

            print('Data loss type : ', args.data_loss_type)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    ## update images to generator
    def update_img(self, img_s, img_t=None, img_s_k=None):
        real_gt = img_s
        if img_t is not None:
            real_lq = img_t

            if self.args.variance > 0:
                lq_var = torch.var(img_t, dim=(1, 2, 3)) * 255 * 255
                lq_var_np = lq_var.numpy()
                gt_var = torch.var(img_s, dim=(1, 2, 3)) * 255 * 255
                gt_var_np = gt_var.numpy()

                lq_gt_pair_list = []
                for i in range(len(lq_var_np)):
                    for j in range(len(gt_var_np)):
                        if abs(lq_var_np[i] - gt_var_np[j]) < self.args.variance:
                            lq_gt_pair_list.append([i, j])
                if len(lq_gt_pair_list) < self.args.variance_batch_size:
                    print(len(lq_gt_pair_list))
                while len(lq_gt_pair_list) < self.args.variance_batch_size:
                    i = np.random.randint(low=0, high=self.args.batch_size - 1)
                    j = np.random.randint(low=0, high=self.args.batch_size - 1)
                    lq_gt_pair_list.append([i, j])

                lq_list, gt_list = [], []
                lq_gt_pair_idx = np.random.randint(low=0, high=len(lq_gt_pair_list) - 1,
                                                   size=self.args.variance_batch_size)
                for idx in lq_gt_pair_idx:
                    lq_gt_pair = lq_gt_pair_list[idx]
                    lq_np, gt_np = img_t[lq_gt_pair[0], :, :, :].numpy(), img_s[lq_gt_pair[1], :, :, :].numpy()
                    lq_list.append(lq_np), gt_list.append(gt_np)

                real_lq = torch.tensor(lq_list)
                real_gt = torch.tensor(gt_list)

            self.img_t = real_lq.cuda(self.args.gpu).detach()

        if img_s_k is not None:
            self.img_s_k = img_s_k.cuda(self.args.gpu).detach()

        self.img_s = real_gt.cuda(self.args.gpu).detach()

        self.loss_dis = 0
        self.loss_gen = 0
        self.loss_data = 0

    ## generating LR iamges
    def generate_LR(self):
        if self.args.use_contrastive_loss and self.img_s.size()[0] > 1:
            self.img_gen, self.contract = self.gen(self.img_s, self.img_t)
        else:
            self.img_gen = self.gen(self.img_s)

        if isinstance(self.img_gen,list):
            if len(self.img_gen) == 3:
                self.per_var = self.img_gen[2]
            self.img_var = self.img_gen[1]
            self.img_gen = self.img_gen[0]
        else:
            self.img_var = None
            self.per_var = None

            ## update discriminator D
    def update_D(self):
        self.dis_opt.zero_grad()

        if self.data_loss_type == 'kl_gau_bic':
            filter = FilterHigh(recursions=1, stride=1, kernel_size=self.args.kernel_size, include_pad=False,
                                gaussian=True).cuda(self.args.gpu)
            img_t = filter(self.img_t)
            img_gen = filter(self.img_gen)
        elif self.data_loss_type == 'kl_gau_ker':
            filter = FilterHigh(recursions=1, stride=1, kernel_size=self.args.kernel_size, include_pad=False,
                                gaussian=True).cuda(self.args.gpu)
            img_t = filter(self.img_t)
            img_gen = filter(self.img_gen)
        else:
            img_t = self.img_t
            img_gen = self.img_gen

        loss_D, fake_score, real_score = self.backward_D_gan(self.dis, img_t, img_gen)

        self.loss_dis = loss_D.item()

        self.fake_score = fake_score.mean().data.item()
        self.real_score = real_score.mean().data.item()

        self.dis_opt.step()

    ## update generator G
    def update_G(self):
        self.gen_opt.zero_grad()

        if self.data_loss_type == 'kl_gau_bic':
            filter = FilterHigh(recursions=1, stride=1, kernel_size=self.args.kernel_size, include_pad=False, gaussian=True).cuda(self.args.gpu)
            img_gen = filter(self.img_gen)
        elif self.data_loss_type == 'kl_gau_ker':
            filter = FilterHigh(recursions=1, stride=1, kernel_size=self.args.kernel_size, include_pad=False, gaussian=True).cuda(self.args.gpu)
            img_gen = filter(self.img_gen)
        else:
            img_gen = self.img_gen

        loss_gan = self.backward_G_gan(img_gen, self.dis) * self.args.gan_ratio
        if self.img_var is None:
            loss_data = get_data_loss(self.img_s, self.img_s_k, self.img_gen, self.data_loss_type,
                                      self.args) * self.args.data_ratio
        else:
            loss_data = get_data_loss(self.img_s, self.img_s_k, self.img_gen, self.data_loss_type, self.args,
                                      var=self.img_var) * self.args.data_ratio
        loss_G = loss_gan + loss_data

        if self.args.use_contrastive_loss:
            loss_con = self.regression_loss(self.contract[0], self.contract[1])
            if self.args.use_contrastive_loss == 2:
                loss_con += self.regression_loss(self.contract[2], self.contract[3])
            loss_con = loss_con.mean() * self.args.con_ratio
            loss_G += loss_con.mean()

        if self.args.perceptual_loss == 'VGG19':
            if self.data_loss_type == 'kl_gau_ker':
                loss_per = self.l1_vgg19(self.img_gen, self.img_s_k) * self.args.per_ratio
            else:
                loss_per = self.l1_vgg19(self.img_gen, core.imresize(self.img_s, scale=1 / int(self.args.scale))) * self.args.per_ratio
            loss_G += loss_per
        elif self.args.perceptual_loss == 'VGG19_Per':
            if self.data_loss_type == 'kl_gau_ker':
                loss_per = self.l1_vgg19_per(self.img_gen, self.img_s_k, self.per_var) * self.args.per_ratio
            else:
                loss_per = self.l1_vgg19_per(self.img_gen, core.imresize(self.img_s, scale=1 / int(self.args.scale)), self.per_var) * self.args.per_ratio
            loss_G += loss_per
        loss_G.backward()  # retain_graph=True)

        self.loss_gen = loss_gan.item()
        self.loss_data = loss_data.item()
        self.loss_total = loss_G.item()
        if self.args.use_contrastive_loss:
            self.loss_con = loss_con.item()
        if self.args.perceptual_loss:
            self.loss_per = loss_per.item()
        self.gen_opt.step()

        if self.args.use_contrastive_loss:
            self.gen._update_target_network_parameters()  # update the key encoder

    ## loss function for discriminator D
    ## real to ones, and fake to zeros
    def backward_D_gan(self, netD, real, fake):
        pred_fake = netD.forward(fake.detach())
        pred_real = netD.forward(real)
        loss_D = 0
        fake_score = 0
        real_score = 0
        cnt = 0
        for it, (out_a, out_b) in enumerate(zip(pred_fake, pred_real)):
            out_fake = torch.sigmoid(out_a).clamp(min=0.0, max=1.0)
            out_real = torch.sigmoid(out_b).clamp(min=0.0, max=1.0)
            all0 = torch.zeros_like(out_fake).cuda(self.gpu).clamp(min=0.0, max=1.0)
            all1 = torch.ones_like(out_real).cuda(self.gpu).clamp(min=0.0, max=1.0)
            ad_fake_loss = nn.functional.binary_cross_entropy(out_fake, all0)
            ad_true_loss = nn.functional.binary_cross_entropy(out_real, all1)
            loss_D += ad_true_loss + ad_fake_loss
            fake_score += out_fake
            real_score += out_real
            cnt += 1
        loss_D /= cnt
        fake_score /= cnt
        real_score /= cnt
        loss_D.backward()
        return loss_D, fake_score, real_score

    def backward_G_gan(self, fake, netD=None):
        outs_fake = netD.forward(fake)
        loss_G = 0
        cnt = 0
        for out_a in outs_fake:
            outputs_fake = torch.sigmoid(out_a)
            all_ones = torch.ones_like(outputs_fake).cuda(self.gpu)
            loss_G += nn.functional.binary_cross_entropy(outputs_fake, all_ones)
            cnt += 1
        loss_G /= cnt
        return loss_G

    def resume(self, model_dir, train=True):
        checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
        self.gen.load_state_dict(checkpoint['gen'])
        if train:
            self.dis.load_state_dict(checkpoint['dis'])
            self.dis_opt.load_state_dict(checkpoint['dis_opt'])
            self.gen_opt.load_state_dict(checkpoint['gen_opt'])
        return checkpoint['ep'], checkpoint['total_it']

    def state_save(self, filename, ep, total_it):
        state = {'dis': self.dis.state_dict(),
                 'gen': self.gen.state_dict(),
                 'dis_opt': self.dis_opt.state_dict(),
                 'gen_opt': self.gen_opt.state_dict(),
                 'ep': ep,
                 'total_it': total_it
                 }
        time.sleep(5)
        torch.save(state, filename)
        return

    def model_save(self, filename, ep, total_it):
        state = {'dis': self.dis.state_dict(),
                 'gen': self.gen.state_dict(),
                 }
        time.sleep(5)
        torch.save(state, filename)
        return

    def assemble_outputs(self):
        images_source = self.img_s.detach()

        images_target = torch.zeros_like(self.img_s)  # template
        margin = (self.img_s.shape[2] - self.img_t.shape[2]) // 2
        images_target[:, :, margin:-margin, margin:-margin] = self.img_t.detach()

        margin = (self.img_s.shape[2] - self.img_gen.shape[2]) // 2

        if self.data_loss_type == 'kl_gau_ker':
            images_kernel = torch.zeros_like(self.img_s)  # template
            images_kernel[:, :, margin:-margin, margin:-margin] = self.img_s_k.detach()
        else:
            images_kernel = torch.zeros_like(self.img_s)  # template
            images_kernel[:, :, margin:-margin, margin:-margin] = core.imresize(self.img_s,
                                                                                scale=1 / int(self.args.scale)).detach()

        images_generated = torch.zeros_like(self.img_s)  # template
        images_generated[:, :, margin:-margin, margin:-margin] = self.img_gen.detach()

        images_blank = torch.zeros_like(self.img_s).detach()  # blank

        row1 = torch.cat((images_source[0:1, ::], images_kernel[0:1, ::], images_generated[0:1, ::]), 3)
        row2 = torch.cat((images_target[0:1, ::], images_blank[0:1, ::], images_blank[0:1, ::]), 3)

        return torch.cat((row1, row2), 2)

    def get_outputs(self):
        img_s = self.img_s.detach()
        img_gen = self.img_gen.detach()
        if self.img_var is not None:
            img_var = self.img_var.detach()
        else:
            img_var = None

        if self.per_var is not None:
            per_var = self.per_var.detach()
        else:
            per_var = None

        return [img_s, img_gen, img_var, per_var]


class VGGFeatureExtractor(nn.Module):
    def __init__(self, feature_layer=34, use_bn=False, use_input_norm=True,
                 device=torch.device('cpu')):
        super(VGGFeatureExtractor, self).__init__()
        self.use_input_norm = use_input_norm
        if use_bn:
            model = torchvision.models.vgg19_bn(pretrained=True)
        else:
            model = torchvision.models.vgg19(pretrained=True)
        if self.use_input_norm:
            mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
            # [0.485 - 1, 0.456 - 1, 0.406 - 1] if input in range [-1, 1]
            std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
            # [0.229 * 2, 0.224 * 2, 0.225 * 2] if input in range [-1, 1]
            self.register_buffer('mean', mean)
            self.register_buffer('std', std)
        self.features = nn.Sequential(*list(model.features.children())[:(feature_layer + 1)])
        # No need to BP to variable
        for k, v in self.features.named_parameters():
            v.requires_grad = False

    def forward(self, x):
        # Assume input range is [0, 1]
        if self.use_input_norm:
            x = (x - self.mean) / self.std
        output = self.features(x)
        return (output * 0.5 + 0.5)

# if __name__ == '__main__':
#     input = torch.ones([16,3,64,64])
#     model = VGGFeatureExtractor(feature_layer=34)
#     output = model(input)
#     print(output.size())  # torch.Size([16, 512, 4, 4])

class PerceptualLossVGG19_Per(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG19_Per, self).__init__()
        loss_network = VGGFeatureExtractor(feature_layer=34).eval()
        if torch.cuda.is_available():
            loss_network = loss_network.cuda()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        # self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y, per_var):
        s = torch.exp(per_var)
        x = self.loss_network(x) * 255
        y = self.loss_network(y) * 255
        return (self.l1_loss(x, y) + torch.mean(torch.mul(s, torch.exp(-torch.div(abs(x - y), s))) - per_var - 1)) / 255

class PerceptualLossVGG19(nn.Module):
    def __init__(self):
        super(PerceptualLossVGG19, self).__init__()
        loss_network = VGGFeatureExtractor(feature_layer=34).eval()
        if torch.cuda.is_available():
            loss_network = loss_network.cuda()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        # self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        return self.l1_loss(self.loss_network(x), self.loss_network(y))
