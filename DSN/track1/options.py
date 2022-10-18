import argparse
from datetime import datetime


class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ## dataset related
        # learning downsampling
        self.parser.add_argument('--test_visible', action='store_true', help='test_visible')
        self.parser.add_argument('--add_noise', action='store_true', help='add_noise')
        self.parser.add_argument('--noise_data', type=str, default='Corrupted_noise', help='Corrupted_noise or DPEDiphone_noise')

        self.parser.add_argument('--train_source', type=str, default='DIV2K_train_HR', help='Source type')
        self.parser.add_argument('--train_target', type=str, default='TrainingSource', help='target type')
        self.parser.add_argument('--valid_source', type=str, default='DIV2K_valid_HR', help='Source type')
        self.parser.add_argument('--valid_target', type=str, default='ValidationSource', help='target type')
        ## data loader related
        self.parser.add_argument('--train_dataroot', type=str, default='/home/tangjingzhu/Works/Dataset/DIV2K/', help='path of train data')
        self.parser.add_argument('--test_dataroot', type=str, default='/home/tangjingzhu/Works/Dataset/DIV2K/', help='path of test data')
        self.parser.add_argument('--phase', type=str, default='train', help='phase for dataloading')
        self.parser.add_argument('--batch_size', type=int, default=16, help='batch size')
        self.parser.add_argument('--variance', type=int, default=0, help='variance')
        self.parser.add_argument('--variance_batch_size', type=int, default=16, help='batch size')

        self.parser.add_argument('--patch_size_down', type=int, default=256, help='cropped image size for learning downsampling')
        self.parser.add_argument('--nThreads', type=int, default=6, help='# of threads for data loader')
        self.parser.add_argument('--flip', action='store_true', help='specified if flip')
        self.parser.add_argument('--rot', action='store_true', help='specified if rotate')
        self.parser.add_argument('--nobin', action='store_true', help='specified if not use bin')
        ## ouptput related
        self.parser.add_argument('--name', type=str, default='', help='folder name to save outputs')
        self.parser.add_argument('--experiment_dir', type=str, default='./experiments', help='path for train saving result images and models')
        self.parser.add_argument('--result_dir', type=str, default='./results', help='path for saving test result images and models')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')
        self.parser.add_argument('--make_down', action='store_true', help='specified if test')
        ## training related
        # common
        self.parser.add_argument('--gpu', type=str, default='cuda', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--scale', type=str, choices=('2', '3', '4'), default='4', help='scale to SR, only support [2, 4]')
        # learning downsampling
        self.parser.add_argument('--resume_down', type=str, default=None, help='load training states for resume the downsampling learning')
        self.parser.add_argument('--epochs_down', type=int, default=500, help='number of epochs for training downsampling')
        self.parser.add_argument('--lr_down', type=float, default=0.00005, help='learning rate for learning downsampling')
        self.parser.add_argument('--milestones', type=str, default='1000-1500-1750')

        ## experimnet related
        self.parser.add_argument('--save_snapshot', type=int, default=10, help='save snapshot')
        self.parser.add_argument('--save_results', action='store_true', help='enable saving intermediate image option')
        ## data loss related
        self.parser.add_argument('--data_loss_type', type=str, choices=('bic', 'avg', 'gau', 'avg_bic', 'gau_bic', 'wavelet_bic', 'kl_gau_bic', 'kl_grad_bic'), default='kl_gau_bic', help='type of available data type')
        # bic
        # avg
        self.parser.add_argument('--box_size', type=int, default=16, help='box size for filtering')
        # gau
        self.parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='gaussian std')
        self.parser.add_argument('--gaussian_ksize', type=int, default=16, help='gaussian kernel size')
        self.parser.add_argument('--gaussian_dense', action='store_true', help='option for dense gaussian')
        # avg_bic gau_bic wavelet_bic
        self.parser.add_argument('--kernel_size', type=int, default=5, help='kernel size for filtering')
        # balance loss
        self.parser.add_argument('--gan_ratio', type=float, default=1, help='ratio between loss')
        self.parser.add_argument('--data_ratio', type=float, default=100, help='ratio between loss')
        self.parser.add_argument('--use_contrastive_loss', type=int, choices=(1, 2, 0), default=0, help='option for use_contrastive_loss')
        self.parser.add_argument('--con_ratio', type=float, default=0, help='ratio between loss')
        self.parser.add_argument('--perceptual_loss', type=str, choices=('LPIPS', 'VGG19_Per', 'VGG19'), default='LPIPS', help='option for use_perceptual_loss')
        self.parser.add_argument('--per_ratio', type=float, default=1, help='ratio between loss')
        ## generator related
        # self.parser.add_argument('--gen_norm', type=str, default='Instance', help='normalization layer in generator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--gen_model', type=str, default='resnet_uncertainty', help='set generator architecture. (DSGAN, DeResnet)')
        # resnet n_res_blocks
        self.parser.add_argument('--n_res_blocks', type=int, default=8)
        # resnet_constract n_res_blocks K
        self.parser.add_argument('--K', type=int, default=8192)
        ## discriminator related
        self.parser.add_argument('--dis_model', type=str, default='unet', help='set discriminator architecture. (FSD, nld_s1, nld_s2)')
        # patch_gan
        self.parser.add_argument('--dis_norm', type=str, default='Instance', help='normalization layer in discriminator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--dis_inc', type=int, choices=(1, 3, 9), default=9, help='set discriminator architecture input channel')

    def parse(self):
        self.opt = self.parser.parse_args()

        if self.opt.name == '':
            self.opt.name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+self.opt.phase+'_'+self.opt.gen_model+'_'+self.opt.dis_model+'_'+self.opt.data_loss_type+'_'+self.opt.train_source+'_'+self.opt.train_target

        self.opt.milestones = list(map(lambda x: int(x), self.opt.milestones.split('-')))

        return self.opt

class TestOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        ## dataset related
        # learning downsampling
        self.parser.add_argument('--valid_source', type=str, default='DIV2K_valid_HR', help='Source type')
        ## data loader related
        self.parser.add_argument('--test_dataroot', type=str, default='/home/tangjingzhu/Works/Dataset/DIV2K/',
                                 help='path of test data')
        self.parser.add_argument('--phase', type=str, default='test', help='phase for dataloading')
        self.parser.add_argument('--nThreads', type=int, default=4, help='# of threads for data loader')
        self.parser.add_argument('--flip', action='store_true', help='specified if flip')
        self.parser.add_argument('--rot', action='store_true', help='specified if rotate')
        self.parser.add_argument('--nobin', action='store_true', help='specified if not use bin')
        ## ouptput related
        self.parser.add_argument('--name', type=str, default='', help='folder name to save outputs')
        self.parser.add_argument('--result_dir', type=str, default='./results',
                                 help='path for saving test result images and models')
        self.parser.add_argument('--img_save_freq', type=int, default=1, help='freq (epoch) of saving images')

        ## training related
        # common
        self.parser.add_argument('--gpu', type=str, default='cuda', help='gpu ids: e.g. 0  0,1,2, 0,2')
        self.parser.add_argument('--scale', type=str, choices=('2', '3', '4'), default='4',
                                 help='scale to SR, only support [2, 4]')
        # learning downsampling
        self.parser.add_argument('--resume_down', type=str, default=None,
                                 help='load training states for resume the downsampling learning')
        ## data loss related
        self.parser.add_argument('--data_loss_type', type=str,
                                 choices=('bic', 'avg', 'gau', 'avg_bic', 'gau_bic', 'wavelet_bic', 'kl_gau_bic', 'kl_grad_bic'),
                                 default='kl_gau_bic', help='type of available data type')
        # bic
        # avg
        self.parser.add_argument('--box_size', type=int, default=16, help='box size for filtering')
        # gau
        self.parser.add_argument('--gaussian_sigma', type=float, default=2.0, help='gaussian std')
        self.parser.add_argument('--gaussian_ksize', type=int, default=16, help='gaussian kernel size')
        self.parser.add_argument('--gaussian_dense', action='store_true', help='option for dense gaussian')
        # avg_bic gau_bic wavelet_bic
        self.parser.add_argument('--kernel_size', type=int, default=5, help='kernel size for filtering')
        # balance loss
        self.parser.add_argument('--gan_ratio', type=float, default=1, help='ratio between loss')
        self.parser.add_argument('--data_ratio', type=float, default=100, help='ratio between loss')
        self.parser.add_argument('--use_contrastive_loss', type=int, choices=(1, 2, 0), default=0, help='option for use_contrastive_loss')
        self.parser.add_argument('--con_ratio', type=float, default=1, help='ratio between loss')
        self.parser.add_argument('--perceptual_loss', type=str, choices=('LPIPS', 'VGG', None), default='LPIPS',
                                 help='option for use_perceptual_loss')
        self.parser.add_argument('--per_ratio', type=float, default=1, help='ratio between loss')
        ## generator related
        # self.parser.add_argument('--gen_norm', type=str, default='Instance', help='normalization layer in generator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--gen_model', type=str, default='resnet_uncertainty',
                                 help='set generator architecture. (DSGAN, DeResnet)')
        # resnet n_res_blocks
        self.parser.add_argument('--n_res_blocks', type=int, default=8)
        # resnet_constract n_res_blocks K
        self.parser.add_argument('--K', type=int, default=8192)
        ## discriminator related
        self.parser.add_argument('--dis_model', type=str, default='unet',
                                 help='set discriminator architecture. (FSD, nld_s1, nld_s2)')
        self.parser.add_argument('--dis_norm', type=str, default='Instance',
                                 help='normalization layer in discriminator [None, Batch, Instance, Layer]')
        self.parser.add_argument('--dis_inc', type=int, choices=(1, 3, 9), default=9,
                                 help='set discriminator architecture input channel')

    def parse(self):
        self.opt = self.parser.parse_args()
        args = vars(self.opt)
        print('\n--- loading options ---')
        for name, value in sorted(args.items()):
            print('%s: %s' % (str(name), str(value)))
        if self.opt.name == '':
            self.opt.name = datetime.now().strftime(
                '%Y-%m-%d_%H:%M:%S') + '_' + self.opt.phase + '_' + self.opt.gen_model + '_' + self.opt.dis_model + '_' + self.opt.data_loss_type + '_' + self.opt.train_source + '_' + self.opt.train_target

        return self.opt


    # def parse(self):
    #     self.opt = self.parser.parse_args()
    #     args = vars(self.opt)
    #     print('\n--- loading options ---')
    #     for name, value in sorted(args.items()):
    #         print('%s: %s' % (str(name), str(value)))
    #     # set irrelevant options
    #     self.opt.dis_norm = 'None'
    #     self.opt.dis_spectral_norm = False
    #
    #     if self.opt.name == '':
    #         self.opt.name = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')+'_'+self.opt.phase
    #
    #     return self.opt
