import os
import yaml
import time
import torch
import torch.nn.functional as F
import torchvision
import numpy as np

from track1 import vis_fea_map

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


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

class Saver():
    def __init__(self, args, test=False):
        self.args = args
        if test:
            default_dir = os.path.join(args.result_dir, args.name)
        else:
            default_dir = os.path.join(args.experiment_dir, args.name)
        if os.path.exists(default_dir):
            new_name = default_dir + '_archived_' + get_time_str()
            print(f'Path already exists. Rename it to {new_name}', flush=True)
            os.rename(default_dir, new_name)
        os.makedirs(default_dir, exist_ok=True)
        self.display_dir = os.path.join(default_dir, 'training_progress')
        self.model_dir = os.path.join(default_dir, 'models')
        self.image_dir = os.path.join(default_dir, 'down_results')
        self.image_var_dir = os.path.join(default_dir, 'img_var_results')
        self.np_var_dir = os.path.join(default_dir, 'np_var_results')

        self.per_var_dir = os.path.join(default_dir, 'per_var_dir')
        self.np_per_var_dir = os.path.join(default_dir, 'np_per_var_dir')

        self.img_save_freq = args.img_save_freq

        ## make directory
        if not os.path.exists(self.display_dir): os.makedirs(self.display_dir)
        if not os.path.exists(self.model_dir): os.makedirs(self.model_dir)
        if not os.path.exists(self.image_dir): os.makedirs(self.image_dir)
        if not os.path.exists(self.image_var_dir): os.makedirs(self.image_var_dir)
        if not os.path.exists(self.np_var_dir): os.makedirs(self.np_var_dir)
        if not os.path.exists(self.per_var_dir): os.makedirs(self.per_var_dir)
        if not os.path.exists(self.np_per_var_dir): os.makedirs(self.np_per_var_dir)

        config = os.path.join(default_dir,'config.yml')
        with open(config, 'w') as outfile:
            yaml.dump(args.__dict__, outfile, default_flow_style=False)

    def write_img_color_down(self, ep, model):
        if (ep + 1) % self.img_save_freq == 0:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_%04d.png' % (self.display_dir, ep)
            torchvision.utils.save_image(assembled_images, img_filename, nrow=1)
            img_filename = '%s/gen_%04d_grad.png' % (self.display_dir, ep)
            torchvision.utils.save_image(color_gradient(assembled_images), img_filename, nrow=1)
        elif ep == -1:
            assembled_images = model.assemble_outputs()
            img_filename = '%s/gen_last.png' % (self.display_dir)
            torchvision.utils.save_image(assembled_images, img_filename, nrow=1)
            img_filename = '%s/gen_last_grad.png' % (self.display_dir)
            torchvision.utils.save_image(color_gradient(assembled_images), img_filename, nrow=1)

    ## save result images
    def write_img_LR(self, ep, num, model, args, fn):
        result_savepath = os.path.join(self.image_dir, 'ep_%04d'%ep)
        filename = fn[0].split('.')[0]

        if not os.path.exists(result_savepath):
            os.mkdir(result_savepath)

        images_list = model.get_outputs()
        
        img_filename = os.path.join(result_savepath, '%s.png'%(filename))
        torchvision.utils.save_image(images_list[1], img_filename, nrow=1)

    def write_img_var(self, ep, num, model, args, fn):
        img_var = model.get_outputs()[2]
        if img_var is not None:
            # result_img_var_savepath = os.path.join(self.image_var_dir, 'ep_%04d' % ep)
            result_img_var_savepath = os.path.join(self.image_var_dir, 'latest')
            if not os.path.exists(result_img_var_savepath):
                os.mkdir(result_img_var_savepath)

            # result_np_var_savepath = os.path.join(self.np_var_dir, 'ep_%04d' % ep)
            result_np_var_savepath = os.path.join(self.np_var_dir, 'latest')
            if not os.path.exists(result_np_var_savepath):
                os.mkdir(result_np_var_savepath)

            filename = fn[0].split('.')[0]

            img_var_filename = os.path.join(result_img_var_savepath, '%s_x%s.png' % (filename, args.scale))
            np_var_filename = os.path.join(result_np_var_savepath, '%s_x%s.npy' % (filename, args.scale))
            # img_var = torch.exp(img_var)
            vis_fea_map.draw_features(img_var.cpu().numpy(),img_var_filename)
            np.save(np_var_filename, img_var.cpu().numpy())

    def write_per_var_1(self, ep, num, model, args, fn):
        per_var = model.get_outputs()[3]
        if per_var is not None:
            # result_img_var_savepath = os.path.join(self.image_var_dir, 'ep_%04d' % ep)
            result_per_var_savepath = os.path.join(self.per_var_dir, 'latest')
            if not os.path.exists(result_per_var_savepath):
                os.mkdir(result_per_var_savepath)

            # result_np_var_savepath = os.path.join(self.np_var_dir, 'ep_%04d' % ep)
            result_np_per_var_savepath = os.path.join(self.np_per_var_dir, 'latest')
            if not os.path.exists(result_np_per_var_savepath):
                os.mkdir(result_np_per_var_savepath)

            filename = fn[0].split('.')[0]

            img_var_filename = os.path.join(result_per_var_savepath, '%s_x%s.png' % (filename, args.scale))
            np_var_filename = os.path.join(result_np_per_var_savepath, '%s_x%s.npy' % (filename, args.scale))
            # img_var = torch.exp(img_var)
            vis_fea_map.draw_features_per(8, 8, per_var.cpu().numpy(),img_var_filename)
            np.save(np_var_filename, per_var.cpu().numpy())

    def write_per_var_3(self, ep, num, model, args, fn):
        per_var = model.get_outputs()[3]
        if per_var is not None:
            # result_img_var_savepath = os.path.join(self.image_var_dir, 'ep_%04d' % ep)
            result_per_var_savepath = os.path.join(self.per_var_dir, 'latest')
            if not os.path.exists(result_per_var_savepath):
                os.mkdir(result_per_var_savepath)

            # result_np_var_savepath = os.path.join(self.np_var_dir, 'ep_%04d' % ep)
            result_np_per_var_savepath = os.path.join(self.np_per_var_dir, 'latest')
            if not os.path.exists(result_np_per_var_savepath):
                os.mkdir(result_np_per_var_savepath)

            filename = fn[0].split('.')[0]

            img_var_filename = os.path.join(result_per_var_savepath, '%s_x%s.png' % (filename, args.scale))
            np_var_filename = os.path.join(result_np_per_var_savepath, '%s_x%s.npy' % (filename, args.scale))
            # img_var = torch.exp(img_var)
            vis_fea_map.draw_features_per(16, 24, per_var.cpu().numpy(),img_var_filename)
            np.save(np_var_filename, per_var.cpu().numpy())

    def write_per_var_5(self, ep, num, model, args, fn):
        per_var = model.get_outputs()[3]
        if per_var is not None:
            # result_img_var_savepath = os.path.join(self.image_var_dir, 'ep_%04d' % ep)
            result_per_var_savepath = os.path.join(self.per_var_dir, 'latest')
            if not os.path.exists(result_per_var_savepath):
                os.mkdir(result_per_var_savepath)

            # result_np_var_savepath = os.path.join(self.np_var_dir, 'ep_%04d' % ep)
            result_np_per_var_savepath = os.path.join(self.np_per_var_dir, 'latest')
            if not os.path.exists(result_np_per_var_savepath):
                os.mkdir(result_np_per_var_savepath)

            filename = fn[0].split('.')[0]

            img_var_filename = os.path.join(result_per_var_savepath, '%s_x%s.png' % (filename, args.scale))
            np_var_filename = os.path.join(result_np_per_var_savepath, '%s_x%s.npy' % (filename, args.scale))
            # img_var = torch.exp(img_var)
            vis_fea_map.draw_features_per(16, 16, per_var.cpu().numpy(),img_var_filename)
            np.save(np_var_filename, per_var.cpu().numpy())

    ## save generator
    def write_model_down(self, ep, total_it, model):
        if ep != -1:
            print('save the down generator @ ep %d' % (ep))
            model.state_save('%s/training_down_%04d.pth' % (self.model_dir, ep), ep, total_it)
            model.model_save('%s/model_down_%04d.pth' % (self.model_dir, ep), ep, total_it)
        else:
            model.state_save('%s/training_down_last.pth' % (self.model_dir), ep, total_it)
            model.model_save('%s/model_down_last.pth' % (self.model_dir), ep, total_it)


