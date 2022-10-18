import os
import numpy as np
import glob
import tqdm
import torch
import random
import pickle
import imageio
import torch.utils.data as data
from PIL import Image, ImageFile

from torchvision.transforms import Compose, RandomCrop, ToTensor
ImageFile.LOAD_TRUNCATED_IMAGES = True


class noiseDataset(data.Dataset):
    def __init__(self, dataset='x2/', size=32):
        super(noiseDataset, self).__init__()

        base = dataset
        import os
        assert os.path.exists(base)

        # self.mat_files = sorted(glob.glob(base + '*.mat'))
        self.noise_imgs = sorted(glob.glob(base + '/' + '*.png'))
        self.pre_process = Compose([RandomCrop(size), ToTensor()])

    def __getitem__(self, index):
        noise = self.pre_process(Image.open(self.noise_imgs[index]))
        norm_noise = (noise - torch.mean(noise, dim=[1, 2], keepdim=True))
        return norm_noise

    def __len__(self):
        return len(self.noise_imgs)


class unpaired_dataset(data.Dataset):
    def __init__(self, args, phase='train'):
        if phase == 'train':
            self.dataroot = args.train_dataroot
            Source_type = args.train_source
            Target_type = args.train_target
            if args.add_noise:
                noiseroot= os.path.join(self.dataroot, args.noise_data)
                self.noises = noiseDataset(noiseroot, args.patch_size_down//int(args.scale))
        else:
            self.dataroot = args.test_dataroot
            Source_type = args.valid_source
            Target_type = args.valid_target

        self.args = args

        ## Source
        images_source = sorted(os.listdir(os.path.join(self.dataroot, Source_type)))
        self.images_source = [os.path.join(self.dataroot, Source_type, x) for x in images_source]
        ## Target
        images_target = sorted(os.listdir(os.path.join(self.dataroot, Target_type)))
        self.images_target = [os.path.join(self.dataroot, Target_type, x) for x in images_target]

        self.phase = phase
        self.binary = False

        print('\nphase: {}'.format(phase))

        ## checking or making binary files to boost loading speed
        if not args.nobin and not os.path.exists(os.path.join(self.dataroot, 'bin')):
            os.mkdir(os.path.join(self.dataroot, 'bin'))
        if not args.nobin:
            if not os.path.exists(os.path.join(self.dataroot, 'bin', Source_type)):
                os.mkdir(os.path.join(self.dataroot, 'bin', Source_type))
                print('no binary file for Source is detected')
                print('making binary for Source ...')
                for i in tqdm.tqdm(range(len(self.images_source))):
                    f = os.path.join(self.dataroot, 'bin', Source_type, self.images_source[i].split('/')[-1].split('.')[0]+'.pt')
                    with open(f, 'wb') as _f:
                        pickle.dump(imageio.imread(self.images_source[i]), _f)
                print('Done')
                self.binary = True
            else:
                print('binary files for {} already exist'.format(Source_type))
                self.binary = True

            if not os.path.exists(os.path.join(self.dataroot, 'bin', Target_type)):
                os.mkdir(os.path.join(self.dataroot, 'bin', Target_type))
                print('no binary file for {} are detected'.format(Target_type))
                print('making binary for {} ...'.format(Target_type))
                for j in tqdm.tqdm(range(len(self.images_target))):
                    f = os.path.join(self.dataroot, 'bin', Target_type, self.images_target[j].split('/')[-1].split('.')[0]+'.pt')
                    with open(f, 'wb') as _f:
                        pickle.dump(imageio.imread(self.images_target[j]), _f)
                print('Done')
                self.binary = True
            else:
                if phase == 'train':
                    print('binary files for {} already exist'.format(Target_type))
                self.binary = True
        else:
            print('do not use binary files')

        ## change base folder to bin if binary option is enabled
        if self.binary:
            images_source = sorted(os.listdir(os.path.join(self.dataroot, 'bin', Source_type))) 
            images_target = sorted(os.listdir(os.path.join(self.dataroot, 'bin', Target_type)))
            self.images_source = [os.path.join(self.dataroot, 'bin', Source_type, x) for x in images_source]
            self.images_target = [os.path.join(self.dataroot, 'bin', Target_type, x) for x in images_target]

        self.images_source_size = len(self.images_source)
        self.images_target_size = len(self.images_target)
    

        ## since we are dealing with unpaired setting,
        ## we can not assure same dataset size between source and target domain.
        ## Therefore we just set dataset size as maximum value of two.
        # self.dataset_size = int(max(patches_source_size, patches_target_size))
        # if phase == 'test':
        #     self.dataset_size = self.images_source_size

        if self.phase == 'train':
            transforms_source = [RandomCrop(args.patch_size_down)]
            transforms_target = [RandomCrop(args.patch_size_down//int(args.scale))]
        else:
            transforms_source = []
            transforms_target = []

        transforms_source.append(ToTensor())
        self.transforms_source = Compose(transforms_source)

        transforms_target.append(ToTensor())
        self.transforms_target = Compose(transforms_target)
            
        if phase == 'train':
            print('Source: %d, Target: %d images'%(self.images_source_size, self.images_target_size))
        else:
            print('Source: %d'%(self.images_source_size))


    def __getitem__(self, index):
        # index_source = index % self.images_source_size
        if self.phase == 'train':
            index_source = random.randint(0, self.images_source_size - 1)  ## for randomness
            index_target = index % self.images_target_size
        else:
            index_source = index
            index_target = index
        data_source, fn = self.load_img(self.images_source[index_source])
        data_target, _ = self.load_img(self.images_target[index_target], domain='target')

        return data_source, data_target, fn

    def load_img(self, img_name, input_dim=3, domain='source'):
        ## loading images
        if self.binary:
            with open(img_name, 'rb') as _f:
                img = pickle.load(_f)
            img = Image.fromarray(img)
        else: 
            img = Image.open(img_name).convert('RGB')
        fn = img_name.split('/')[-1]
        
        ## apply different transfomation along domain
        if domain == 'source':
            img = self.transforms_source(img)
        else:
            img = self.transforms_target(img)

        if self.phase == 'train':
            ## rotating
            rot = self.args.rot and random.random() < 0.5
            if rot:
                img = img.transpose(1, 2)

            ## flipping
            flip_h = self.args.flip and random.random() < 0.5
            flip_v = self.args.flip and random.random() < 0.5
            if flip_h:
                img = torch.flip(img, [2])
            if flip_v:
                img = torch.flip(img, [1])

            if self.args.add_noise and domain == 'target':
                noise = self.noises[np.random.randint(0, len(self.noises))]
                img = torch.clamp(img + noise, 0, 1)

        return img, fn

    def __len__(self):
        if self.args.variance > 0 and self.phase == 'train':
            return self.images_target_size * 2
        else:
            return self.images_target_size


class paired_dataset(data.Dataset): # only for joint SR
    def __init__(self, args):
        self.dataroot = args.test_dataroot
        self.args = args

        if args.realsr:
            test_hr = args.test_lr
        else:
            if args.test_hr is None:
                raise NotImplementedError("test_hr set should be given")
            test_hr = args.test_hr
        
        ## HR
        images_hr = sorted(os.listdir(os.path.join(self.dataroot, test_hr)))
        images_hr = images_hr[int(args.test_range.split('-')[0])-1: int(args.test_range.split('-')[1]) ]
        self.images_hr = [os.path.join(self.dataroot, test_hr, x) for x in images_hr]
        ## LR
        images_lr = sorted(os.listdir(os.path.join(self.dataroot, args.test_lr)))
        images_lr = images_lr[int(args.test_range.split('-')[0])-1: int(args.test_range.split('-')[1]) ]
        self.images_lr = [os.path.join(self.dataroot, args.test_lr, x) for x in images_lr]

        self.images_hr_size = len(self.images_hr)
        self.images_lr_size = len(self.images_lr)

        assert(self.images_hr_size == self.images_lr_size)

        transforms = []
        transforms.append(ToTensor())
        self.transforms = Compose(transforms)


        print('\njoint training option is enabled')
        print('HR set: {},  LR set: {}'.format(args.test_hr, args.test_lr))
        print('number of test images for SR : %d images' %(self.images_hr_size))


    def __getitem__(self, index):
        data_hr, fn_hr = self.load_img(self.images_hr[index])
        data_lr, fn_lr = self.load_img(self.images_lr[index], domain='lr')

        return data_hr, data_lr, fn_lr

    def load_img(self, img_name, input_dim=3, domain='hr'):
        ## loading images
        img = Image.open(img_name).convert('RGB')
        fn = img_name.split('/')[-1]
        
        ## apply transfomation
        img = self.transforms(img)
    
        ## rotating and flipping
        rot = self.args.rot and random.random() < 0.5
        flip_h = self.args.flip and random.random() < 0.5
        flip_v = self.args.flip and random.random() < 0.5    
        if rot:
            img = img.transpose(1,2)
        if flip_h:
            img = torch.flip(img, [2])
        if flip_v:
            img = torch.flip(img, [1])

        return img, fn

    def __len__(self):
        return self.images_hr_size