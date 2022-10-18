import os
import torch
from saver import Saver
from options import TestOptions
from PIL import Image
import torchvision.transforms.functional as TF
from tqdm import tqdm

from track1.trainer_down_byol_per_vgg19_l1 import AdaptiveDownsamplingModel

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

# parse options
parser = TestOptions()
args = parser.parse()


# daita loader
print('\nmaking dataset ...')
dataset = [os.path.join(args.test_dataroot, args.valid_source, x) for x in os.listdir(os.path.join(args.test_dataroot, args.valid_source)) if is_image_file(x)]

# generator
print('\nmaking generator ...')
ADM = AdaptiveDownsamplingModel(args)

if args.resume_down is None:
    raise NotImplementedError('put trained downsampling generator for testing')
else:
    ep0, total_it = ADM.resume(args.resume_down, train=False)
ep0 += 1
print('load generator successfully!')

saver = Saver(args, test=True)

print('\ntest start ...')
ADM.eval()
number = 0
with torch.no_grad():
    for file in tqdm(dataset):
        # load HR image
        fn = [os.path.basename(file)]
        img_s = Image.open(file)
        img_s = TF.to_tensor(img_s)
        img_s = img_s.unsqueeze(0)
        ADM.update_img(img_s)
        ADM.generate_LR()
        img_var = ADM.img_var
        per_var = ADM.per_var
        if img_var is not None:
            saver.write_img_var(1, (number + 1), ADM, args, fn)
        # if per_var is not None:
        #     saver.write_per_var_5(1, (number + 1), ADM, args, fn)
        saver.write_img_LR(1, (number+1), ADM, args, fn)
        number += 1
print('\ntest done!')