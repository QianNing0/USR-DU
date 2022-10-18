# USR-DU
(IJCAI-2022) Official PyTorch code for our  paper USR-DU: [Learning Degradation Uncertainty for Unsupervised Real-world Image Super-resolution](https://www.ijcai.org/proceedings/2022/0176.pdf). 

## Abstract
Acquiring degraded images with paired highresolution (HR) images is often challenging, impeding the advance of image super-resolution in real-world applications. By generating realistic low-resolution (LR) images with degradation similar to that in real-world scenarios, simulated paired LR-HR data can be constructed for supervised training. However, most of the existing work ignores the degradation uncertainty of the generated realistic LR images, since only one LR image has been generated given an HR image. To address this weakness, we propose learning the degradation uncertainty of generated LR images and sampling multiple LR images from the learned LR image (mean) and degradation uncertainty (variance) and construct LR-HR pairs to train the super-resolution (SR) networks. Specifically, uncertainty can be learned by minimizing the proposed loss based on Kullback-Leibler (KL) divergence. Furthermore, the uncertainty in the feature domain is exploited by a novel perceptual loss; and we propose to calculate the adversarial loss from the gradient information in the SR stage for stable training performance and better visual quality. Experimental results on popular real-world datasets show that our proposed method has performed better than other unsupervised approaches.

## Requirements

* Python 3.7
* PyTorch >= 1.7
* matplotlib
* imageio
* pyyaml
* scipy
* numpy
* tqdm
* PIL

## Hierarchy of codes
```
USR-DU
|
|----Checkpoint
|       |----track1
|               |----DSN
|               |----SRN
|       |----track2
|               |----DSN
|               |----SRN
|
|----DSN
|     |---bicubic_pytorch
|     |---discriminator
|     |---generator
|     |---kernel_results
|     |---KernelGAN
|     |---track1
|     |---track2
|     |---run.sh
|
|----SRN
|     |---code
|     |---realsr_options
|     |---scripts
|     |---inference.py
|     |---run.sh
```
The Checkpoint and kernel_results can be found in (https://pan.baidu.com/s/1yfkWK-Kv5rWaqIEtfuOpDg?pwd=s9xl code: s9xl)

## Data Preparation
As *HR* is not responsible to pixel aligned with *LR*, we recommend to use [DIV2K](https://cv.snu.ac.kr/research/EDSR/DIV2K.tar) dataset as *HR* dataset, which is consist of clean and high-resolution images.

For *LR* datasets, you should put a bunch of images that undergo a similar downsampling process. For example, NTIRE 2020 RWSR challenge offers two tracks for unsupervised SR training ([Track1](https://competitions.codalab.org/competitions/22220), [Track2](https://competitions.codalab.org/competitions/22221)), and some images are from same camera setting ([RealSR](https://github.com/csjcai/RealSR)).

The images are ***unpaired***.

## Pipeline
Pipeline can be presented as  `Training DSN  -->  Generating LRs  -->  Training SRN`.

Take ***Track1*** as an example below. Other bash scripts can be seen in *USR-DU/DSN/run.sh* and *USR-DU/SRN/run.sh*.

### DownSampling Network (DSN)
The DSN model can be trained with:
```
cd USR-DU/DSN
CUDA_VISIBLE_DEVICES=0 python track1/train_byol_per_vgg19_l1.py \
                              --gen_model resnet_uncertainty_per_vgg19 --dis_model unet \
                              --dis_inc 3 --data_loss_type kl_gau_bic \
                              --gan_ratio 1 --data_ratio 100 --per_ratio 1 --perceptual_loss VGG19_Per \
                              --name mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1 \
                              --train_source corrupted_train_y --train_target corrupted_train_x \
                              --test_visible --valid_source corrupted_valid_y --valid_target corrupted_valid_x \
                              --train_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ \
                              --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ \
                              --nobin --save_results --epochs_down 500 --lr_down 0.00005 \
                              --save_snapshot 10 --milestone 250-1000 --nThreads 6
```

### Generate LRs
The training data for SRN can be generated with:
```
cd USR-DU/DSN
CUDA_VISIBLE_DEVICES=0 python track1/test_down_per_vgg19_l1.py \
                              --gen_model resnet_uncertainty_per_vgg19 --dis_model unet \
                              --dis_inc 3 \ 
                              --name mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1_ep420 \
                              --resume Checkpoint\track1\DSN\training_down_0420.pth \
                              --valid_source corrupted_train_y 
                              --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ --nobin
```

### Super-Resolution Network (SRN)
We build our SRN codes based on [BasicSR](https://github.com/xinntao/BasicSR).

The SRN model can be trained with:
```
cd USR-DU/SRN
CUDA_VISIBLE_DEVICES=0 python code/train.py -opt realsr_options/ep420_track1_c_grad_lap_sam_vgg19_l1.yml
```
Please specify the configurations in `*.yml` file.

### Inference HRs
The HRs can be inferenced with:
```
cd USR-DU/SRN
CUDA_VISIBLE_DEVICES=0 python inference.py \
                              --modle_path Checkpoint\track1\SRN\net_g_18000.pth \
                              --input input_dir --output result_dir
```

### Citation
```
@inproceedings{ijcai2022p176,
  title     = {Learning Degradation Uncertainty for Unsupervised Real-world Image Super-resolution},
  author    = {Ning, Qian and Tang, Jingzhu and Wu, Fangfang and Dong, Weisheng and Li, Xin and Shi, Guangming},
  booktitle = {Proceedings of the Thirty-First International Joint Conference on
               Artificial Intelligence, {IJCAI-22}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Lud De Raedt},
  pages     = {1261--1267},
  year      = {2022},
  month     = {7},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2022/176},
  url       = {https://doi.org/10.24963/ijcai.2022/176},
}
```