######## Track1
#### kl_gau_bic  lr: 0.00005  ep500  milestones 250-1000
## train
CUDA_VISIBLE_DEVICES=0 python track1/train_byol_per_vgg19_l1.py --gen_model resnet_uncertainty_per_vgg19 --dis_model unet --dis_inc 3 --data_loss_type kl_gau_bic --gan_ratio 1 --data_ratio 100 --per_ratio 1 --perceptual_loss VGG19_Per --name mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1 --train_source corrupted_train_y --train_target corrupted_train_x --test_visible --valid_source corrupted_valid_y --valid_target corrupted_valid_x --train_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ --nobin --save_results --epochs_down 500 --lr_down 0.00005 --save_snapshot 10 --milestone 250-1000 --nThreads 6
## test
CUDA_VISIBLE_DEVICES=0 python track1/test_down_per_vgg19_l1.py --gen_model resnet_uncertainty_per_vgg19 --dis_model unet --dis_inc 3 --name mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1_ep420 --resume /home/tangjingzhu/Works/Real-SR/_Experiments/DSN_ADL/experiments/mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1/models/training_down_0420.pth --valid_source corrupted_train_y --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ --nobin
## valid
CUDA_VISIBLE_DEVICES=0 python track1/test_down_per_vgg19_l1.py --gen_model resnet_uncertainty_per_vgg19 --dis_model unet --dis_inc 3 --name valid_mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1_ep420 --resume /home/tangjingzhu/Works/Real-SR/_Experiments/DSN_ADL/experiments/mile250_track1_resnet_uncertainty_unet_kl_gau_bic_per_vgg19_l1/models/training_down_0420.pth --valid_source corrupted_valid_y --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ --nobin

######## Track 2
#### preprocess for kernel estimation
CUDA_VISIBLE_DEVICES=0 python3 ./preprocess/KernelGAN/train.py --X4 --input-dir /home/tangjingzhu/Works/Dataset/NITRE2020/Track2/DPEDiphone_train_x --output-dir /home/tangjingzhu/Works/Real-SR/_Experiments/DSN_ADL/kernel_results

#### kl_gau_ker  lr: 0.00005  ep200  milestones 50-100
## train
CUDA_VISIBLE_DEVICES=0 python track2/train_byol_per_vgg19_l1.py --gen_model resnet_uncertainty_per_vgg19 --dis_model unet --dis_inc 3 --data_loss_type kl_gau_ker --gan_ratio 1 --data_ratio 100 --per_ratio 1 --perceptual_loss VGG19_Per --name mile50_100_track2_resnet_uncertainty_unet_kl_gau_ker_per_vgg19_l1 --train_source DPEDiphone_train_y --train_target DPEDiphone_train_x --valid_source corrupted_valid_y --valid_target corrupted_valid_x --train_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track2/ --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track1/ --nobin --save_results --epochs_down 200 --lr_down 0.00005 --save_snapshot 5 --milestone 50-100 --nThreads 6 --batch_size 15
## test
CUDA_VISIBLE_DEVICES=0 python track2/test_down_per_vgg19_l1.py --gen_model resnet_uncertainty_per_vgg19 --dis_model unet --dis_inc 3 --name mile50_100_track2_resnet_uncertainty_unet_kl_gau_ker_per_vgg19_l1_ep175 --resume /home/tangjingzhu/Works/Real-SR/_Experiments/DSN_ADL/experiments/mile50_100_track2_resnet_uncertainty_unet_kl_gau_ker_per_vgg19_l1/models/training_down_0175.pth --valid_source DPEDiphone_train_y --test_dataroot /home/tangjingzhu/Works/Dataset/NITRE2020/Track2/ --nobin
