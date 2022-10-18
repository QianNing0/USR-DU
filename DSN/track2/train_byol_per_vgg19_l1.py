import os
import yaml
import torch
import numpy as np
from tqdm import tqdm
from bicubic_pytorch import core

from track2.saver import Saver
from track2.options import Options
from track2.trainer_down_byol_per_vgg19_l1 import AdaptiveDownsamplingModel
from track2.dataset import unpaired_dataset
from track2.utility import log_writer, plot_loss_down, plot_psnr, timer, calc_psnr, quantize, plot_loss, plot_score

torch.manual_seed(0)

## parse options
parser = Options()
args = parser.parse()

## data loader
print('preparing dataset ...')
dataset = unpaired_dataset(args, phase='train')
train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.nThreads,
                                           drop_last=True)
print(len(dataset), len(train_loader))


dataset = unpaired_dataset(args, phase='test')
test_loader_down = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.nThreads)
print(len(dataset), len(test_loader_down))

ep0 = 0
total_it = 0

## Adaptive Downsampling Model
print('\nMaking Adpative-Downsampling-Model...')
ADM = AdaptiveDownsamplingModel(args)
if args.resume_down is not None:
    ep0, total_it = ADM.resume(args.resume_down)
    print('\nLoad downsampling generator from {}'.format(args.resume_down))

## saver and training log
saver = Saver(args)
data_timer, train_timer_down, eval_timer_down = timer(), timer(), timer()
training_log = log_writer(args.experiment_dir, args.name)

## losses
loss_dis = []  # discriminator loss
fake_score = []
real_score = []
loss_gen = []  # generator loss
loss_data = []  # data loss
loss_total = []  # total loss
if args.use_contrastive_loss:
    loss_con = []
if args.perceptual_loss:
    loss_per = []
psnrs = []
axis = []
print('\ntraining start')
for ep in range(ep0, args.epochs_down):

    training_log.write('\n[ epoch %03d/%03d ]   G lr %.06f D lr %.06f'
                       % (
                       ep + 1, args.epochs_down, ADM.gen_opt.param_groups[0]['lr'], ADM.dis_opt.param_groups[0]['lr']))
    print_txt = '|    Progress   |    Dis    |    Gen    |    data    |    fake    |    real    |'
    training_log.write('-' * len(print_txt))
    training_log.write(print_txt)

    loss_dis_item = 0
    fake_score_item = 0
    real_score_item = 0
    loss_gen_item = 0
    loss_data_item = 0
    loss_total_item = 0
    if args.use_contrastive_loss:
        loss_con_item = 0
    if args.perceptual_loss:
        loss_per_item = 0
    cnt = 0
    data_timer.tic()
    for it, (img_s, img_t, _, img_s_k) in enumerate(train_loader):
        if img_t.size(0) != args.batch_size:
            continue
        data_timer.hold()

        train_timer_down.tic()
        ADM.update_img(img_s, img_t, img_s_k)
        ADM.generate_LR()
        train_timer_down.hold()

        ## train downsampling network ADM
        train_timer_down.tic()
        ADM.update_D()
        ADM.update_G()
        train_timer_down.hold()

        loss_dis_item += ADM.loss_dis
        fake_score_item += ADM.fake_score
        real_score_item += ADM.real_score
        loss_gen_item += ADM.loss_gen
        loss_data_item += ADM.loss_data
        loss_total_item += ADM.loss_total
        if args.use_contrastive_loss:
            loss_con_item += ADM.loss_con
        if args.perceptual_loss:
            loss_per_item += ADM.loss_per
        cnt += 1

        ## print training log with save
        if (it + 1) % (len(train_loader) // 1) == 0:
            loss_dis_item_avg = loss_dis_item / cnt
            fake_score_item_avg = fake_score_item / cnt
            real_score_item_avg = real_score_item / cnt
            loss_gen_item_avg = loss_gen_item / cnt
            loss_data_item_avg = loss_data_item / cnt
            loss_total_item_avg = loss_total_item / cnt
            if args.use_contrastive_loss:
                loss_con_item_avg = loss_con_item / cnt
                loss_con_item = 0
            if args.perceptual_loss:
                loss_per_item_avg = loss_per_item / cnt
                loss_per_item = 0
            training_log.write('|   %04d/%04d   |  %.05f  |  %.05f  |  %.06f  |  %.05f  |  %.05f  |  %.01f+%.01fs  '
                               % ((it + 1), len(train_loader), loss_dis_item_avg, loss_gen_item_avg,
                                  loss_data_item_avg, fake_score_item_avg, real_score_item_avg,
                                  train_timer_down.release(), data_timer.release()))
            loss_dis_item = 0
            fake_score_item = 0
            real_score_item = 0
            loss_gen_item = 0
            loss_data_item = 0
            cnt = 0

            if args.save_results:
                saver.write_img_color_down(ep, ADM)

        data_timer.tic()
    ADM.dis_sch.step()
    ADM.gen_sch.step()

    training_log.write('-' * len(print_txt))

    loss_dis.append(loss_dis_item_avg)
    fake_score.append(fake_score_item_avg)
    real_score.append(real_score_item_avg)
    loss_gen.append(loss_gen_item_avg)
    loss_data.append(loss_data_item_avg)
    loss_total.append(loss_total_item_avg)

    if args.use_contrastive_loss:
        loss_con.append(loss_con_item_avg)
        plot_loss(os.path.join(args.experiment_dir, args.name), loss_con, 'con')
    if args.perceptual_loss:
        loss_per.append(loss_per_item_avg)
        plot_loss(os.path.join(args.experiment_dir, args.name), loss_per, 'per')
    plot_loss_down(os.path.join(args.experiment_dir, args.name), loss_dis, loss_gen, loss_data)
    plot_loss(os.path.join(args.experiment_dir, args.name), loss_total, 'total')
    plot_score(os.path.join(args.experiment_dir, args.name), fake_score, real_score)

    if (ep + 1) % args.save_snapshot == 0:
        saver.write_model_down(ep + 1, total_it + 1, ADM)
        eval_timer_down.tic()
        ADM.eval()
        psnr_sum = 0
        cnt = 0
        with torch.no_grad():
            for img_s, img_t, fn, img_s_k in tqdm(test_loader_down, ncols=80):
                ADM.update_img(img_s)
                ADM.generate_LR()
                img_var = ADM.img_var
                per_var = ADM.per_var
                if img_var is not None:
                    saver.write_img_var(ep + 1, cnt, ADM, args, fn)
                    # if per_var is not None and (ep + 1) % (10 * args.save_snapshot) == 0:
                    #     saver.write_per_var_5(ep + 1, cnt, ADM, args, fn)
                saver.write_img_LR(ep + 1, cnt, ADM, args, fn)
                cnt += 1
                img_fake = quantize(ADM.img_gen).detach().cpu()
                img_t = quantize(img_t)
                if args.test_visible:
                    psnr_sum += calc_psnr(
                        img_fake, img_t, args.scale
                    )
        eval_timer_down.hold()
        training_log.write('PSNR on test set: %.04f, %.01fs' % (psnr_sum / (cnt), eval_timer_down.release()))
        psnrs.append(psnr_sum / (cnt))
        axis.append(ep + 1)
        plot_psnr(os.path.join(args.experiment_dir, args.name), psnrs, axis)
        ADM.train()


    ## Save last generator and state
    training_log.write('Saving last generator and training state..')
    saver.write_model_down(-1, total_it + 1, ADM)
    print('\ndone!')

