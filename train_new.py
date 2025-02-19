import os
import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import torch
from torch import optim
from tqdm import tqdm

from loss.pose3d import loss_mpjpe, n_mpjpe, loss_velocity, loss_limb_var, loss_limb_gt, loss_angle, \
    loss_angle_velocity, bone_len_loss, body_part_orientive_loss, loss_limb_gt_hyperbone, focal_mpjpe
from loss.pose3d import jpe as calculate_jpe
from loss.pose3d import p_mpjpe as calculate_p_mpjpe
from loss.pose3d import mpjpe as calculate_mpjpe
from loss.pose3d import acc_error as calculate_acc_err
from data.const import H36M_JOINT_TO_LABEL, H36M_UPPER_BODY_JOINTS, H36M_LOWER_BODY_JOINTS, H36M_1_DF, H36M_2_DF, \
    H36M_3_DF
from data.reader.h36m import DataReaderH36M
from data.reader.motion_dataset import MotionDataset3D
from utils.data import flip_data
from utils.tools import set_random_seed, get_config, print_args, create_directory_if_not_exists
from torch.utils.data import DataLoader

# from utils.learning_ej import load_model, AverageMeter, decay_lr_exponentially
from utils.learning import load_model, AverageMeter, decay_lr_exponentially
from utils.tools import count_param_numbers
from utils.data import Augmenter2D
import glob

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    # parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=654262, type=int, help='random seed')
    # parser.add_argument("--config", type=str, default="configs/h36m/MotionAGFormer-base.yaml", help="Path to the config file.")
    # parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
    #                     help='checkpoint directory')
    parser.add_argument('--new-checkpoint', type=str, metavar='PATH', default='checkpoint',
                        help='new checkpoint directory')
    parser.add_argument('--checkpoint-file', type=str, help="checkpoint file name")
    # parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    parser.add_argument('--num-cpus', default=8, type=int, help='Number of CPU cores')
    # parser.add_argument('--use-wandb', action='store_true')
    # parser.add_argument('--wandb-name', default=None, type=str)
    # parser.add_argument('--wandb-run-id', default=None, type=str)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--refine', action='store_true')    
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)
    
def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    model_pos.eval()
           
    with torch.no_grad():
        for batch_input, batch_gt in tqdm(test_loader):
            N, T = batch_gt.shape[:2]
            if torch.cuda.is_available():
                batch_input = batch_input.cuda()
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if args.flip:    
                batch_input_flip = flip_data(batch_input)
                predicted_3d_pos_1 = model_pos(batch_input)
                predicted_3d_pos_flip = model_pos(batch_input_flip)
                predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip)                   # Flip back
                predicted_3d_pos = (predicted_3d_pos_1+predicted_3d_pos_2) / 2
            else:
                predicted_3d_pos = model_pos(batch_input)
            if args.root_rel:
                predicted_3d_pos[:,:,0,:] = 0     # [N,T,17,3]
            else:
                batch_gt[:,0,0,2] = 0

            if args.gt_2d:
                predicted_3d_pos[...,:2] = batch_input[...,:2]
            results_all.append(predicted_3d_pos.cpu().numpy())
    results_all = np.concatenate(results_all)
    results_all = datareader.denormalize(results_all)
    _, split_id_test = datareader.get_split_id()
    actions = np.array(datareader.dt_dataset['test']['action'])
    factors = np.array(datareader.dt_dataset['test']['2.5d_factor'])
    gts = np.array(datareader.dt_dataset['test']['joints_2.5d_image'])
    sources = np.array(datareader.dt_dataset['test']['source'])

    num_test_frames = len(actions)
    frames = np.array(range(num_test_frames))
    action_clips = actions[split_id_test]
    factor_clips = factors[split_id_test]
    source_clips = sources[split_id_test]
    frame_clips = frames[split_id_test]
    gt_clips = gts[split_id_test]
    assert len(results_all)==len(action_clips)
    
    e1_all = np.zeros(num_test_frames)
    e2_all = np.zeros(num_test_frames)
    oc = np.zeros(num_test_frames)
    acc_err_all = np.zeros(num_test_frames - 2)
    results = {}
    results_procrustes = {}
    results_accelaration = {}
    action_names = sorted(set(datareader.dt_dataset['test']['action']))
    for action in action_names:
        results[action] = []
        results_procrustes[action] = []
        results_accelaration[action] = []
    block_list = ['s_09_act_05_subact_02', 
                  's_09_act_10_subact_02', 
                  's_09_act_13_subact_01']
    for idx in range(len(action_clips)):
        source = source_clips[idx][0][:-6]
        if source in block_list:
            continue
        frame_list = frame_clips[idx]
        action = action_clips[idx][0]
        factor = factor_clips[idx][:,None,None]
        gt = gt_clips[idx]
        pred = results_all[idx]
        pred *= factor
        
        # Root-relative Errors
        pred = pred - pred[:,0:1,:]
        gt = gt - gt[:,0:1,:]
        err1 = calculate_mpjpe(pred, gt)
        err2 = calculate_p_mpjpe(pred, gt)
        acc_err = calculate_acc_err(pred, gt)
        acc_err_all[frame_list[:-2]] += acc_err

        e1_all[frame_list] += err1
        e2_all[frame_list] += err2
        oc[frame_list] += 1
    for idx in range(num_test_frames):
        if e1_all[idx] > 0:
            err1 = e1_all[idx] / oc[idx]
            err2 = e2_all[idx] / oc[idx]
            action = actions[idx]
            acc_err = acc_err_all[idx] / oc[idx]
            results[action].append(err1)
            results_procrustes[action].append(err2)
            results_accelaration[action].append(acc_err)
    final_result = []
    final_result_procrustes = []
    final_result_acceleration = []
    summary_table = prettytable.PrettyTable()
    summary_table.field_names = ['test_name'] + action_names
    for action in action_names:
        final_result.append(np.mean(results[action]))
        final_result_procrustes.append(np.mean(results_procrustes[action]))
        final_result_acceleration.append(np.mean(results_accelaration[action]))
    summary_table.add_row(['P1'] + final_result)
    summary_table.add_row(['P2'] + final_result_procrustes)
    print(summary_table)
    e1 = np.mean(np.array(final_result))
    e2 = np.mean(np.array(final_result_procrustes))
    acceleration_error = np.mean(np.array(final_result_acceleration))
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Acceleration error:', acceleration_error, 'mm/s^2')
    print('----------')
    return e1, e2, results_all, final_result, final_result_procrustes
        
def train_epoch(args, model_pos, train_loader, losses, optimizer, has_3d, has_gt):
    model_pos.train()

    # for idx, (batch_input, batch_gt) in tqdm(enumerate(train_loader)):
    for batch_input, batch_gt in tqdm(train_loader):   
        batch_size = len(batch_input)        
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            batch_gt = batch_gt.cuda()
        with torch.no_grad():
            if args.no_conf:
                batch_input = batch_input[:, :, :, :2]
            if not has_3d:
                conf = copy.deepcopy(batch_input[:,:,:,2:])    # For 2D data, weight/confidence is at the last channel
            if args.root_rel:
                batch_gt = batch_gt - batch_gt[:,:,0:1,:]
            else:
                batch_gt[:,:,:,2] = batch_gt[:,:,:,2] - batch_gt[:,0:1,0:1,2] # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        predicted_3d_pos = model_pos(batch_input)    # (N, T, 17, 3)

        optimizer.zero_grad()
        if has_3d:
            loss_3d_pos = loss_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_scale = n_mpjpe(predicted_3d_pos, batch_gt)
            loss_3d_velocity = loss_velocity(predicted_3d_pos, batch_gt)
            loss_lv = loss_limb_var(predicted_3d_pos)
            loss_lg = loss_limb_gt(predicted_3d_pos, batch_gt)
            loss_a = loss_angle(predicted_3d_pos, batch_gt)
            loss_av = loss_angle_velocity(predicted_3d_pos, batch_gt)
            loss_bonelen = bone_len_loss(predicted_3d_pos, batch_gt)
            loss_bodypart, loss_bodypart_angle = body_part_orientive_loss(predicted_3d_pos, batch_gt)
            
            loss_total = loss_3d_pos + \
                         args.lambda_scale       * loss_3d_scale + \
                         args.lambda_3d_velocity * loss_3d_velocity + \
                         args.lambda_lv          * loss_lv + \
                         args.lambda_lg          * loss_lg + \
                         args.lambda_a           * loss_a  + \
                         args.lambda_av          * loss_av
            losses['3d_pos'].update(loss_3d_pos.item(), batch_size)
            losses['3d_scale'].update(loss_3d_scale.item(), batch_size)
            losses['3d_velocity'].update(loss_3d_velocity.item(), batch_size)
            losses['lv'].update(loss_lv.item(), batch_size)
            losses['lg'].update(loss_lg.item(), batch_size)
            losses['angle'].update(loss_a.item(), batch_size)
            losses['angle_velocity'].update(loss_av.item(), batch_size)
            # losses['bone_len'].update(loss_bonelen.item(), batch_size)
            # losses['body_part_orientation'].update(loss_bodypart.item(), batch_size)
            # losses['body_part_angle'].update(loss_bodypart_angle.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        else:
            loss_2d_proj = loss_2d_weighted(predicted_3d_pos, batch_gt, conf)
            loss_total = loss_2d_proj
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)
            losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        optimizer.step()

def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))


    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': 6,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': 6,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    # train_dataset = MotionDataset3D(args, args.subset_list, 'train', mask=False, flip=False)
    # train_dataset2 = MotionDataset3D(args, args.subset_list, 'train', mask=False, flip=True)
    # train_dataset3 = MotionDataset3D(args, args.subset_list, 'train', mask=True, flip=False)
    # train_dataset4 = MotionDataset3D(args, args.subset_list, 'train', mask=True, flip=True)
    train_dataset = MotionDataset3D(args, args.subset_list, 'train')
    test_dataset = MotionDataset3D(args, args.subset_list, 'test')

    # train_dataset = ConcatDataset([train_dataset, train_dataset2, train_dataset3, train_dataset4])
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    

    datareader = DataReaderH36M(n_frames=args.n_frames, sample_stride=1, data_stride_train=args.n_frames//3, data_stride_test=args.n_frames, dt_root = 'data/motion3d', dt_file=args.dt_file)
    min_loss = 100000
    model_backbone = load_model(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    print('GPU: ', torch.cuda.is_available())
    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        # model_backbone = model_backbone.cuda()
    model_backbone = model_backbone.cuda()

    if args.refine == True:
        print('Implementing refinement')

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            # model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
            # model_backbone.load_state_dict(checkpoint['model'], strict=False) #motionagformer
            model_pos = model_backbone            
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=False) # take what you have
            # model_backbone.load_state_dict(checkpoint['model'], strict=True) #motionagformer
        model_pos = model_backbone
        
    # if args.partial_train:
    #     model_pos = partial_train_layers(model_pos, args.partial_train)


    if not opts.evaluate:        
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d), len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                # print(optimizer.state_dict()['param_groups'][0]['params'])
                # print(checkpoint['optimizer']['param_groups'][0]['params'])
                # for item in optimizer.state_dict()['param_groups'][0]['params']:
                #     if item in checkpoint['optimizer']['param_groups'][0]['params']:
                #         optimizer.load_state_dict(checkpoint['optimizer'])
                # optimizer.load_state_dict(checkpoint['optimizer'])
                pass
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
                
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)
        
        # Training
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            # losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            N = 0
                        
            # Curriculum Learning
            # if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
            #     train_epoch(args, model_pos, posetrack_loader_2d, losses, optimizer, has_3d=False, has_gt=True)
            #     train_epoch(args, model_pos, instav_loader_2d, losses, optimizer, has_3d=False, has_gt=False)
            train_epoch(args, model_pos, train_loader_3d, losses, optimizer, has_3d=True, has_gt=True) 
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
            else:
                e1, e2, results_all,_,_ = evaluate(args, model_pos, test_loader, datareader)
                print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                    losses['3d_pos'].avg,
                    e1, e2))
                train_writer.add_scalar('Error P1', e1, epoch + 1)
                train_writer.add_scalar('Error P2', e2, epoch + 1)
                train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
                # train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
                train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
                train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
                train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
                train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
                train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)
                
            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            if args.refine == True:
                chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch_refine.bin')
                chk_path_best = os.path.join(opts.checkpoint, 'best_epoch_refine.bin'.format(epoch))           
            else:
                chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
                chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))
            
            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            # if (epoch + 1) % args.checkpoint_frequency == 0:
            #     save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < 39.8:
                save_checkpoint(os.path.join(opts.checkpoint, f'latest_epoch_{e1:02}_{e2:02}.bin'), epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
                
    if opts.evaluate:
        ensemble = True
        if not ensemble:
            e1, e2, results_all, final_result_p1, final_result_p2 = evaluate(args, model_pos, test_loader, datareader)
        else:
            model_list = glob.glob(os.path.join(opts.checkpoint, "*.bin"))
            model_list.remove(os.path.join(opts.checkpoint, "latest_epoch.bin"))
            model_list.remove(os.path.join(opts.checkpoint, "best_epoch.bin"))
            print('We have these models', model_list)
            
            final_result_p1_sum=[0 for i in range(15)]
            final_result_p2_sum=[0 for i in range(15)]

            final_result_p1_min=[1000000 for i in range(15)]
            final_result_p2_min=[1000000 for i in range(15)]

            for model_path in model_list:
                checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
                print('Loading checkpoint', model_path)
                model_backbone.load_state_dict(checkpoint['model_pos'], strict=False)
                model_pos = model_backbone
            
                e1, e2, results_all, final_result_p1, final_result_p2 = evaluate(args, model_pos, test_loader, datareader)

                for i in range(len(final_result_p1)):
                    final_result_p1_sum[i] += final_result_p1[i]
                    final_result_p2_sum[i] += final_result_p2[i]

                    if final_result_p1[i] < final_result_p1_min[i]:
                        final_result_p1_min[i] = final_result_p1[i]
                    if final_result_p2[i] < final_result_p2_min[i]:
                        final_result_p2_min[i] = final_result_p2[i]
                print(final_result_p1_min)
                print(final_result_p2_min)
            final_result_p1_mean = [x / len(model_list) for x in final_result_p1_sum]
            final_result_p2_mean = [x / len(model_list) for x in final_result_p2_sum]

            e1_ensemble = sum(final_result_p1_mean)/15
            e2_ensemble = sum(final_result_p2_mean)/15

            e1_ensemble2 = sum(final_result_p1_min)/15
            e2_ensemble2 = sum(final_result_p2_min)/15
            print('Ensemble result: e1 %f e2 %f' % (e1_ensemble, e2_ensemble))
            print('Ensemble2 result: e1 %f e2 %f' % (e1_ensemble2, e2_ensemble2))

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)