import os
import tqdm

import torch
import numpy as np
import torch.nn as nn

from lib.helpers.save_helper import get_checkpoint_state
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.save_helper import save_checkpoint
from lib.losses.centernet_loss import compute_centernet3d_loss

from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections


from progress.bar import Bar
import time
import math
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import shutil

import copy
import cv2

def log10(x):
    return torch.log(x) / math.log(10)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count

class Trainer(object):
    def __init__(self,
                 cfg,
                 model,
                 optimizer,
                 train_loader,
                 lr_scheduler,
                 warmup_lr_scheduler,
                 logger):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        #self.test_loader = test_loader
        self.lr_scheduler = lr_scheduler
        self.warmup_lr_scheduler = warmup_lr_scheduler
        self.logger = logger
        self.epoch = 0
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.tensorboard_interval = 20
        # loading pretrain/resume model
        if cfg.get('pretrain_model'):
            assert os.path.exists(cfg['pretrain_model'])
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=cfg['pretrain_model'],
                            map_location=self.device,
                            logger=self.logger)

        if cfg.get('resume_model', None):
            assert os.path.exists(cfg['resume_model'])
            self.epoch = load_checkpoint(model=self.model.to(self.device),
                                         optimizer=self.optimizer,
                                         filename=cfg['resume_model'],
                                         map_location=self.device,
                                         logger=self.logger)
            self.lr_scheduler.last_epoch = self.epoch - 1

        self.gpu_ids = list(map(int, cfg['gpu_ids'].split(',')))
        self.model = torch.nn.DataParallel(model).cuda()

        self.writer = SummaryWriter()


    def train(self):
        start_epoch = self.epoch

        progress_bar = tqdm.tqdm(range(start_epoch, self.cfg['max_epoch']), dynamic_ncols=True, leave=True, desc='epochs')
        for epoch in range(start_epoch, self.cfg['max_epoch']):
            # reset random seed
            # ref: https://github.com/pytorch/pytorch/issues/5059
            np.random.seed(np.random.get_state()[1][0] + epoch)
            # train one epoch
            self.train_one_epoch()
            self.epoch += 1

            # update learning rate
            if self.warmup_lr_scheduler is not None and epoch < 5:
                self.warmup_lr_scheduler.step()
            else:
                self.lr_scheduler.step()


            # save trained model
            if (self.epoch % self.cfg['save_frequency']) == 0:
                os.makedirs(self.cfg['save_dir'], exist_ok=True)
                ckpt_name = os.path.join(self.cfg['save_dir'], 'checkpoint_epoch_%d' % self.epoch)
                save_checkpoint(get_checkpoint_state(self.model, self.optimizer, self.epoch), ckpt_name)

                #self.inference()

            progress_bar.update()

        return None


    def train_one_epoch(self):
        self.model.train()
        self.stats = {}  # reset stats dict
        self.stats['train'] = {}  # reset stats dict
        # loss_stats = ['depth_map']
        loss_stats = ['seg', 'offset2d', 'size2d', 'offset3d', 'depth', 'size3d', 'heading', 'center2kpt_offset', 'kpt_heatmap', 'kpt_heatmap_offset']
        data_time, batch_time = AverageMeter(), AverageMeter()
        avg_loss_stats = {l: AverageMeter() for l in loss_stats}
        num_iters = len(self.train_loader)
        bar = Bar('{}/{}'.format("3D", self.cfg['save_dir']), max=num_iters)
        end = time.time()
        for batch_idx, (inputs, targets, infos) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            for key in targets.keys():
                targets[key] = targets[key].to(self.device)

            # train one batch
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # total_loss, stats_batch = compute_depth_map_network_loss(outputs, targets, infos)
            total_loss, stats_batch = compute_centernet3d_loss(outputs, targets)
            total_loss.backward()
            self.optimizer.step()

            if batch_idx % self.tensorboard_interval == 0:
                for key in stats_batch:
                    self.writer.add_scalar(key, stats_batch[key], (self.epoch) * len(self.train_loader) + batch_idx)

            batch_time.update(time.time() - end)
            end = time.time()
            Bar.suffix = '{phase}: [{0}][{1}/{2}]|Tot: {total:} |ETA: {eta:} '.format(
                self.epoch, batch_idx, num_iters, phase="train",
                total=bar.elapsed_td, eta=bar.eta_td)

            for l in avg_loss_stats:
                avg_loss_stats[l].update(
                    stats_batch[l], inputs.shape[0])
                Bar.suffix = Bar.suffix + '|{} {:.4f} '.format(l, avg_loss_stats[l].avg)

            bar.next()
        bar.finish()



    # def inference(self):
    #     self.max_objs = self.test_loader.dataset.max_objs
    #
    #     self.model.eval()
    #
    #     ### save prediction depth map ###
    #     depth_map_pred = []
    #     depth_map_gt = {"rgb":[], "gt":[], 'fore_gt': []}
    #
    #     ### save rgb detection results ###
    #     rgb_results = {}
    #
    #     with torch.no_grad():
    #         progress_bar = tqdm.tqdm(total=len(self.test_loader), leave=True, desc='Evaluation Progress')
    #         for batch_idx, (inputs, target, info) in enumerate(self.test_loader):
    #             inputs = inputs.to(self.device)
    #
    #             rgb_outputs = self.model(inputs)
    #
    #             if 'depth_map' in rgb_outputs.keys():
    #                 depth_map_pred.append(rgb_outputs['depth_map'])
    #                 depth_map_gt['gt'].append(target['depth_gt'])
    #                 depth_map_gt['fore_gt'].append(target['fore_depth_gt'])
    #                 depth_map_gt['rgb'].append(inputs.cpu())
    #
    #             rgb_dets = self.process_dets2result(rgb_outputs, info)
    #             rgb_results.update(rgb_dets)
    #             progress_bar.update()
    #
    #     progress_bar.close()
    #
    #     # save the result for evaluation.
    #     self.logger.info('==> self.epoch:{} ...'.format(self.epoch))
    #     self.save_results(rgb_results, './'+self.cfg['save_dir']+'/rgb_outputs')
    #
    #     self.evaluate_detection()
    #     self.evaluate_depth(depth_map_pred, depth_map_gt)

    # def process_dets2result(self, outputs, info):
    #     dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs)
    #     dets = dets.detach().cpu().numpy()
    #
    #     # get corresponding calibs & transform tensor to numpy
    #     calibs = [self.test_loader.dataset.get_calib(index) for index in info['img_id']]
    #     info = {key: val.detach().cpu().numpy() for key, val in info.items()}
    #     cls_mean_size = self.test_loader.dataset.cls_mean_size
    #     dets = decode_detections(dets=dets,
    #                              info=info,
    #                              calibs=calibs,
    #                              cls_mean_size=cls_mean_size,
    #                              threshold=self.cfg.get('threshold', 0.2))
    #
    #     return dets

    # def save_results(self, results, output_dir):
    #     output_dir = os.path.join(output_dir, 'data')
    #     if os.path.exists(output_dir):
    #         shutil.rmtree(output_dir, True)
    #     os.makedirs(output_dir, exist_ok=False)
    #
    #     for img_id in results.keys():
    #         output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
    #
    #         f = open(output_path, 'w')
    #         for i in range(len(results[img_id])):
    #             class_name = self.class_name[int(results[img_id][i][0])]
    #             f.write('{} 0.0 0'.format(class_name))
    #             for j in range(1, len(results[img_id][i])):
    #                 f.write(' {:.2f}'.format(results[img_id][i][j]))
    #             f.write('\n')
    #         f.close()
    #
    # def evaluate_detection(self):
    #     self.test_loader.dataset.eval(results_dir='./'+self.cfg['save_dir']+'/rgb_outputs/data', logger=self.logger)
    #
    # def evaluate_depth(self, results, gt):
    #
    #     all_metric, fore_metric = self.eval_depth_map(results, gt)
    #
    #     #self.logger.info('==> self.epoch:{}'.format(self.epoch))
    #     #self.logger.info('==> epe:{}'.format(epe))
    #     self.logger.info('==> all metric')
    #     self.logger.info('==> rmse:{}'.format(all_metric['rmse_mean']))
    #     self.logger.info('==> rmse_log:{}'.format(all_metric['rmse_log_mean']))
    #     self.logger.info('==> silog:{}'.format(all_metric['silog_mean']))
    #     self.logger.info('==> absrel:{}'.format(all_metric['absrel_mean']))
    #     self.logger.info('==> sqlres:{}'.format(all_metric['sqlres_mean']))
    #
    #     self.logger.info('==> fore metric')
    #     self.logger.info('==> rmse:{}'.format(fore_metric['fore_rmse_mean']))
    #     self.logger.info('==> rmse_log:{}'.format(fore_metric['fore_rmse_log_mean']))
    #     self.logger.info('==> silog:{}'.format(fore_metric['fore_silog_mean']))
    #     self.logger.info('==> absrel:{}'.format(fore_metric['fore_absrel_mean']))
    #     self.logger.info('==> sqlres:{}'.format(fore_metric['fore_sqlres_mean']))
    #
    #
    # def eval_depth_map(self, result, gt):
    #     SAVE_PATH = './' + self.cfg['save_dir'] + '/'
    #
    #     if os.path.exists(SAVE_PATH + 'val_pred/'):
    #         pass
    #     else:
    #         os.mkdir(SAVE_PATH + 'val_pred/')
    #
    #     index = 0
    #     epe_sum = 0
    #     rmse_sum = 0
    #     rmse_log_sum = 0
    #     silog_sum = 0
    #     absres_sum = 0
    #     sqlres_sum = 0
    #     delta1_sum = 0
    #     delta2_sum = 0
    #     delta3_sum = 0
    #
    #     fore_epe_sum = 0
    #     fore_rmse_sum = 0
    #     fore_rmse_log_sum = 0
    #     fore_silog_sum = 0
    #     fore_absres_sum = 0
    #     fore_sqlres_sum = 0
    #     fore_delta1_sum = 0
    #     fore_delta2_sum = 0
    #     fore_delta3_sum = 0
    #
    #     for i, pred_item in enumerate(result):
    #         gt_item = gt['gt'][i]
    #         fore_gt_item = gt['fore_gt'][i]
    #         rgb_item = gt['rgb'][i]
    #         image_output = {}
    #
    #         pred_depth_map = pred_item[:, 0, :, :]
    #         #pred_depth_uncertainty = pred_item[:, 1, :, :]
    #
    #         valid_ind = (gt_item > 0)
    #         fore_valid_ind = (fore_gt_item > 0)
    #
    #         ### first: decode to depth   then: upsample
    #         pred_depth_map_decode = pred_depth_map.unsqueeze(dim=1)
    #         pred_depth_map_decode = F.interpolate(pred_depth_map_decode, size=[384, 1280], mode="bilinear")
    #         pred_depth_map_decode = 1. / (pred_depth_map_decode.sigmoid() + 1e-6) - 1.
    #
    #         ### visual prediction and error map ###
    #         if i%40==0:
    #             if os.path.exists(SAVE_PATH + 'val_pred/' + '{}/'.format(str(i))):
    #                 pass
    #             else:
    #                 os.mkdir(SAVE_PATH + 'val_pred/' + '{}/'.format(str(i)))
    #
    #             pred_depth_map_decode_visual = copy.deepcopy(pred_depth_map_decode)
    #             inputs = rgb_item.cpu().numpy()  # (2, 3, 384, 1280)
    #             target = gt_item.cpu().numpy()  # (2, 384, 1280)
    #             #pred = pred_depth_map_decode.detach().cpu()  # (2, 2, 96, 320)
    #             error_map = disp_error_image(pred_depth_map_decode_visual.cpu().numpy()[0, 0, :, :], target[0, :, :])  # [2, 3, 384, 1280]
    #             pred_depth_map_decode_visual = pred_depth_map_decode_visual.cpu().numpy()  # [2, 1, 384, 1280]
    #
    #             target = depth_render_kitti(target, 80)  # [h, w, 3]
    #             pred = depth_render_kitti(pred_depth_map_decode_visual, 80)  # [h, w, 3]
    #
    #             concat_pred_target = np.hstack([pred, target])
    #             concat_rgb_errormap = np.hstack([inputs[0].transpose(1, 2, 0) * 255, error_map])
    #             concat_all = np.vstack([concat_pred_target, concat_rgb_errormap])
    #             cv2.imwrite(SAVE_PATH + 'val_pred/' + '{}/{}_{}.png'.format(str(i), str(i), self.epoch), concat_all)
    #
    #         pred_depth_map_decode = pred_depth_map_decode.squeeze(dim=1)
    #
    #         pred_depth_map_decode_valid = pred_depth_map_decode[valid_ind].cuda()
    #         gt_valid = gt_item[valid_ind].cuda()
    #         fore_pred_depth_map_decode_valid = pred_depth_map_decode[fore_valid_ind].cuda()
    #         fore_gt_valid = fore_gt_item[fore_valid_ind].cuda()
    #
    #         abs_diff = (pred_depth_map_decode_valid - gt_valid).abs()
    #         fore_abs_diff = (fore_pred_depth_map_decode_valid - fore_gt_valid).abs()
    #
    #         epe = float(abs_diff.mean())
    #         fore_epe = float(fore_abs_diff.mean())
    #         ### RMSE  均方根误差
    #         mse = float((torch.pow(abs_diff, 2)).mean())
    #         rmse = math.sqrt(mse)
    #
    #         fore_mse = float((torch.pow(fore_abs_diff, 2)).mean())
    #         fore_rmse = math.sqrt(fore_mse)
    #
    #         ### RMSE log  对数均方根误差
    #         rmse_log = torch.pow(log10(pred_depth_map_decode_valid) - log10(gt_valid), 2)
    #         rmse_log = math.sqrt(rmse_log.mean())
    #
    #         fore_rmse_log = torch.pow(log10(fore_pred_depth_map_decode_valid) - log10(fore_gt_valid), 2)
    #         fore_rmse_log = math.sqrt(fore_rmse_log.mean())
    #
    #         ### SILog
    #         err = log10(pred_depth_map_decode_valid) - log10(gt_valid)
    #         silog = math.sqrt((err ** 2).mean() - (err.mean() ** 2)) * 100
    #
    #         fore_err = log10(fore_pred_depth_map_decode_valid) - log10(fore_gt_valid)
    #         fore_silog = math.sqrt((fore_err ** 2).mean() - (fore_err.mean() ** 2)) * 100
    #
    #         ### Abs Rel  绝对相对误差
    #         absrel = float((abs_diff / gt_valid).mean())
    #
    #         fore_absrel = float((fore_abs_diff / fore_gt_valid).mean())
    #
    #         ### Sql Rel  平方相对误差
    #         sqlres = float((abs_diff ** 2 / gt_valid).mean())
    #
    #         fore_sqlres = float((fore_abs_diff ** 2 / fore_gt_valid).mean())
    #
    #         ### accuracy  精确度
    #         maxRatio = torch.max(pred_depth_map_decode_valid / gt_valid, gt_valid / pred_depth_map_decode_valid)
    #         delta1 = float((maxRatio < 1.25).float().mean())
    #         delta2 = float((maxRatio < 1.25 ** 2).float().mean())
    #         delta3 = float((maxRatio < 1.25 ** 3).float().mean())
    #
    #         fore_maxRatio = torch.max(fore_pred_depth_map_decode_valid / fore_gt_valid,
    #                                   fore_gt_valid / fore_pred_depth_map_decode_valid)
    #         fore_delta1 = float((fore_maxRatio < 1.25).float().mean())
    #         fore_delta2 = float((fore_maxRatio < 1.25 ** 2).float().mean())
    #         fore_delta3 = float((fore_maxRatio < 1.25 ** 3).float().mean())
    #
    #         epe_sum += epe
    #         rmse_sum += rmse
    #         rmse_log_sum += rmse_log
    #         silog_sum += silog
    #         absres_sum += absrel
    #         sqlres_sum += sqlres
    #         delta1_sum += delta1
    #         delta2_sum += delta2
    #         delta3_sum += delta3
    #
    #         fore_epe_sum += fore_epe
    #         fore_rmse_sum += fore_rmse
    #         fore_rmse_log_sum += fore_rmse_log
    #         fore_silog_sum += fore_silog
    #         fore_absres_sum += fore_absrel
    #         fore_sqlres_sum += fore_sqlres
    #         fore_delta1_sum += fore_delta1
    #         fore_delta2_sum += fore_delta2
    #         fore_delta3_sum += fore_delta3
    #
    #         index += 1
    #
    #     all_metric = {}
    #     fore_metric = {}
    #
    #     all_metric['epe_mean'] = float(epe_sum / index)
    #     all_metric['rmse_mean'] = float(rmse_sum / index)
    #     all_metric['rmse_log_mean'] = float(rmse_log_sum / index)
    #     all_metric['silog_mean'] = float(silog_sum / index)
    #     all_metric['absrel_mean'] = float(absres_sum / index)
    #     all_metric['sqlres_mean'] = float(sqlres_sum / index)
    #     all_metric['delta1_mean'] = float(delta1_sum / index)
    #     all_metric['delta2_mean'] = float(delta2_sum / index)
    #     all_metric['delta3_mean'] = float(delta3_sum / index)
    #
    #     fore_metric['fore_epe_mean'] = float(fore_epe_sum / index)
    #     fore_metric['fore_rmse_mean'] = float(fore_rmse_sum / index)
    #     fore_metric['fore_rmse_log_mean'] = float(fore_rmse_log_sum / index)
    #     fore_metric['fore_silog_mean'] = float(fore_silog_sum / index)
    #     fore_metric['fore_absrel_mean'] = float(fore_absres_sum / index)
    #     fore_metric['fore_sqlres_mean'] = float(fore_sqlres_sum / index)
    #     fore_metric['fore_delta1_mean'] = float(fore_delta1_sum / index)
    #     fore_metric['fore_delta2_mean'] = float(fore_delta2_sum / index)
    #     fore_metric['fore_delta3_mean'] = float(fore_delta3_sum / index)
    #
    #     return all_metric, fore_metric


