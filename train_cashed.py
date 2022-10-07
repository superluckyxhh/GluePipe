import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler
import numpy as np
import time
import os
import os.path as osp
import argparse
from omegaconf import OmegaConf
from models import get_matching_module
from extractors.superpoint.model import SuperPoint
from models.superglue.model import SuperGlue
from dataset.megadepth import MegaDepthCachedDataset
from loss_metric.loss import criterion
from loss_metric.metric import CameraPoseAUC, AccuracyUsingEpipolarDist, test_hpatches
from dataset.local_affline_transform import get_laf_to_sideinfo_converter
from utils.feature_util import prepare_features_output, data_to_device
import utils.distributed as disted
from tqdm import tqdm


def train_epoch(model, loader, optimizer, train_config, cur_epoch, max_epoch):
    cur_lr = optimizer.state_dict()['param_groups'][0]['lr']
    loader_bar = tqdm(loader)
    for i, batch in enumerate(loader_bar):    
        batch = data_to_device(batch, 'cuda')
        y_true = batch['ground_truth']
        
        y_pred = model(batch)
        
        loss_dict = criterion(y_true, y_pred, margin=train_config['margin'])
        loss = loss_dict['loss']
        metric_loss = loss_dict['metric_loss']
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loader_bar.desc = f"Train epoch [{cur_epoch+1}/{max_epoch}]  Learning rate:{cur_lr}  Loss:{loss:.4f}  Metric loss:{metric_loss:.4f}"
    
    return None


def test_epoch(model, loader, epipolar_dist_metric, camera_pose_auc_metric, cur_epoch, max_epoch):
    loader_bar = tqdm(loader)
    
    for i, batch in enumerate(loader_bar):
        batch = data_to_device(batch, 'cuda')
        transformation = {k: v[0] for k, v in batch['transformation'].items()}

        with torch.no_grad():
            pred = model.inference(batch)
            
        pred = {k: v[0] for k, v in pred.items()}
        kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
        matches, conf = pred['matches0'], pred['matching_scores0']

        matched_mask = matches > -1
        matched_kpts0 = kpts0[matched_mask]
        matched_kpts1 = kpts1[matches[matched_mask]]

        epipolar_dist_metric.get_p_ms(matched_kpts0, matched_kpts1, transformation, len(kpts0))
        camera_pose_auc_metric.get_pose_error(matched_kpts0, matched_kpts1, transformation)
    
    epipolar_dist_dict = epipolar_dist_metric.compute_precision_matchscore()
    precision, matching_score = np.array(epipolar_dist_dict['Precision']), np.array(epipolar_dist_dict['Matching Score'])
    pos_auc = camera_pose_auc_metric.compute_pose_auc()
    pos_auc_5 = np.array(pos_auc['AUC@5.0deg'])
    pos_auc_10 = np.array(pos_auc['AUC@10.0deg'])
    pos_auc_20 = np.array(pos_auc['AUC@20.0deg'])
    
    print(f"Test epoch:{cur_epoch+1} in MegaDepth, Mean Precision:{precision:.4f}, Mean Matching Score:{matching_score:.4f}, AUC@5:{pos_auc_5}, AUC@10:{pos_auc_10}, AUC@20:{pos_auc_20}")
    
    return {'Precision':precision, 'Matching Score':matching_score, 
            'Pos AUC@5':pos_auc_5, 'Pos AUC@10':pos_auc_10, 'Pos AUC@20':pos_auc_20}


def main():
    parser = argparse.ArgumentParser(description='Processing configuration for training')
    parser.add_argument('--config', type=str, help='path to config file', default='/home/user/code/GluePipeline/configs/cashed_config.yaml')
    parser.add_argument('--world_size', type=str, help='RANK', default=1)
    parser.add_argument('--dist_url', type=str, help='Dist Url', default='env://')
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    matching_config = config['matching']
    data_config = config['data']
    train_config = config['train']
    log_config = config['logging']
    
    # Setup Seed
    disted.init_distributed_mode(args)
    seed = train_config['seed'] + disted.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    print(f'Set seed to {seed}')
    
    # Load extractor for test hpatches
    extractor = SuperPoint({
        'nms_radius': 1,
        'keypoint_threshold': 0.01,
        'max_keypoints': 2048
    })
    
    # Load Model and Weights (IF EXISTED)
    model = get_matching_module(matching_config['name'])(matching_config).cuda()
    if train_config.get('sota_weight_path', None) is not None:
        weight_path = train_config['sota_weight_path']
        state_dicts = torch.load(weight_path)
        model.load_state_dict(state_dicts['model'])
        print(f'Loaded Trained Weight {weight_path} Done')
    
        
    # Load Dataset
    train_scenes = []
    val_scenes = []
    
    with open(data_config['train_list_path'], 'r') as f:
        scenes = f.readlines()
        for line in scenes:
            scene = line.strip('\n')
            train_scenes.append(scene)
            
    with open(data_config['test_list_path'], 'r') as f:
        scenes = f.readlines()
        for line in scenes:
            scene = line.strip('\n')
            val_scenes.append(scene)
    
    # Train Dataset
    train_dataset = MegaDepthCachedDataset(
        root=data_config['root_path'], 
        scene_list=train_scenes,
        feature_root=data_config['features_dir'],
        overlap_scope=data_config['train_pairs_overlap'], 
        limit_num_kpts=data_config['max_keypoints'],
        target_size=[960, 720], 
        random_crop=True, 
        random_kpts=False, 
        max_pairs_per_scene=None
    )
    
    train_sampler = RandomSampler(train_dataset, num_samples=train_config['steps_per_epoch'])
    batch_train_sampler = BatchSampler(
        train_sampler, data_config['batch_size_per_gpu'], drop_last=True
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_sampler=batch_train_sampler,
        collate_fn=train_dataset.collate_fn,
        num_workers=data_config['dataloader_workers_per_gpu'],
        pin_memory=True,
    )
    
    
    # Test Dataset
    test_dataset = MegaDepthCachedDataset(
        root=data_config['root_path'], 
        scene_list=val_scenes, 
        feature_root=data_config['features_dir'], 
        overlap_scope=data_config['train_pairs_overlap'],
        limit_num_kpts=data_config['max_keypoints'],
        target_size=[960, 720],
        random_crop=False,
        random_kpts=False,
        max_pairs_per_scene=data_config['val_max_pairs_per_scene'])

    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=data_config['batch_size_per_gpu'],
        shuffle=False,
        collate_fn=test_dataset.collate_fn,
        num_workers=data_config['dataloader_workers_per_gpu'],
        pin_memory=True, 
        drop_last=False)


    # Load Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=train_config['lr'])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=1,
        gamma=train_config['scheduler_gamma'])
    

    # Checkpoints Settings
    limit_numbers_per_epoch = train_config['steps_per_epoch']
    flag_pos = matching_config['geometric_position']
    flag_rot = matching_config['geometric_orientation']
    flag_residual = matching_config['residual']
    
    sota_name = (
        # f'test_' +
        f'{time.strftime("%m%d", time.localtime())}_' + 
        f'superglue_' +
        # f'imc2022_geometry_' +
        # f'geo_pos_{flag_pos}_' +
        # f'geo_rot_{flag_rot}_' +
        # f'residual_{flag_residual}_' +
        # f'cat_offset_cat_rotmatrix_'
        f'limit_{limit_numbers_per_epoch}'
    )

    artifact_path = os.path.join(train_config['weight_save_path'], sota_name)
    os.makedirs(artifact_path, exist_ok=True)
    
    # Evaluation Settings 
    if config.get('evaluation', True):
        eval_config = config['evaluation']
        epipolar_dist_metric = AccuracyUsingEpipolarDist(eval_config['epipolar_dist_threshold'])
        camera_pose_auc_metric = CameraPoseAUC(eval_config['camera_auc_thresholds'],
                                                eval_config['camera_auc_ransac_inliers_threshold'])
        print('Evaluation prepared !')
        
    
    for epoch in range(train_config['epochs']):
        # Train Epoch and Save Checkpoint
        train_epoch(model, train_loader, optimizer, train_config, epoch, train_config['epochs'])
        scheduler.step()
        
        if epoch % train_config['save_interval'] == 0 or epoch == train_config['epochs'] - 1:
            torch.save({'model':model.state_dict()}, f'{artifact_path}/model-epoch{epoch}.pth')
        
        # Test Epoch in MegaDepth and Save Evaluation
        eval_dict = test_epoch(model, test_loader, epipolar_dist_metric, camera_pose_auc_metric, epoch, train_config['epochs'])
        with open(os.path.join(log_config['pose_auc_root_path'], sota_name+'.txt'), 'a+') as f:
            _context = 'epoch:' + str(epoch) + ' ' + \
                        'precision:' + str(eval_dict['Precision']) + ' ' + \
                        'match score:' + str(eval_dict['Matching Score'])+ ' ' + \
                        'pos AUC @5 @10 @20:' + str(eval_dict['Pos AUC@5'])+ ' ' + \
                        str(eval_dict['Pos AUC@10']) + ' ' + \
                        str(eval_dict['Pos AUC@20']) + '\n'
            f.write(_context)
        
        # Test Epoch in Hpatches and Save Evaluation
        eval_hpatches_dict = test_hpatches(model, extractor, epoch, train_config['epochs'])
        with open(os.path.join(log_config['homography_path_root'], sota_name+'.txt'), 'a+') as g:
            _hp_context = 'epoch:' + str(epoch) + ' ' + \
                        'AUC@3:' + str(eval_hpatches_dict['auc@3']) + ' ' + \
                        'AUC@5:' + str(eval_hpatches_dict['auc@5'])+ ' ' + \
                        'AUC@10:' + str(eval_hpatches_dict['auc@10'])+ ' ' + \
                        'inliers:' + str(eval_hpatches_dict['inliers'])+ '\n'
            g.write(_hp_context)
                  
    print('Train Finish')
        

if __name__ == '__main__':
    main()