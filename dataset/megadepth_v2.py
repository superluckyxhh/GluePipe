import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import os.path as osp
import deepdish as dd
from collections import OrderedDict
from logzero import logger

from utils.geometry import get_inverse_transformation, reproject_keypoints



class MegaDepthCachedDataset(Dataset):
    def __init__(
        self, root, scene_list, 
        feature_root, overlap_scope, 
        limit_num_kpts, target_size, 
        random_crop = False,
        random_kpts = False,
        max_pairs_per_scene=None
    ):
        self.root = root
        self.scene_list = scene_list
        self.feature_root = feature_root
        self.target_size = target_size
        self.random_crop = random_crop
        self.limit_num_kpts = limit_num_kpts
        self.random_kpts = random_kpts
        
        self.image_pairs_dict, self.scene_pairsNum_dict = self.parse_filter_pairs(osp.join(root, 'pairs'), 
                                                        scene_list, overlap_scope, max_pairs_per_scene)
        all_length = sum(self.scene_pairsNum_dict.values())
        logger.info(f'{all_length} pairs of image are processed')
    
    
    @staticmethod
    def parse_filter_pairs(pairs_path, scene_list, overlap_scope, max_pairs_per_scene):
        min_overlap, max_overlap = overlap_scope
        meta_pairs_dict = {scene: osp.join(pairs_path, scene, 'sparse-txt', 'pairs.txt') for scene in scene_list}
        image_pairs = {}
        
        for scene, meta_pair_path in meta_pairs_dict.items():
            filter_pairs_list = []
            try:
                with open(meta_pair_path, 'r') as f:
                    raw = f.readlines()
                for line in raw:
                    line = line.strip('\n').split(' ')
                    overlap = float(line[-1])
                    if min_overlap <= overlap <= max_overlap:
                        filter_pairs_list.append(line)
                image_pairs[scene] = filter_pairs_list
            except FileNotFoundError:
                image_pairs[scene] = filter_pairs_list
                
        scene_pairs_num = {k: len(v) for k, v in image_pairs.items()}
                
        if max_pairs_per_scene is not None:
            scene_pairs_num = {k: min(v, max_pairs_per_scene) for k, v in scene_pairs_num.items()}
        
        return image_pairs, scene_pairs_num
    
    
    @staticmethod
    def parse_image_dict_line(pair_line):
        im0_name, im1_name = pair_line[0:2]
        camera_params = [float(x) for x in pair_line[4:]] 
        overlap = pair_line[-1]
        K0, K1, Transform = camera_params[:9], camera_params[9:18], camera_params[18:-1]
        K0 = np.array(K0).astype(np.float32).reshape(3, 3)
        K1 = np.array(K1).astype(np.float32).reshape(3, 3)
        Transform = np.array(Transform).astype(np.float32).reshape(4, 4)
        R, T = Transform[:3, :3], Transform[:3, -1]
        
        return im0_name, im1_name, K0, K1, R, T, float(overlap)
    
    
    def random_crop_process(self, target_size, im_size, lafs, scores, descriptors, K, depth):
        if target_size[0] < im_size[0]:
            if self.random_crop:
                start_width = np.random.randint(0, im_size[0]-target_size[0])
            else:
                start_width = (im_size[0] - target_size[0]) // 2
            end_width = start_width + target_size[0]
            
            depth = depth[:, start_width:end_width]
            kpts_crop_mask = (lafs[:, 0, 2] >= start_width) & (lafs[:, 0, 2] < end_width)
            K[0, 2] -= start_width
            lafs = lafs[kpts_crop_mask]
            lafs[:, 0, 2] -= start_width
            scores = scores[kpts_crop_mask]
            descriptors = descriptors[kpts_crop_mask]
                
        elif target_size[1] < im_size[1]:
            if self.random_crop:
                start_height = np.random.randint(0, im_size[1]-target_size[1])
            else:
                start_height = (im_size[1] - target_size[1]) // 2
            end_height = start_height + target_size[1]
            depth = depth[start_height:end_height, :]
            kpts_crop_mask = (lafs[:, 1, 2] >= start_height) & (lafs[:, 1, 2] < end_height)
            K[1, 2] -= start_height

            lafs = lafs[kpts_crop_mask]
            lafs[:, 1, 2] -= start_height
            scores = scores[kpts_crop_mask]
            descriptors = descriptors[kpts_crop_mask]
        
        return lafs, scores, descriptors, K, depth

    
    def limit_kpts(self, lafs, scores, descriptors, depth, im_id, limit_data, limit_transform):
        cur_num_kpts = lafs.size(0)
        if cur_num_kpts > self.limit_num_kpts:
            if self.random_kpts:
                kpts_select_ids = torch.randperm(cur_num_kpts)[:self.limit_num_kpts]
            else:
                kpts_select_ids = torch.topk(scores, self.limit_num_kpts, dim=0).indices
       
            limit_data[f'lafs{im_id}'] = lafs[kpts_select_ids]
            limit_data[f'scores{im_id}'] = scores[kpts_select_ids]
            limit_data[f'descriptors{im_id}'] = descriptors[kpts_select_ids]
            limit_transform[f'depth{im_id}'] = depth[
                limit_data[f'lafs{im_id}'][:, 1, 2].type(torch.int64),
                limit_data[f'lafs{im_id}'][:, 0, 2].type(torch.int64),
            ]
        else:
            limit_data[f'lafs{im_id}'][:cur_num_kpts] = lafs
            limit_data[f'scores{im_id}'][:cur_num_kpts] = scores
            limit_data[f'descriptors{im_id}'][:cur_num_kpts] = descriptors
            limit_transform[f'depth{im_id}'][:cur_num_kpts] = depth[
                lafs[:, 1, 2].type(torch.int64),
                lafs[:, 0, 2].type(torch.int64),
            ]
        
        return limit_data, limit_transform
    
    
    def compute_assignment(self, limit_data, positive_threshold, negative_threshold=None):
        """Given image pair, keypoints detected in each image, return set of ground truth correspondences"""
        # define module constants
        # index of keypoint that don't have a match
        UNMATCHED_INDEX = -1
        # index of keypoints to ignore during loss calculation  
        IGNORE_INDEX = -2 

        if negative_threshold is None:
            negative_threshold = positive_threshold
        
        lafs0, lafs1 = limit_data['lafs0'], limit_data['lafs1']
        kpts0, kpts1 = lafs0[:, :, -1], lafs1[:, :, -1]
       
        transformation = limit_data['transformation']
        transformation_inv = get_inverse_transformation(transformation)
        num0, num1 = kpts0.size(0), kpts1.size(0)

        # skip step if no keypoint are detected
        if num0 == 0 or num1 == 0:
            return None, None
        # establish ground truth correspondences given transformation
        kpts0_transformed, mask0 = reproject_keypoints(kpts0, transformation)
        kpts1_transformed, mask1 = reproject_keypoints(kpts1, transformation_inv)
        reprojection_error_0_to_1 = torch.cdist(kpts0_transformed, kpts1, p=2)  # batch_size x num0 x num1
        reprojection_error_1_to_0 = torch.cdist(kpts1_transformed, kpts0, p=2)  # batch_size x num1 x num0

        min_dist0, nn_matches0 = reprojection_error_0_to_1.min(-1)  # batch_size x num0
        min_dist1, nn_matches1 = reprojection_error_1_to_0.min(-1)  # batch_size x num1
        gt_matches0, gt_matches1 = nn_matches0.clone(), nn_matches1.clone()
        device = gt_matches0.device
        cross_check_consistent0 = torch.arange(num0, device=device) == gt_matches1.gather(0, gt_matches0)
        gt_matches0[~cross_check_consistent0] = UNMATCHED_INDEX

        cross_check_consistent1 = torch.arange(num1, device=device) == gt_matches0.gather(0, gt_matches1)
        gt_matches1[~cross_check_consistent1] = UNMATCHED_INDEX

        # so far mutual NN are marked MATCHED and non-mutual UNMATCHED
        symmetric_dist = 0.5 * (min_dist0[cross_check_consistent0] + min_dist1[cross_check_consistent1])

        gt_matches0[cross_check_consistent0][symmetric_dist > positive_threshold] = IGNORE_INDEX
        gt_matches0[cross_check_consistent0][symmetric_dist > negative_threshold] = UNMATCHED_INDEX

        gt_matches1[cross_check_consistent1][symmetric_dist > positive_threshold] = IGNORE_INDEX
        gt_matches1[cross_check_consistent1][symmetric_dist > negative_threshold] = UNMATCHED_INDEX

        gt_matches0[~cross_check_consistent0][min_dist0[~cross_check_consistent0] <= negative_threshold] = IGNORE_INDEX
        gt_matches1[~cross_check_consistent1][min_dist1[~cross_check_consistent1] <= negative_threshold] = IGNORE_INDEX

        # mutual NN with sym_dist <= pos.th ==> MATCHED
        # mutual NN with  pos.th < sym_dist <= neg.th ==> IGNORED
        # mutual NN with neg.th < sym_dist => UNMATCHED
        # non-mutual with dist <= neg.th ==> IGNORED
        # non-mutual with dist > neg.th ==> UNMATCHED

        # ignore kpts with unknown depth data
        gt_matches0[~mask0] = IGNORE_INDEX
        gt_matches1[~mask1] = IGNORE_INDEX

        # also ignore MATCHED point if its nearest neighbor is invalid
        gt_matches0[cross_check_consistent0][~mask1.gather(0, nn_matches0)[cross_check_consistent0]] = IGNORE_INDEX
        gt_matches1[cross_check_consistent1][~mask0.gather(0, nn_matches1)[cross_check_consistent1]] = IGNORE_INDEX

        assignment = {
           'gt_matches0': gt_matches0, 'gt_matches1': gt_matches1
        }
        
        return assignment
    
    
    def __len__(self):
        return sum(self.scene_pairsNum_dict.values())
    
    
    def __getitem__(self, idx):
        for s, pair_nums in self.scene_pairsNum_dict.items():
            if idx < pair_nums:
                scene, scene_idx = s, idx
                break
            else:
                idx -= pair_nums
        
        # logger.info(f'Scene:{scene}, idx:{scene_idx} is processing...')
        meta_data_line = self.image_pairs_dict[scene][scene_idx]
        im0_name, im1_name, K0, K1, R, T, overlap = self.parse_image_dict_line(meta_data_line)
        
        base0_name = im0_name[:-4]
        base1_name = im1_name[:-4]
        
        im0_path = osp.join(self.root, 'phoenix/S6/zl548/MegaDepth_v1', scene, 'dense0', 'imgs', im0_name)
        im1_path = osp.join(self.root, 'phoenix/S6/zl548/MegaDepth_v1', scene, 'dense0', 'imgs', im1_name)
        
        im0 = cv2.imread(im0_path)
        im1 = cv2.imread(im1_path)
        
        # [width, height]
        ori_im0_size = im0.shape[:2][::-1]
        ori_im1_size = im1.shape[:2][::-1]
        
        feature_path = osp.join(self.feature_root, scene)
        
        lafs0 = dd.io.load(osp.join(feature_path, base0_name+'_lafs.h5'))
        scores0 = dd.io.load(osp.join(feature_path, base0_name+'_scores.h5'))
        descriptors0 = dd.io.load(osp.join(feature_path, base0_name+'_descriptors.h5'))
        
        # [width, height]
        image_size0 = dd.io.load(osp.join(feature_path, base0_name+'_size.h5'))
        
        # origin size depth map
        depth0 = dd.io.load(osp.join(
            self.root, 'phoenix/S6/zl548/MegaDepth_v1', scene,
            'dense0', 'depths', base0_name+'.h5'))['depth']
        
        # Resize depth map to target size
        depth0 = cv2.resize(depth0, image_size0, interpolation=cv2.INTER_NEAREST)
        scale0 = np.diag([image_size0[0] / ori_im0_size[0], image_size0[1] / ori_im0_size[1], 1.0]).astype(np.float32)
        # im_size of K
        K0 = np.dot(scale0, K0)
        
        lafs1 = dd.io.load(osp.join(feature_path, base1_name+'_lafs.h5'))
        scores1 = dd.io.load(osp.join(feature_path, base1_name+'_scores.h5'))
        descriptors1 = dd.io.load(osp.join(feature_path, base1_name+'_descriptors.h5'))
        image_size1 = dd.io.load(osp.join(feature_path, base1_name+'_size.h5'))
        depth1 = dd.io.load(osp.join(
            self.root, 'phoenix/S6/zl548/MegaDepth_v1', scene,
            'dense0', 'depths', base1_name+'.h5'))['depth']
        depth1 = cv2.resize(depth1, image_size1, interpolation=cv2.INTER_NEAREST)
        scale1 = np.diag([image_size1[0] / ori_im1_size[0], image_size1[1] / ori_im1_size[1], 1.0]).astype(np.float32)
        K1 = np.dot(scale1, K1)
        
        # Random Crop / Center Crop
        # E.g. lafs shape: [num_kpts, 2, 3] 
        #      scores shape:[num_kpts,]
        #      descriptors shape: [num_kpts, desc_dim]
        lafs0, scores0, descriptors0, K0, depth0 = self.random_crop_process(self.target_size, image_size0, lafs0, scores0, descriptors0, K0, depth0)
        lafs1, scores1, descriptors1, K1, depth1 = self.random_crop_process(self.target_size, image_size1, lafs1, scores1, descriptors1, K1, depth1)
        
        # Numpy to Tensor
        lafs0 = torch.from_numpy(lafs0)
        lafs1 = torch.from_numpy(lafs1)
        
        scores0 = torch.from_numpy(scores0)
        scores1 = torch.from_numpy(scores1)
        
        descriptors0 = torch.from_numpy(descriptors0)
        descriptors1 = torch.from_numpy(descriptors1)
        
        depth0 = torch.from_numpy(depth0)
        depth1 = torch.from_numpy(depth1)
        
        # Limit number of keypoints
        desc_dim = descriptors0.size(1)
        limit_data = {
            'lafs0': torch.zeros(self.limit_num_kpts, 2, 3),
            'scores0': torch.zeros(self.limit_num_kpts),
            'descriptors0': torch.zeros(self.limit_num_kpts, desc_dim),
            'lafs1': torch.zeros(self.limit_num_kpts, 2, 3),
            'scores1': torch.zeros(self.limit_num_kpts),
            'descriptors1': torch.zeros(self.limit_num_kpts, desc_dim),
        }
        limit_transformation = {
            'type':['3d_reprojection'],
            'K0': torch.from_numpy(K0),
            'K1': torch.from_numpy(K1),
            'R': torch.from_numpy(R),
            'T': torch.from_numpy(T),
            'depth0': torch.zeros(self.limit_num_kpts),
            'depth1': torch.zeros(self.limit_num_kpts)
        }

        limit_data, limit_transformation = self.limit_kpts(
                    lafs0, scores0, descriptors0, 
                    depth0, 0, limit_data, limit_transformation
        )
        limit_data, limit_transformation = self.limit_kpts(
                    lafs1, scores1, descriptors1, 
                    depth1, 1, limit_data, limit_transformation
        )
        
        limit_data['transformation'] = limit_transformation
        
        # Make Ground-Truth Assignment
        gt_assignment = self.compute_assignment(
            limit_data, positive_threshold=2, negative_threshold=7)
        
        limit_data['ground_truth'] = gt_assignment
    
        return limit_data
        
        
    def collate_fn(self, batch):
        """ batch : List type """
        batch_size = len(batch)
        transformation = {
            'type': ['3d_reprojection'],
            'K0': torch.stack([x['transformation']['K0'] for x in batch]),
            'K1': torch.stack([x['transformation']['K1'] for x in batch]),
            'R': torch.stack([x['transformation']['R'] for x in batch]),
            'T': torch.stack([x['transformation']['T'] for x in batch]),
            'depth0': torch.stack([x['transformation']['depth0'] for x in batch]),
            'depth1': torch.stack([x['transformation']['depth1'] for x in batch]),
        }
        ground_truth = {
            'gt_matches0': torch.stack([x['ground_truth']['gt_matches0'] for x in batch]),
            'gt_matches1': torch.stack([x['ground_truth']['gt_matches1'] for x in batch]),
        }
        result = {
            'lafs0': torch.stack([x['lafs0'] for x in batch]),
            'lafs1': torch.stack([x['lafs1'] for x in batch]),
            'scores0': torch.stack([x['scores0'] for x in batch]),
            'scores1': torch.stack([x['scores1'] for x in batch]),
            'descriptors0': torch.stack([x['descriptors0'] for x in batch]),
            'descriptors1': torch.stack([x['descriptors1'] for x in batch]),
            'image0_size': self.target_size,
            'image1_size': self.target_size,
        }
        
        result['transformation'] = transformation
        result['ground_truth'] = ground_truth
        
        return result
        