import os
import argparse
import cv2
import random
import numpy as np
import torch
import csv
from glob import glob
from collections import namedtuple
from  tqdm import tqdm

from models.imc2022.model import IMCNet
from extractors.superpoint.model import SuperPoint
from models.superglue.model import SuperGlue
from utils.metric_utils import (
    compute_pose_error, compute_epipolar_error,
    estimate_pose, pose_auc, read_image,
)

torch.set_grad_enabled(False)

Gt = namedtuple('Gt', ['K', 'R', 'T'])
eps = 1e-15


def load_covisibilty_data(file_path):
    covisibility_dict = {}
    with open(file_path, 'r') as f:
        raw = f.readlines()[1:]
    for line in raw:
        ths = line.strip('\n').split(',')
        
        name0_1 = ths[0]
        covisibility = float(ths[1])
        F_matrix = np.array([float(v) for v in ths[2].split(' ')]).reshape(3, 3)
        covisibility_dict[name0_1] = covisibility
        # covis_dict['fundamental_matrix'] = F_matraix
    
    return covisibility_dict
        
           
def get_images_features(file_root, im_id, target_size, extractor):
    im_path = os.path.join(file_root, im_id+'.jpg')
    image = cv2.cvtColor(cv2.imread(im_path), cv2.COLOR_BGR2GRAY)
    
    w, h = image.shape[:2][::-1]
    w_new, h_new = target_size[0], target_size[1]
    scales = (float(w) / float(w_new), float(h) / float(h_new))
    
    image = cv2.resize(image.astype('float32'), (w_new, h_new))
    t_image = torch.from_numpy(image / 255.).float()[None, None]
    

    features = extractor({'image':t_image})
    keypoints = features['keypoints'][0].unsqueeze(0)
    descriptors = features['descriptors'][0].unsqueeze(0)
    scores = features['scores'][0].unsqueeze(0)
    
    return t_image, keypoints, descriptors, scores
    
    
def load_calibration(file_path):
    calib_dicts = {}
    with open(file_path, 'r') as f:
        raw = f.readlines()[1:]
    for line in raw:
        ths = line.strip('\n').split(',')
        assert len(ths) == 4, 'Invalid length of calibrate values'
        
        image_id = ths[0]
        K = np.array([float(v) for v in ths[1].split(' ')]).reshape(3, 3)
        R = np.array([float(v) for v in ths[2].split(' ')]).reshape(3, 3)
        T = np.array([float(v) for v in ths[3].split(' ')])
        calib_dicts[image_id] = Gt(K=K, R=R, T=T)
        
    return calib_dicts


def load_scale_data(file_path):
    scale_dicts = {}
    with open(file_path, 'r') as f:
        raw = f.readlines()[1:]
    for line in raw:
        ths = line.strip('\n').split(',')
        scene_id = ths[0]
        scale_value = float(ths[1])
        scale_dicts[scene_id] = scale_value
    
    return scale_dicts
    


def normalize_keypoints(keypoints, K):
    C_x = K[0, 2]
    C_y = K[1, 2]
    f_x = K[0, 0]
    f_y = K[1, 1]
    keypoints = (keypoints - np.array([[C_x, C_y]])) / np.array([[f_x, f_y]])
    return keypoints


def compute_essential_matrix(F, K1, K2, kp1, kp2):
    '''Compute the Essential matrix from the Fundamental matrix, given the calibration matrices. Note that we ask participants to estimate F, i.e., without relying on known intrinsics.'''
    
    # Warning! Old versions of OpenCV's RANSAC could return multiple F matrices, encoded as a single matrix size 6x3 or 9x3, rather than 3x3.
    # We do not account for this here, as the modern RANSACs do not do this:
    # https://opencv.org/evaluating-opencvs-new-ransacs
    assert F.shape[0] == 3, 'Malformed F?'

    # Use OpenCV's recoverPose to solve the cheirality check:
    # https://docs.opencv.org/4.5.4/d9/d0c/group__calib3d.html#gadb7d2dfcc184c1d2f496d8639f4371c0
    E = np.matmul(np.matmul(K2.T, F), K1).astype(np.float64)
    
    kp1n = normalize_keypoints(kp1, K1)
    kp2n = normalize_keypoints(kp2, K2)
    num_inliers, R, T, mask = cv2.recoverPose(E, kp1n, kp2n)

    return E, R, T


def quaternion_from_matrix(matrix):
    '''Transform a rotation matrix into a quaternion.'''

    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    m00 = M[0, 0]
    m01 = M[0, 1]
    m02 = M[0, 2]
    m10 = M[1, 0]
    m11 = M[1, 1]
    m12 = M[1, 2]
    m20 = M[2, 0]
    m21 = M[2, 1]
    m22 = M[2, 2]

    K = np.array([[m00 - m11 - m22, 0.0, 0.0, 0.0],
              [m01 + m10, m11 - m00 - m22, 0.0, 0.0],
              [m02 + m20, m12 + m21, m22 - m00 - m11, 0.0],
              [m21 - m12, m02 - m20, m10 - m01, m00 + m11 + m22]])
    K /= 3.0

    # The quaternion is the eigenvector of K that corresponds to the largest eigenvalue.
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]

    if q[0] < 0:
        np.negative(q, q)

    return q


def compute_error_for_one_example(q_gt, T_gt, q, T, scale):
    '''Compute the error metric for a single example.
    
    The function returns two errors, over rotation and translation. 
    These are combined at different thresholds by ComputeMaa in order to compute the mean Average Accuracy.
    '''
    
    q_gt_norm = q_gt / (np.linalg.norm(q_gt) + eps)
    q_norm = q / (np.linalg.norm(q) + eps)

    loss_q = np.maximum(eps, (1.0 - np.sum(q_norm * q_gt_norm)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # Apply the scaling factor for this scene.
    T_gt_scaled = T_gt * scale
    T_scaled = T * np.linalg.norm(T_gt) * scale / (np.linalg.norm(T) + eps)

    err_t = min(np.linalg.norm(T_gt_scaled - T_scaled), np.linalg.norm(T_gt_scaled + T_scaled))

    return err_q * 180 / np.pi, err_t


def compute_MAA(err_q, err_t, thresholds_q, thresholds_t):
    '''Compute the mean Average Accuracy at different tresholds, for one scene.'''
    
    assert len(err_q) == len(err_t)
    
    acc, acc_q, acc_t = [], [], []
    for th_q, th_t in zip(thresholds_q, thresholds_t):
        acc += [(np.bitwise_and(np.array(err_q) < th_q, np.array(err_t) < th_t)).sum() / len(err_q)]
        acc_q += [(np.array(err_q) < th_q).sum() / len(err_q)]
        acc_t += [(np.array(err_t) < th_t).sum() / len(err_t)]
    return np.mean(acc), np.array(acc), np.array(acc_q), np.array(acc_t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Image pair matching and pose evaluation with IMCNet',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--scene_root', type=str, default='/home/user/dataset/imc2022/train', 
        help='photo to tourium root contains 9 scenes')
    parser.add_argument(
        '--output_path', type=str, default='/home/user/code/GluePipeline/logs/mma')
    parser.add_argument(
        '--target_size', type=int, nargs='+', default=[960, 720],
        help='Resize the input image before running inference. If two numbers, '
             'resize to the exact dimensions, if one number, resize the max '
             'dimension, if -1, do not resize')
    parser.add_argument(
        '--imc_weights', default='/home/user/code/GluePipeline/sota_weights/imc2022_adaptive_limit_100000/model-epoch11.pth',
        help='IMCNet (Megadepth) weights')
    parser.add_argument(
        '--max_keypoints', type=int, default=1024,
        help='Maximum number of keypoints detected by Superpoint'
             ' (\'-1\' keeps all keypoints)')
    parser.add_argument(
        '--keypoint_threshold', type=float, default=0.005,
        help='SuperPoint keypoint detector confidence threshold')
    parser.add_argument(
        '--nms_radius', type=int, default=4,
        help='SuperPoint Non Maximum Suppression (NMS) radius'
        ' (Must be positive)')
    parser.add_argument(
        '--sinkhorn_iterations', type=int, default=20,
        help='Number of Sinkhorn iterations performed by SuperGlue')
    parser.add_argument(
        '--match_threshold', type=float, default=0.2,
        help='IMCNet match threshold')

    args = parser.parse_args()
    
    if len(args.target_size) == 2:
        print('Will resize to {}x{} (WxH)'.format(
            args.target_size[0], args.target_size[1]))
    
    scale_file_path = os.path.join(args.scene_root, 'scaling_factors.csv')
    scale_dicts = load_scale_data(scale_file_path)
    
    show_images = True
    num_show_images = 1
    max_pairs_per_scene = 50
    verbose = True
    
    thresholds_q = np.linspace(1, 10, 10)
    thresholds_t = np.geomspace(0.2, 5, 10)

    errors = {scene: {} for scene in scale_dicts.keys()}
    mAA = {scene: {} for scene in scale_dicts.keys()}
    
    all_scenes = [os.path.join(args.scene_root, scene) for scene in os.listdir(args.scene_root)]
    all_scenes = [scene for scene in all_scenes if os.path.isdir(scene)]
    print(f'Total {len(all_scenes)} scenes')
    
    for _scene in all_scenes:
        name = _scene.split('/')[-1]
        # Load all pairs, find those with a co-visibility over 0.1, and subsample them.
        pair_covisibility_path = os.path.join(_scene, 'pair_covisibility.csv')
        covisibility_dict = load_covisibilty_data(pair_covisibility_path)
        pairs = [pair for pair, covis in covisibility_dict.items() if covis >= 0.1]
        print(f'-- Processing scene "{name}": found {len(pairs)} pairs (will keep {min(len(pairs), max_pairs_per_scene)})', flush=True)
        
        random.shuffle(pairs)
        pairs = pairs[:max_pairs_per_scene]
        
        # Extract the images in these pairs (we don't need to load images we will not use).
        ids = []
        for pair in pairs:
            cur_ids = pair.split('-')
            assert cur_ids[0] > cur_ids[1]
            ids += cur_ids
        ids = list(set(ids))
        
        # Load ground truth data.
        calibration_path = os.path.join(_scene, 'calibration.csv')
        calibration_dict = load_calibration(calibration_path)
        
        # Load images and extract SuperPoint features.
        extractor = SuperPoint({'max_keypoints':args.max_keypoints})
        im_root = os.path.join(_scene, 'images')
        images_dict = {}
        kpts_dict = {}
        desc_dict = {}
        score_dict = {}
        print('Extracting features...')
        for id in tqdm(ids):
            tim, keypoints, descriptors, scores = get_images_features(im_root, id, args.target_size, extractor)
            images_dict[id] = tim
            kpts_dict[id] = keypoints
            desc_dict[id] = descriptors
            score_dict[id] = scores
        print(f'Extracted features for {len(kpts_dict)} images (avg: {np.mean([len(v) for v in desc_dict.values()])})')
        
        # Process the pairs.
        max_err_acc_q_new = []
        max_err_acc_t_new = []
        
        for counter, pair in enumerate(pairs):
            id0, id1 = pair.split('-')
            
            # For SuperGlue
            pair_matching_data = {
                'keypoints0':kpts_dict[id0],
                'keypoints1':kpts_dict[id1],
                'scores0':score_dict[id0],
                'scores1':score_dict[id1],
                'descriptors0':desc_dict[id0],
                'descriptors1':desc_dict[id1],
                'image0_size':args.target_size[::-1],
                'image1_size':args.target_size[::-1],
            }

            matching_model = SuperGlue({})
            pred = matching_model(pair_matching_data)
            
            pred = {k: v[0] for k, v in pred.items()}
            kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
            kpts0_array, kpts1_array = np.array(kpts0), np.array(kpts1)
            matches, conf = pred['matches0'], pred['matching_scores0']

            matched_mask = matches > -1
            matched_kpts0 = np.array(kpts0[matched_mask])
            matched_kpts1 = np.array(kpts1[matches[matched_mask]])
            
            matching_pair_ids = [[idx, int(matches[idx].numpy())] for idx in range(len(matches))]
            F, inlier_mask = cv2.findFundamentalMat(matched_kpts0, matched_kpts1, cv2.USAC_MAGSAC, 
                                    ransacReprojThreshold=0.25, confidence=0.99999, maxIters=10000)
            
            matches_after_ransac = []
            for _match, is_inlier in zip(matching_pair_ids, inlier_mask):
                if is_inlier == 1:
                    matches_after_ransac.append(_match)
            matches_after_ransac = np.array(matches_after_ransac) 
            
            # inlier keypoints
            inlier_kpts0 = kpts0_array[matches_after_ransac[:, 0], :]
            inlier_kpts1 = kpts1_array[matches_after_ransac[:, 1], :]
            
            E, R, T = compute_essential_matrix(F, calibration_dict[id0].K, calibration_dict[id1].K, inlier_kpts0, inlier_kpts1)
            q = quaternion_from_matrix(R)
            T = T.flatten()
            
            # Get the ground truth relative pose difference for this pair of images.
            R0_gt, T0_gt = calibration_dict[id0].R, calibration_dict[id0].T.reshape((3, 1))
            R1_gt, T1_gt = calibration_dict[id1].R, calibration_dict[id1].T.reshape((3, 1))
            dR_gt = np.dot(R1_gt, R0_gt.T)
            dT_gt = (T1_gt - np.dot(dR_gt, T0_gt)).flatten()
            q_gt = quaternion_from_matrix(dR_gt)
            q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
            
            # Given ground truth and prediction, compute the error for the example above.
            err_q, err_t = compute_error_for_one_example(q_gt, dT_gt, q, T, scale_dicts[name])
            errors[name][pair] = [err_q, err_t]
            print(f'{pair}, err_q={(err_q):.02f} (deg), err_t={(err_t):.02f} (m)', flush=True)
        
        # Histogram the errors over this scene.
        mAA[name] = compute_MAA([v[0] for v in errors[name].values()], [v[1] for v in errors[name].values()], thresholds_q, thresholds_t)
        print(f'Mean average Accuracy on "{name}": {mAA[name][0]:.05f}')
        print()

print('------- SUMMARY -------')
print()
for scene in scale_dicts.keys():
    print(f'-- Mean average Accuracy on "{scene}": {mAA[scene][0]:.05f}')
print()
print(f'Mean average Accuracy on dataset: {np.mean([mAA[scene][0] for scene in mAA]):.05f}')