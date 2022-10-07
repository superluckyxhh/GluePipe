import os
import math
import cv2
import types
import json
import numpy as np
import torch
from tqdm import tqdm
import torch
import time
from omegaconf import OmegaConf
from models import get_matching_module
from extractors.superpoint.model import SuperPoint
from models.superglue.model import SuperGlue


def error_auc(errors, thresholds=[3, 5, 10]):
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = []
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index-1]]
        x = errors[:last_index] + [thr]
        aucs.append(np.trapz(y, x) / thr)

    return {f'auc@{t}': auc for t, auc in zip(thresholds, aucs)}


def get_bitmap(image_path, new_shape=None):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1]
    ori_shape = image.shape[:2]

    if new_shape:
        new_shape = np.array(new_shape)
        ori_shape = np.array(ori_shape)
        scale = max(new_shape / ori_shape)

        image = image[:int(new_shape[0] / scale), 
                      :int(new_shape[1] / scale)]
        image = cv2.resize(
            image, (new_shape[1], new_shape[0]),
            interpolation=cv2.INTER_AREA
        )

    # image = image.astype(np.float32) / 255.
    # image = torch.from_numpy(image).permute(2, 0, 1)

    return image, ori_shape


def adapt_homography_to_processing(H, new_shape, ori_shape0, ori_shape1):
    new_shape = np.array(new_shape)
    ori_shape0 = np.array(ori_shape0)
    ori_shape1 = np.array(ori_shape1)

    scale0 = max(new_shape / ori_shape0)
    up_scale = np.diag(np.array([1. / scale0, 1. / scale0, 1.]))

    scale1 = max(new_shape / ori_shape1)
    down_scale = np.diag(np.array([scale1, scale1, 1.]))

    H = down_scale @ H @ up_scale
    return H


def generate_sift(bitmap, extractor):
    gray = cv2.cvtColor(bitmap, cv2.COLOR_RGB2GRAY)
    kpts = np.array([[kp.pt[0], kp.pt[1]] for kp in extractor.detect(gray)])
    return torch.from_numpy(kpts).float()


def generate_superpoint(bitmap, extractor):
    gray = cv2.cvtColor(bitmap, cv2.COLOR_RGB2GRAY)
    gray = torch.from_numpy(gray / 255.)[None].float()

    preds = extractor({'image': gray[None].to('cuda')})
 
    return {k: torch.stack(v) for k, v in preds.items()}


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


def main(cfg, config):
    assert os.path.exists(cfg.path_data)
    scenes = sorted(os.listdir(cfg.path_data))

    matching_config = config['matching']
    train_config = config['train']
    
    # Load Model and Weights (IF EXISTED)
    matcher = get_matching_module(matching_config['name'])(matching_config).eval().to(cfg.device)
    if train_config.get('sota_weight_path', None) is not None:
        weight_path = train_config['sota_weight_path']
        state_dicts = torch.load(weight_path)
        matcher.load_state_dict(state_dicts['model'])
        print(f'Loaded Trained Weight {weight_path} Done')

    extractor = SuperPoint({
        'nms_radius': 1,
        'keypoint_threshold': 0.01,
        'max_keypoints': 2048
    }).to(cfg.device)

    i_results = []
    v_results = []
    mean_of_inliers = 0
    for scene in tqdm(scenes[::-1], total=len(scenes)):
        scene_cls = scene.split('_')[0]
        path_scene = os.path.join(cfg.path_data, scene)

        path0 = os.path.join(path_scene, '1.ppm')
        im0, ori_shape0 = get_bitmap(path0, cfg.resize)
        # im0 = im0.unsqueeze(0).to(cfg.device)
        features0 = generate_superpoint(
            im0, extractor
        )  # .unsqueeze(0).to(cfg.device)
        keypoints0 = features0['keypoints']
        scores0 = features0['scores']
        descriptors0 = features0['descriptors']
        descriptors0 = descriptors0.permute(0, 2, 1)

        shape = im0.shape[:2]
        corners = np.array([
            [0,            0,            1],
            [shape[1] - 1, 0,            1],
            [0,            shape[0] - 1, 1],
            [shape[1] - 1, shape[0] - 1, 1]
        ])

        sum_of_inliers = 0
        for idx in range(2, 7):
            path1 = os.path.join(path_scene, f'{idx}.ppm')
            im1, ori_shape1 = get_bitmap(path1, cfg.resize)
            features1 = generate_superpoint(
                im1, extractor
            )  # .unsqueeze(0).to(cfg.device)

            keypoints1 = features1['keypoints']  
            scores1 = features1['scores']
            descriptors1 = features1['descriptors']
            descriptors1 = descriptors1.permute(0, 2, 1)

            pair_data = {
                'keypoints0':keypoints0,
                'keypoints1':keypoints1,
                'scores0':scores0,
                'scores1':scores1,
                'descriptors0':descriptors0,
                'descriptors1':descriptors1,
                'image0_size':cfg.resize[::-1],
                'image1_size':cfg.resize[::-1],
            }
            
            pred = matcher.inference(pair_data)
            
            kpts0, kpts1 = pred['keypoints0'].detach().cpu().numpy(), pred['keypoints1'].detach().cpu().numpy()
            matches, conf = pred['matches0'].detach().cpu().numpy(), pred['matching_scores0'].detach().cpu().numpy()
            matched_mask = matches > -1
            ids1 = matches[matched_mask]
            mkpts0 = kpts0[matched_mask]
            mkpts1 = kpts1[:, ids1].squeeze(0)
            
            if len(mkpts0) < 4:
                # print('Match Failed, return error=999')
                failed_error = 999
                if scene_cls == 'i':
                    i_results.append(failed_error)
                else:
                    v_results.append(failed_error)
                continue

            H_pred, inliers = cv2.findHomography(
                mkpts0, mkpts1, method=cv2.RANSAC
            )
            if H_pred is None:
                # print('Find Homography Failed, return error=999')
                failed_error = 999
                if scene_cls == 'i':
                    i_results.append(failed_error)
                else:
                    v_results.append(failed_error)
                continue

            inliers = inliers.flatten().astype(bool)
            n_inliers = np.sum(inliers)
            sum_of_inliers += float(n_inliers)

            pred_corners = np.dot(corners, np.transpose(H_pred))
            pred_corners = pred_corners[:, :2] / pred_corners[:, 2:]

            H_real = np.loadtxt(os.path.join(path_scene, f'H_1_{idx}'))
            H_real = adapt_homography_to_processing(
                H_real, cfg.resize, ori_shape0, ori_shape1
            )

            # Real corners
            real_corners = np.dot(corners, np.transpose(H_real))
            real_corners = real_corners[:, :2] / real_corners[:, 2:]

            error = np.mean(
                np.linalg.norm(real_corners - pred_corners, axis=1)
            )
            # print(f'[{scene_cls}]: Inliers: {n_inliers} | Error: {error}')

            if scene_cls == 'i':
                i_results.append(error)
            else:
                v_results.append(error)

        mean_of_inliers += sum_of_inliers / 5.

    mean_of_inliers /= float(len(scenes))

    v_results = np.array(v_results).astype(np.float32)
    i_results = np.array(i_results).astype(np.float32)
    results = np.concatenate((i_results, v_results), axis=0)

    # Compute auc
    auc_of_homo_i = error_auc(i_results)
    auc_of_homo_v = error_auc(v_results)
    auc_of_homo = error_auc(results)

    dumps = {
        'inliers': mean_of_inliers,
        **{k: v for k, v in auc_of_homo.items()},
        **{f'i_{k}': v for k, v in auc_of_homo_i.items()},
        **{f'v_{k}': v for k, v in auc_of_homo_v.items()},
    }
    print(f'-- Homography results: \n{dumps}')

    with open(cfg.path_dump, 'w') as json_file:
        json.dump(dumps, json_file)


if __name__ == "__main__":
    cfg = types.SimpleNamespace()
    cfg.config_path = '/home/user/code/GluePipeline/configs/cashed_config.yaml'
    config = OmegaConf.load(cfg.config_path)
    
    cfg.path_data = '/home/user/dataset/hpatches'
    log_path = config['logging']['homography_path_root']
    cfg.path_dump = os.path.join(log_path, 'homography_evaluate.json')
    
    cfg.resize = (480, 640)
    cfg.max_features = 2048

    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    main(cfg, config)