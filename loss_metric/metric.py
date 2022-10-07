import numpy as np
import torch
import cv2
import os
from tqdm import tqdm
import kornia

from utils.geometry import normalize_with_intrinsics


class AccuracyUsingEpipolarDist:
    def __init__(self, threshold=5e-4):
        self.threshold = threshold
        self.precision = []
        self.matching_score = []

    def get_p_ms(self, matched_kpts0, matched_kpts1, transformation, num_detected_kpts):
        K0 = transformation['K0']
        K1 = transformation['K1']
        R = transformation['R']
        T = transformation['T'].unsqueeze(-1)

        E = kornia.geometry.epipolar.essential_from_Rt(
            R1=torch.eye(3, device=R.device).unsqueeze(0),
            t1=torch.zeros(1, 3, 1, device=R.device),
            R2=R.unsqueeze(0), t2=T.unsqueeze(0)
        )

        num_matched_kpts = matched_kpts0.shape[0]
        if num_matched_kpts > 0:
            matched_kpts0 = normalize_with_intrinsics(matched_kpts0, K0)
            matched_kpts1 = normalize_with_intrinsics(matched_kpts1, K1)

            epipolar_dist = kornia.geometry.epipolar.symmetrical_epipolar_distance(
                matched_kpts0.unsqueeze(0),
                matched_kpts1.unsqueeze(0),
                E
            ).squeeze(0)

            num_correct_matches = (epipolar_dist < self.threshold).sum()
            precision = num_correct_matches / num_matched_kpts
            matching_score = num_correct_matches / num_detected_kpts
        else:
            precision, matching_score = matched_kpts0.new_tensor(0.), matched_kpts0.new_tensor(0.)
        self.precision.append(precision)
        self.matching_score.append(matching_score)
        

    def compute_precision_matchscore(self):
        return {
            'Precision':torch.mean(torch.tensor(self.precision)),
            'Matching Score': torch.mean(torch.tensor(self.matching_score))
        }
        
        
class CameraPoseAUC:
    def __init__(self, auc_thresholds, ransac_inliers_threshold):
        self.auc_thresholds = auc_thresholds
        self.ransac_inliers_threshold = ransac_inliers_threshold
        self.pose_errors = []

    @staticmethod
    def __rotation_error(R_true, R_pred):
        angle = torch.arccos(torch.clip(((R_true * R_pred).sum() - 1) / 2, -1, 1))
        return torch.abs(torch.rad2deg(angle))

    @staticmethod
    def __translation_error(T_true, T_pred):
        angle = torch.arccos(torch.cosine_similarity(T_true, T_pred, dim=0))[0]
        angle = torch.abs(torch.rad2deg(angle))
        return torch.minimum(angle, 180. - angle)
    
    def get_pose_error(self, matched_kpts0, matched_kpts1, transformation):
        device = matched_kpts0.device
        K0 = transformation['K0']
        K1 = transformation['K1']
        R = transformation['R']
        T = transformation['T'].unsqueeze(-1)

        # estimate essential matrix from point matches in calibrated space
        num_matched_kpts = matched_kpts0.shape[0]
        if num_matched_kpts >= 5:
            # convert to calibrated space and move to cpu for OpenCV RANSAC
            matched_kpts0_calibrated = normalize_with_intrinsics(matched_kpts0, K0).cpu().numpy()
            matched_kpts1_calibrated = normalize_with_intrinsics(matched_kpts1, K1).cpu().numpy()

            threshold = 2 * self.ransac_inliers_threshold / (K0[[0, 1], [0, 1]] + K1[[0, 1], [0, 1]]).mean()
            E, mask = cv2.findEssentialMat(
                matched_kpts0_calibrated,
                matched_kpts1_calibrated,
                np.eye(3),
                threshold=float(threshold),
                prob=0.99999,
                method=cv2.RANSAC
            )
            if E is None:
                error = torch.tensor(np.inf).to(device)
            else:
                E = torch.FloatTensor(E).to(device)
                mask = torch.BoolTensor(mask[:, 0]).to(device)

                best_solution_n_points = -1
                best_solution = None
                for E_chunk in E.split(3):
                    R_pred, T_pred, points3d = kornia.geometry.epipolar.motion_from_essential_choose_solution(
                        E_chunk, K0, K1,
                        matched_kpts0, matched_kpts1,
                        mask=mask
                    )
                    n_points = points3d.size(0)
                    if n_points > best_solution_n_points:
                        best_solution_n_points = n_points
                        best_solution = (R_pred, T_pred)
                R_pred, T_pred = best_solution

                R_error, T_error = self.__rotation_error(R, R_pred), self.__translation_error(T, T_pred)
                error = torch.maximum(R_error, T_error)
        else:
            error = torch.tensor(np.inf).to(device)
        self.pose_errors.append(error)
    
    def compute_pose_auc(self):
        errors = self.pose_errors
        errors = torch.sort(torch.tensor(errors)).values
        recall = (torch.arange(len(errors), device=errors.device) + 1) / len(errors)
        zero = torch.zeros(1, device=errors.device)
        errors = torch.cat([zero, errors])
        recall = torch.cat([zero, recall])

        aucs = {}
        for threshold in self.auc_thresholds:
            threshold = torch.tensor(threshold).to(errors.device)
            last_index = torch.searchsorted(errors, threshold)
            r = torch.cat([recall[:last_index], recall[last_index - 1].unsqueeze(0)])
            e = torch.cat([errors[:last_index], threshold.unsqueeze(0)])
            area = torch.trapz(r, x=e) / threshold
            aucs[f'AUC@{threshold}deg'] = area
        return aucs
    

# ****** Test HPatches *******

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


def generate_superpoint(bitmap, extractor):
    gray = cv2.cvtColor(bitmap, cv2.COLOR_RGB2GRAY)
    gray = torch.from_numpy(gray / 255.)[None].float()

    preds = extractor({'image': gray[None].to('cuda')})
 
    return {k: torch.stack(v) for k, v in preds.items()}


def test_hpatches(matcher, extractor, cur_epoch, max_epoch):
    path_data = '/home/user/dataset/hpatches'
    resize = (480, 640)
    max_features = 2048
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    assert os.path.exists(path_data)
    scenes = sorted(os.listdir(path_data))
    extractor = extractor.to(device)
    
    i_results = []
    v_results = []
    mean_of_inliers = 0
    for scene in tqdm(scenes[::-1], total=len(scenes)):
        scene_cls = scene.split('_')[0]
        path_scene = os.path.join(path_data, scene)

        path0 = os.path.join(path_scene, '1.ppm')
        im0, ori_shape0 = get_bitmap(path0, resize)
        features0 = generate_superpoint(
            im0, extractor
        )
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
            im1, ori_shape1 = get_bitmap(path1, resize)
            features1 = generate_superpoint(
                im1, extractor
            )

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
                'image0_size':resize[::-1],
                'image1_size':resize[::-1],
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
                H_real, resize, ori_shape0, ori_shape1
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
    print(f'Test epoch:{cur_epoch+1} in HPatches, Homography results: \n{dumps}')
    return dumps