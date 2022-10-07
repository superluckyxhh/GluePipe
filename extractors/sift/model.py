import cv2
import torch
import numpy as np
from scipy.spatial import KDTree


class SIFT(torch.nn.Module):
    def __init__(self, nms_radius, max_keypoints, normalize_desc=True, root_norm=True):
        super().__init__()
        self.nms_radius = nms_radius
        self.max_keypoints = max_keypoints
        self.normalize_desc = normalize_desc
        self.root_norm = root_norm
        self.features = cv2.SIFT_create(contrastThreshold=-10000, edgeThreshold=-10000)
    
    
    def compute_features(self, im):
        kpts, descriptors = self.features.detectAndCompute(im, None)
        kpts = np.array(kpts)

        responses = np.array([k.response for k in kpts], dtype=np.float32)
        kpts_pt = np.array([k.pt for k in kpts], dtype=np.float32)

        if self.nms_radius > 0:
            nms_mask = nms_keypoints(kpts_pt, responses, self.nms_radius)
        else:
            nms_mask = np.ones((kpts_pt.shape[0],), dtype=bool)

        responses = responses[nms_mask]
        kpts = kpts[nms_mask]
        top_score_idx = np.argpartition(-responses, min(self.max_keypoints, len(responses) - 1))[:self.max_keypoints]

        return kpts[top_score_idx], responses[top_score_idx], descriptors[nms_mask][top_score_idx]


    def normalize_descriptors(self, descriptors: np.ndarray, root_norm: bool = True) -> np.ndarray:
        """
        Normalize descriptors.
        If root_norm=True apply RootSIFT-like normalization, else regular L2 normalization.
        Args:
            descriptors: array (N, 128) with unnormalized descriptors
            root_norm: boolean flag indicating whether to apply RootSIFT-like normalization

        Returns:
            descriptors: array (N, 128) with normalized descriptors
        """
        descriptors = descriptors.astype(np.float32)
        if root_norm:
            # L1 normalize
            norm = np.linalg.norm(descriptors, ord=1, axis=1, keepdims=True)
            descriptors /= norm
            # take square root of descriptors
            descriptors = np.sqrt(descriptors)
        else:
            # L2 normalize
            norm = np.linalg.norm(descriptors, ord=2, axis=1, keepdims=True)
            descriptors /= norm
            
        return descriptors
    
    
    def forward(self, im):
        keypoints, scores, descriptors = self.compute_features(im)
        
        if self.normalize_desc:
            descriptors = self.normalize_descriptors(descriptors, self.root_norm)

        return keypoints, scores, descriptors
    

def nms_keypoints(kpts: np.ndarray, responses: np.ndarray, radius: float) -> np.ndarray:
    kd_tree = KDTree(kpts)

    sorted_idx = np.argsort(-responses)
    kpts_to_keep_idx = []
    removed_idx = set()

    for idx in sorted_idx:
        # skip point if it was already removed
        if idx in removed_idx:
            continue

        kpts_to_keep_idx.append(idx)
        point = kpts[idx]
        neighbors = kd_tree.query_ball_point(point, r=radius)
        # Variable `neighbors` contains the `point` itself
        removed_idx.update(neighbors)

    mask = np.zeros((kpts.shape[0],), dtype=bool)
    mask[kpts_to_keep_idx] = True
    
    return mask