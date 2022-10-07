import cv2
import torch
import os
import argparse
import numpy as np
from omegaconf import OmegaConf
from extractors import get_feature_extractor
from models import get_matching_module


def read_image(im_path, resize=None, tensor_type=True):
    im = cv2.imread(im_path)
    ori_im = im.copy()
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    if resize is not None:
        new_width, new_height = resize
        im = cv2.resize(im, (new_width, new_height))
    if tensor_type:
        tim = (torch.FloatTensor(im) / 255.).unsqueeze(0).unsqueeze(0)
    else:
        tim = None
    return ori_im, im, tim


def plot_image_keypoints(im, keypoints, resize, save_path):
    """Plot Images With Keypoints
    1. Origin image (HxWx3), keypoints in origin image
    Args:
        im (_type_): Origin image (type:uint8 & channel:BGR) [H, W, 3]
        keypoints (_type_): Keypoints in resized image [1, nums, 2]
    """
    point_size = 1
    # B G R
    point_color = (0, 0, 255) 
    thickness = 2
    line_type=cv2.LINE_AA
    if resize is not None:
        new_width, new_height = resize
        im = cv2.resize(im, (new_width, new_height))

    keypoints = keypoints.squeeze(0).detach().cpu().numpy()
    keypoints = np.round(keypoints).astype(int)
    
    for coord in keypoints:
        x, y = coord
        cv2.circle(im, (x, y), point_size, point_color, thickness, lineType=line_type)

    cv2.imwrite(save_path, im)
    

def plot_matchings(im0, im1, kpts0, kpts1, matches, save_path):
    point_size = 1
    point_color = (0, 0, 255)
    line_color = (0, 255, 0)
    point_thickness = 4
    line_thickness = 1
    line_type = cv2.LINE_AA
    
    h0, w0, _ = im0.shape
    h1, w1, _ = im1.shape
    margin = 10
    
    H = h0 + h1 + margin
    W = w0
    
    plot_map = np.zeros((H, W, 3),dtype=np.uint8)
    plot_map[:h0, :, :] = im0
    plot_map[h0+margin:, :, :] = im1
    
    kpts0 = kpts0.squeeze(0).detach().cpu().numpy()
    kpts0 = np.round(kpts0).astype(int)
    kpts1 = kpts1.squeeze(0).detach().cpu().numpy()
    kpts1 = np.round(kpts1).astype(int)
    
    matches = matches.squeeze(0).detach().cpu().numpy()
    for idx0 in range(len(matches)):
        idx1 = matches[idx0]
        if idx1 == -1:
            continue
    
        x0, y0 = kpts0[idx0]
        x1, y1 = kpts1[idx1]
        y1 = y1 + margin + h0
        
        cv2.circle(plot_map, (x0, y0), point_size, point_color, point_thickness, lineType=line_type)
        cv2.circle(plot_map, (x1, y1), point_size, point_color, point_thickness, lineType=line_type)
        cv2.line(plot_map, (x0, y0), (x1, y1), line_color, line_thickness)

    cv2.imwrite(save_path, plot_map)


def main():
    parser = argparse.ArgumentParser(description='Visulaize image with keypoints in file')
    parser.add_argument('--image_root', type=str, default='/home/user/dataset/Aachen_day_night_v11/images_upright/db', help='path to image file')
    parser.add_argument('--image0_name', type=str, default='1181.jpg', help='image name')
    parser.add_argument('--image1_name', type=str, default='1189.jpg', help='image name')
    parser.add_argument('--config_path', type=str, default='/home/user/code/GluePipeline/configs/visual_config.yaml', 
                        help='path to directory with saved experiment')
    parser.add_argument('--resize', type=int, default=None, help='resize shape')
    parser.add_argument('--save_kpts_root', type=str, default='/home/user/code/GluePipeline/visual_result/keypoints', help='path to a resulting image with matched points visualized') 
    parser.add_argument('--save_match_root', type=str, default='/home/user/code/GluePipeline/visual_result/matching')
    parser.add_argument('--device', type=str, help='device to use for inference', default='cpu')
    parser.add_argument('--plot_keypoints', type=bool, default=True)
    parser.add_argument('--plot_matchings', type=bool, default=False)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    
    # im_list = [os.path.join(args.image_root, im_name) for im_name in os.listdir(args.image_root)]
    im0_path = os.path.join(args.image_root, args.image0_name)
    im1_path = os.path.join(args.image_root, args.image1_name)
    
    ori_im0, im0, tim0 = read_image(im0_path, resize=args.resize)
    ori_im1, im1, tim1 = read_image(im1_path, resize=args.resize)
    timm = torch.cat([tim0, tim1], dim=0)
    
    print('origin image0 shape:', ori_im0.shape)
    print('image0 reshape:', im0.shape)
    print('tensor image0 shape:', tim0.shape)
    
    print('origin image1 shape:', ori_im1.shape)
    print('image1 reshape:', im1.shape)
    print('tensor image1 shape:', tim1.shape)
    
    print('batch images shape:', timm.shape)
    
    feature_extractor = get_feature_extractor(feature_config['extractor_name'])(feature_config)
    im_dict = {'image':timm}
    feature_dict = feature_extractor(im_dict)
    
    keypoints = torch.stack(feature_dict['keypoints'], dim=0)
    scores = torch.stack(feature_dict['scores'], dim=0)
    descriptors = torch.stack(feature_dict['descriptors'], dim=0)
    
    keypoints0 = keypoints[0].unsqueeze(0)
    keypoints1 = keypoints[1].unsqueeze(0)
    scores0 = scores[0].unsqueeze(0)
    scores1 = scores[1].unsqueeze(0)
    descriptors0 = descriptors[0].unsqueeze(0)
    descriptors1 = descriptors[1].unsqueeze(0)
        
        
    print('Keypoints0 shape:', keypoints0.shape)
    print('Keypoints1 shape:', keypoints1.shape)
    print('Scores0 shape:', scores0.shape)
    print('Scores1 shape:', scores1.shape)
    print('Descriptors0 shape:', descriptors0.shape)
    print('Descriptors1 shape:', descriptors1.shape)
    
    if args.plot_keypoints:
        save_path0 = os.path.join(args.save_kpts_root, feature_config['extractor_name'].lower()+'_'+args.image0_name)
        save_path1 = os.path.join(args.save_kpts_root, feature_config['extractor_name'].lower()+'_'+args.image1_name)
        
        plot_image_keypoints(ori_im0, keypoints0, args.resize, save_path0)
        plot_image_keypoints(ori_im1, keypoints1, args.resize, save_path1)

    feature_dict = {
        'keypoints0': keypoints0,
        'keypoints1': keypoints1,
        'scores0': scores0,
        'scores1': scores1,
        'descriptors0': descriptors0,
        'descriptors1': descriptors1,
        'image0_size': tim0.shape,
        'image1_size': tim1.shape,
    }
    
    matcher_config = config['matching']

    matcher = get_matching_module(matcher_config['matching_name'])(matcher_config)
    matching_result = matcher(feature_dict)
    # use -1 for invalid match
    matches0 = matching_result['matches0']
    matches1 = matching_result['matches1']
    
    if args.plot_matchings:
        save_path = os.path.join(args.save_match_root, args.image0_name[:-4]+'_'+args.image1_name[:-4]+'.jpg')
        plot_matchings(ori_im0, ori_im1, keypoints0, keypoints1, matches0, save_path)

if __name__ == '__main__':
    main()