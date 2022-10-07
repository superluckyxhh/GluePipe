from dataset.megadepth import MegaDepthCachedDataset
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, BatchSampler


with open('/home/user/code/GluePipeline/asset_split/megadepth_valid_2.0.txt', 'r') as f:
    all_scenes = []
    scenes = f.readlines()
    for line in scenes:
        scene = line.strip('\n')
        all_scenes.append(scene)



ds = MegaDepthCachedDataset(
    '/home/user/dataset/MegaDepth/', 
    all_scenes,
    '/home/user/dataset/MegaDepth/features/SuperPointNet_960_720', 
    [0.15, 0.7], 
    1024,
    [960, 720], 
    True, False, 
    None
)

batch_size = 4
loader = DataLoader(ds, batch_size, shuffle=True, collate_fn=ds.collate_fn)

# train_sampler = RandomSampler(train_dataset, num_samples=train_config['steps_per_epoch'])
# batch_train_sampler = BatchSampler(
#     train_sampler, data_config['batch_size_per_gpu'], drop_last=True
# )
# train_loader = DataLoader(
#     train_dataset, 
#     batch_sampler=batch_train_sampler,
#     collate_fn=train_dataset.collate_fn,
#     # num_workers=data_config['dataloader_workers_per_gpu'],
#     pin_memory=True,
# )
    
for i, batch_data in enumerate(tqdm(loader)):
    lafs0, responses0, desc0 = batch_data['keypoints0'], batch_data['scores0'], batch_data['descriptors0']
    lafs1, responses1, desc1 = batch_data['keypoints1'], batch_data['scores1'], batch_data['descriptors1']
    im0_size, im1_size = batch_data['image0_size'], batch_data['image1_size']
    
    gt_matches0 = batch_data['ground_truth']['gt_matches0']
    gt_matches1 = batch_data['ground_truth']['gt_matches1']
    
    transformation = batch_data['transformation']
    K0, K1, R, T = transformation['K0'], transformation['K1'], transformation['R'], transformation['T']
    depth0, depth1 = transformation['depth0'], transformation['depth1']

    print(f'**************{i}-th iter**************')
    print('kpts0 shape:', lafs0.shape)
    print('kpts1 shape:', lafs1.shape)
    print('responses0 shape:', responses0.shape)
    print('responses1 shape:', responses1.shape)
    print('desc0 shape:', desc0.shape)
    print('desc1 shape:', desc1.shape)
    print('im0 size:', im0_size)
    print('im1 size:', im1_size)
    print('-'*20)
    
    print('gt0 shape:', gt_matches0.shape)
    print('gt1 shape:', gt_matches1.shape)
    print('-'*20)
    
    print('K0 shape:', K0.shape)
    print('K1 shape:', K1.shape)
    print('R shape:', R.shape)
    print('T shape:', T.shape)
    print('depth0 shape:', depth0.shape)
    print('depth1 shape:', depth1.shape)
    
    
    
    print('-'*30)