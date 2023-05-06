import os
import shutil
import time
from collections import defaultdict
from glob import glob

import zipfile

import h5py
import torch
from timm import create_model
from timm.data import create_transform
from timm.models import get_pretrained_cfg
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader
from multiprocessing import Pool
import threading

# total_image_num = 99000000
# model_name = 'vit_small_patch16_224_in21k'
from download_repair_error import hex_range

model_name = 'vit_tiny_patch16_224_in21k'
# device_id = '2'
# batch_size = 64
# start = '219'
# end = '2ff'

# device_id = '3'
# batch_size = 64
# start = '300'
# end = '31f'

device_id = '6'
batch_size = 64
start = '360'
end = '37f'




device = f'cuda:{device_id}'


def process_zip(zip_path):
    img_key_2_zip = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for key in z.namelist():
            img_key_2_zip[key] = zip_path
    return img_key_2_zip


class YFCCZipsDataset(Dataset):
    def __init__(self, zip_root='images', transform=None, device='0', extract_folder='tmp_extract'):
        self.zip_root = zip_root
        self.transform = transform
        self.extract_folder = extract_folder
        self.device = device
        # self.lock = threading.Lock()
        if not os.path.exists(self.extract_folder):
            os.makedirs(self.extract_folder)

        # build id2zip
        self.img_key_2_zip = {}
        self.zip_2_img_keys = defaultdict(list)
        self.zip_paths = glob(os.path.join(zip_root, '*.zip'))
        zip_range = hex_range(start, end)
        range_zip_paths = [os.path.join(zip_root, zip_str) + '.zip' for zip_str in zip_range]
        self.zip_paths = list(set(self.zip_paths) & set(range_zip_paths))

        # for zip_path in tqdm(self.zip_paths):
        #     with zipfile.ZipFile(zip_path, 'r') as z:
        #         for key in z.namelist():
        #             self.img_key_2_zip[key] = zip_path
        with Pool(processes=16) as pool:
            results = list(tqdm(pool.imap(process_zip, self.zip_paths), total=len(self.zip_paths)))
        # self.img_key_2_zip = {}
        for result in tqdm(results):
            self.img_key_2_zip.update(result)
            for img_key, zip_path in result.items():
                self.zip_2_img_keys[zip_path].append(img_key)

            # self.zip_2_img_keys[]
        self.img_keys = sorted(list(self.img_key_2_zip.keys()))
        self.extract_device_folder = os.path.join(self.extract_folder, f'device_{self.device}')
        if not os.path.exists(self.extract_device_folder):
            os.makedirs(self.extract_device_folder)

    def __len__(self):
        return len(self.img_keys)

    def __getitem__(self, idx):
        # t0 = time.time()
        img_key = self.img_keys[idx]
        # print(f'{idx}-{img_key}')
        zip_path = self.img_key_2_zip[img_key]
        maybe_extract_path = os.path.join(self.extract_device_folder, os.path.basename(zip_path).split('.')[0])
        if not os.path.exists(maybe_extract_path):
            os.makedirs(maybe_extract_path)
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(maybe_extract_path)
        # else:
        #     print(f'{maybe_extract_path} already exists.')
        try:
            img_path = os.path.join(maybe_extract_path, img_key)
            # print(img_path)
            # t00 = time.time()
            image = Image.open(img_path).convert('RGB')
            # t11 = time.time()
            if self.transform:
                image = self.transform(image)
            # t22 = time.time()
            self.zip_2_img_keys[zip_path].remove(img_key)
            # t33 = time.time()
            if len(self.zip_2_img_keys[zip_path]) == 0:
                # print(f'retree {maybe_extract_path}')
                shutil.rmtree(maybe_extract_path)
            # t44 = time.time()
            img_key = img_key.strip().split('/')[-1].split('.')[0]
            # t1 = time.time()
            # print(f'__getitem__ time = {t1 - t0}')
            # print(f'open image time = {t11 - t00}')
            # print(f'transform time = {t22 - t11}')
            # print(f'remove defaultdict list time = {t33 - t22}')
            # print(f'maybe rmtree time = {t44 - t33}')
            return img_key, image
        except:
            return img_key, None


def collate_fn(batch):
    # t0 = time.time()
    batch = list(filter(lambda x: x[1] is not None, batch))
    if len(batch) == 0:
        return [], []

    img_keys, images = zip(*batch)
    images = torch.stack(images, dim=0)
    # t1 = time.time()
    # print(f'collate_fn time = {t1 - t0}')
    return img_keys, images


config = get_pretrained_cfg(model_name)
print(config)
tfms = create_transform(
    input_size=config['input_size'],
    interpolation=config['interpolation'],
    mean=config['mean'],
    std=config['std'],
    crop_pct=config['crop_pct']
)

dataset = YFCCZipsDataset(transform=tfms, device=device_id)
print(f'len(dataset) = {len(dataset)}')

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)  # , num_workers=8) unsupported num_workers
model = create_model(model_name, pretrained=True, num_classes=10).to(device)

# with h5py.File('...hdf5', 'a') as hdf5_file:
#     hdf5_features = hdf5_file.create_dataset(video_id, data=video_features, dtype="f", compression='gzip',
#                                              compression_opts=9)


with h5py.File(f"features_{start}-{end}.hdf5", "a") as f:
    all_keys = f.keys()
    for batch_keys, batch_data in tqdm(dataloader):
        # t0 = time.time()
        if len(batch_keys) == 0:
            continue
        key_miss = False
        for key in batch_keys:
            if key not in all_keys:
                key_miss = True
                break
        if not key_miss:
            continue
        # t1 = time.time()
        # new_batch_data = []
        # new_batch_keys = []
        # for ii in range(len(batch_keys)):
        #     if batch_data[ii] is None:
        #         continue
        #     new_batch_data.append(batch_data[ii])
        #     new_batch_keys.append(batch_keys[ii])
        # new_batch_data = torch.stack(new_batch_data)

        feature = model.forward_features(batch_data.to(device))
        # t2 = time.time()
        feature = feature[:, :, 0].detach().to('cpu').numpy()
        # t3 = time.time()
        # 将feature和key保存到HDF5文件中
        for i in range(len(batch_data)):
            key = batch_keys[i]
            data = feature[i]
            f.create_dataset(key, data=data, compression="gzip", dtype="f", compression_opts=9)
        # t4 = time.time()
        # print(f'batch judge missing time = {t1 - t0}')
        # print(f'batch forward feature time = {t2 - t1}')
        # print(f'batch tensor to numpy time = {t3 - t2}')
        # print(f'batch numpy to h5 time = {t4 - t3}')
