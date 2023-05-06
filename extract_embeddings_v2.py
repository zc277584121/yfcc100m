import argparse
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

from download_repair_error import hex_range


def process_zip(zip_path):
    img_key_2_zip = {}
    with zipfile.ZipFile(zip_path, 'r') as z:
        for key in z.namelist():
            img_key_2_zip[key] = zip_path
    return img_key_2_zip


class YFCCZipsDataset(Dataset):
    def __init__(self, zip_file, transform=None, extract_root='./extract_folder'):
        self.zip_file = zip_file
        self.transform = transform
        self.extract_root = extract_root
        # self.lock = threading.Lock()
        self.img_paths = []
        if not os.path.exists(self.extract_root):
            os.makedirs(self.extract_root)
        self.extract_folder = os.path.join(extract_root, os.path.basename(self.zip_file).split('.')[0])
        with zipfile.ZipFile(self.zip_file, 'r') as z:
            z.extractall(self.extract_folder)
            self.img_paths = [os.path.join(self.extract_folder, rel_path) for rel_path in z.namelist()]
        self.img_paths = sorted(self.img_paths)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img_key = img_path.strip().split('/')[-1].split('.')[0]
        # print(f'{idx}-{img_key}')
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return img_key, image
        except:
            with open(f'errs/{img_key}.txt', 'a') as f:
                f.write(f'{img_path}\n')
            return img_key, None


def collate_fn(batch):
    batch = list(filter(lambda x: x[1] is not None, batch))
    if len(batch) == 0:
        return [], []

    img_keys, images = zip(*batch)
    images = torch.stack(images, dim=0)
    return img_keys, images


def extract_one_zip(args, zip_id):
    model_name = args.model_name
    device_id = args.device_id
    batch_size = args.batch_size
    device = f'cuda:{device_id}'

    config = get_pretrained_cfg(model_name)
    # print(config)
    tfms = create_transform(
        input_size=config['input_size'],
        interpolation=config['interpolation'],
        mean=config['mean'],
        std=config['std'],
        crop_pct=config['crop_pct']
    )

    extract_root = './extract_folder'
    dataset = YFCCZipsDataset(zip_file=f'images/{zip_id}.zip', transform=tfms, extract_root=extract_root)
    print(f'image num = {len(dataset)}')

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn)  # , num_workers=8) unsupported num_workers
    model = create_model(model_name, pretrained=True, num_classes=10).to(device)

    with h5py.File(f"features/features_{zip_id}.hdf5", "a") as f:
        all_keys = f.keys()
        for batch_keys, batch_data in tqdm(dataloader):
            if len(batch_keys) == 0:
                continue
            key_miss = False
            for key in batch_keys:
                if key not in all_keys:
                    key_miss = True
                    break
            if not key_miss:
                continue

            feature = model.forward_features(batch_data.to(device))
            feature = feature[:, 0, :].detach().to('cpu').numpy()
            for i in range(len(batch_data)):
                key = batch_keys[i]
                data = feature[i]
                f.create_dataset(key, data=data, compression="gzip", dtype="f", compression_opts=9)
    extract_id_root = os.path.join(extract_root, zip_id)
    if os.path.isdir(extract_id_root):
        shutil.rmtree(extract_id_root)


def extract_zips(args):
    start = args.start
    end = args.end
    shard_list = hex_range(start, end)
    for shard in shard_list:
        print(f'extract {shard}...')
        extract_one_zip(args=args, zip_id=shard)


if __name__ == '__main__':
    formatter = lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog, max_help_position=80)
    parser = argparse.ArgumentParser(description='feature extract collect', formatter_class=formatter)

    parser.add_argument('--model_name', type=str, default='vit_tiny_patch16_224_in21k')
    parser.add_argument('--device_id', type=str, default='0')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--start', type=str, required=True)
    parser.add_argument('--end', type=str, required=True)
    args = parser.parse_args()
    print(args)
    t0 = time.time()
    extract_zips(args)
    t1 = time.time()
    print(f'time = {t1 - t0}')
