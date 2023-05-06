from timm import create_model
from timm.data import create_transform
from timm.models import get_pretrained_cfg
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader

total_image_num = 99000000
model_name = 'vit_small_patch16_224_in21k'
device = 'cuda:3'
batch_size = 64


class CustomImageDataset(Dataset):
    def __init__(self, img_path='./towhee_bird0.jpg', transform=None):
        self.img_path = img_path
        self.transform = transform

    def __len__(self):
        return total_image_num

    def __getitem__(self, idx):
        image = Image.open(self.img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image


config = get_pretrained_cfg(model_name)
print(config)
tfms = create_transform(
    input_size=config['input_size'],
    interpolation=config['interpolation'],
    mean=config['mean'],
    std=config['std'],
    crop_pct=config['crop_pct']
)

dataset = CustomImageDataset(transform=tfms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
model = create_model(model_name, pretrained=True, num_classes=10).to(device)

for batch_ind in tqdm(range(total_image_num // batch_size)):
    batch_data = next(iter(dataloader))
    # print(batch_data.shape)
    feature = model.forward_features(batch_data.to(device))
    # print("feature.shape = ", feature.shape)
    # print()

# from towhee.dc2 import pipe, ops, DataCollection
# op = ops.image_embedding.timm(model_name='vit_small_patch16_224_in21k', device='cuda:3')
# 
# 
# 
# train_loader = Data.DataLoader(dataset=training_data, batch_size=BATCH_SIZE,
#                 shuffle=True)
# p = (
#     pipe.input('path')
#         .map('path', 'img', ops.image_decode())
#         .map('img', 'vec', op)
#         .output('img', 'vec')
# )
# t0 = time.time()
# p.batch(['./towhee_bird.jpeg']*100)
# t1 = time.time()
# t = t1 - t0
# print('t = ', t)


# t_list = []
# times = 100
# for i in range(times):
#     t0 = time.time()
#     p('./towhee_bird.jpeg')
#     t1 = time.time()
#     t = t1 - t0
#     if i == 0 :
#         print('first time = ', t)
#     else:
#         t_list.append(t)
# print(f'from 1 to {times}, average time = {sum(t_list) / times}')
