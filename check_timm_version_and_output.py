import timm
import torch
from PIL import Image
from timm import create_model
from timm.data import create_transform
try:
    from timm.models import get_pretrained_cfg  # 0.6.12
except ImportError:
    from timm.models.registry import _model_default_cfgs  # 0.4.12
    def get_pretrained_cfg(model_name):
        return _model_default_cfgs[model_name]

image = Image.open('towhee_bird0.jpg').convert('RGB')
device = 'cuda:3'
model_name = 'vit_tiny_patch16_224_in21k'
config = get_pretrained_cfg(model_name)
print(config)
tfms = create_transform(
    input_size=config['input_size'],
    interpolation=config['interpolation'],
    mean=config['mean'],
    std=config['std'],
    crop_pct=config['crop_pct']
)
image = tfms(image)
model = create_model(model_name, pretrained=True, num_classes=10).to(device)
batch_data = torch.unsqueeze(image, dim=0)
feature = model.forward_features(batch_data.to(device))
print(feature.shape)
if len(feature.shape) == 3:
    # print(feature[0][0][:20])
    # print(feature[0][0][-20:])
    cls_token_feature = feature[0, 0, :]
    print(cls_token_feature.shape)
    print(cls_token_feature[:10])
    print(cls_token_feature[-10:])
else:
    cls_token_feature = feature[0, :]
    print(cls_token_feature.shape)
    print(cls_token_feature[:10])
    print(cls_token_feature[-10:])
print('')