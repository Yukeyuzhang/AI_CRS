### Preprocessing
```python
from skimage.measure import regionprops
import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt

def normalize_zero_one(image):
    image = image.astype(np.float32)

    image[image<-1000] = -1000
    image[image>1000] = 1000

    minimum = np.min(image)
    # print(minimum)
    maximum = np.max(image)
    # print(maximum)
    if maximum >minimum:
        ret = (image - minimum)/(maximum - minimum)
    else:
        ret = image * 0.
    return ret

def whitening(image):
    image = image.astype(np.float32)
    mean = np.mean(image)
    std = np.std(image)
    if std > 0:
        ret = (image - mean)/std
    else:
        ret = image * 0.
    return ret

i = 1
ct_path = '/home/NumberSet'
seg_path = '/home/NumberLabel'
save_path = '/home/NumberSEG'

for i in range(1, n):
    ct = nib.load(os.path.join(ct_path, 'Number'+str(i)+'.nii.gz'))
    ct_data = ct.get_fdata()

    seg = nib.load(os.path.join(seg_path, 'Number'+str(i)+'.nii.gz'))
    seg_data = seg.get_fdata()
    mask = seg_data.astype(np.uint8)

    regions = regionprops(mask)
    z_min, y_min, x_min, z_max, y_max, x_max = regions[0].bbox


    ct_seg_data = ct_data[z_min:z_max, y_min:y_max, x_min:x_max]

    ct_seg = nib.Nifti1Image(ct_seg_data, ct.affine, ct.header)
    nib.save(ct_seg, os.path.join(save_path, 'Number'+str(i)+'.nii.gz'))

```

## CNN
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18

class VGG16_3D(nn.Module):
    def __init__(self, num_classes=2):
        super(VGG16_3D, self).__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv3d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Conv2
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Conv3
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Conv4
            nn.Conv3d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # Conv5
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 4 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x


class ResNet_3D(nn.Module):
    def __init__(self, num_classes):
        super(ResNet_3D, self).__init__()
        self.resnet3d = r3d_18(pretrained=False)
        self.resnet3d.stem[0] = torch.nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        self.resnet3d.fc = nn.Linear(self.resnet3d.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet3d(x)


class DenseBlock3D(nn.Module):
    def __init__(self, num_layers, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseBlock3D, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                nn.Sequential(
                    nn.BatchNorm3d(num_input_features + i * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(num_input_features + i * growth_rate, bn_size * growth_rate, kernel_size=1, stride=1, bias=False),
                    nn.BatchNorm3d(bn_size * growth_rate),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
                    nn.Dropout3d(drop_rate)
                )
            )

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], dim=1)
        return x

class TransitionLayer3D(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionLayer3D, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm3d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False),
            nn.AvgPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.layer(x)

class DenseNet_3D(nn.Module):
    def __init__(self, growth_rate=32, block_config=(4, 4, 4), num_init_features=64, bn_size=4, drop_rate=0, num_classes=2):
        super(DenseNet_3D, self).__init__()
        
        # Initial convolution
        self.features = nn.Sequential(
            nn.Conv3d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        # Dense blocks and transition layers
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            self.features.add_module(f'denseblock{i + 1}', DenseBlock3D(num_layers, num_features, growth_rate, bn_size, drop_rate))
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                self.features.add_module(f'transition{i + 1}', TransitionLayer3D(num_features, num_features // 2))
                num_features //= 2
        
        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm3d(num_features))
        
        # Classification layer
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool3d(x, (1, 1, 1)).view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class AlexNet_3D(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNet_3D, self).__init__()
        self.features = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            nn.Conv3d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool3d((4, 4, 4))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
```

## ViT
```python
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
import torch.nn as nn

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob ==0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=1, embed_dim=16*16*16, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size, img_size)
        patch_size = (patch_size, patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1], img_size[2] // patch_size[2])
        self.num_patches = self.grid_size[0] * self.grid_size[1] * self.grid_size[2]
        
        self.proj = nn.Conv3d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.,
                 proj_drop_ratio=0.):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim = -1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
def _init_vit_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)

    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)

class ViT3D(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_c=3, num_classes=1000,
                 embed_dim=512, depth=12, num_heads=4, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=PatchEmbed3D,
                 norm_layer=None, act_layer=None):
        super(ViT3D, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]

        self.stage1 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[0], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[1], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[2], norm_layer=norm_layer, act_layer=act_layer))

        self.stage2 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[3], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[4], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[5], norm_layer=norm_layer, act_layer=act_layer))
        
        self.stage3 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[6], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[7], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[8], norm_layer=norm_layer, act_layer=act_layer))
        
        self.stage4 = nn.Sequential(Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[9], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[10], norm_layer=norm_layer, act_layer=act_layer),
                                    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop_ratio=drop_ratio,
                                          attn_drop_ratio=attn_drop_ratio,
                                          drop_path_ratio=dpr[11], norm_layer=norm_layer, act_layer=act_layer))

        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()

        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward(self, x):
        x =self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        x = self.pos_drop(x + self.pos_embed)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.norm(x)
        x = self.pre_logits(x[:,0])
        x = self.head(x)

        return x
```

## SwinT
```python
import numpy as np
import torch
from functools import partial
import torch.nn.functional as F
import torch.nn as nn
from einops import rearrange

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep

def window_partition(x, window_size):
    x_shape = x.size()
    b, d, h, w, c = x_shape
    x = x.view(
        b,
        d // window_size[0],
        window_size[0],
        h // window_size[1],
        window_size[1],
        w // window_size[2],
        window_size[2],
        c,
    )
    windows = (
        x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous().view(-1, window_size[0] * window_size[1] * window_size[2], c)
    )

    return windows

def window_reverse(windows, window_size, dims):
    b, d, h, w = dims
    x = windows.view(
        b,
        d // window_size[0],
        h // window_size[1],
        w // window_size[2],
        window_size[0],
        window_size[1],
        window_size[2],
        -1,
    )
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous().view(b, d, h, w, -1)

    return x

def get_window_size(x_size, window_size, shift_size=None):
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class WindowAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        window_size = 3,
        qkv_bias = False,
        qk_scale = None,
        attn_drop_ratio=0.,
        proj_drop_ratio=0.,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1),
                num_heads,
            )
        )
        coords_d = torch.arange(self.window_size[0])
        coords_h = torch.arange(self.window_size[1])
        coords_w = torch.arange(self.window_size[2])

        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 2] += self.window_size[2] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[1] - 1) * (2 * self.window_size[2] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[2] - 1

        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.clone()[:n, :n].reshape(-1)
        ].reshape(n, n, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)
        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn).to(v.dtype)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(
        self,
        dim,
        num_heads,
        window_size,
        shift_size,
        mlp_ratio = 4.,
        qkv_bias = True,
        drop_ratio = 0.,
        attn_drop_ratio = 0.,
        drop_path_ratio = 0.,
        act_layer = 'GELU',
        norm_layer = nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=self.window_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop_ratio=attn_drop_ratio,
            proj_drop_ratio=drop_ratio,
        )

        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(hidden_size=dim, mlp_dim=mlp_hidden_dim, act=act_layer, dropout_rate=drop_ratio, dropout_mode="swin")

    def forward_part(self, x, mask_matrix):
        x = self.norm1(x)

        b, d, h, w, c = x.shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        pad_l = pad_t = pad_d0 = 0
        pad_d1 = (window_size[0] - d % window_size[0]) % window_size[0]
        pad_b = (window_size[1] - h % window_size[1]) % window_size[1]
        pad_r = (window_size[2] - w % window_size[2]) % window_size[2]
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b, pad_d0, pad_d1))
        _, dp, hp, wp, _ = x.shape
        dims = [b, dp, hp, wp]

        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1], -shift_size[2]), dims=(1, 2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None
        x_windows = window_partition(shifted_x, window_size)
        attn_windows = self.attn(x_windows, mask=attn_mask)
        attn_windows = attn_windows.view(-1, *(window_size + (c,)))
        shifted_x = window_reverse(attn_windows, window_size, dims)
        
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1], shift_size[2]), dims=(1, 2, 3))
        else:
            x = shifted_x

        if pad_d1 > 0 or pad_r > 0 or pad_b > 0:
            x = x[:, :d, :h, :w, :].contiguous()

        return x

    def forward(self, x, mask_matrix):
        shortcut = x
        x = self.forward_part(x, mask_matrix)
        x = shortcut + self.drop_path(x)

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class PatchMerging(nn.Module):
    
    def __init__(
            self,
            dim,
            norm_layer = nn.LayerNorm,
            ):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        x_shape = x.size()

        b, d, h, w, c = x_shape
        pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 0::2, 0::2, 1::2, :]
        x4 = x[:, 1::2, 0::2, 1::2, :]
        x5 = x[:, 0::2, 1::2, 0::2, :]
        x6 = x[:, 0::2, 0::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = self.norm(x)
        x = self.reduction(x)
        return x

def compute_mask(dims, window_size, shift_size, device): #attention mask 
    cnt = 0
  
    d, h, w = dims
    img_mask = torch.zeros((1, d, h, w, 1), device=device)
    for d in slice(-window_size[0]), slice(-window_size[0], -shift_size[0]), slice(-shift_size[0], None):
        for h in slice(-window_size[1]), slice(-window_size[1], -shift_size[1]), slice(-shift_size[1], None):
            for w in slice(-window_size[2]), slice(-window_size[2], -shift_size[2]), slice(-shift_size[2], None):
                img_mask[:, d, h, w, :] = cnt
                cnt += 1

    mask_windows = window_partition(img_mask, window_size)
    mask_windows = mask_windows.squeeze(-1)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

    return attn_mask

class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        window_size,
        drop_path,
        mlp_ratio = 4.,
        qkv_bias = False,
        drop_ratio = 0.,
        attn_drop_ratio = 0.,
        norm_layer = nn.LayerNorm,
    ):

        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.no_shift = tuple(0 for i in window_size)
        self.depth = depth
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=self.window_size,
                    shift_size=self.no_shift if (i % 2 == 0) else self.shift_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_ratio=drop_ratio,
                    attn_drop_ratio=attn_drop_ratio,
                    drop_path_ratio=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.downsample = PatchMerging(dim=dim, norm_layer=norm_layer)

    def forward(self, x):
        x_shape = x.size()

        b, c, d, h, w = x_shape
        window_size, shift_size = get_window_size((d, h, w), self.window_size, self.shift_size)
        x = rearrange(x, "b c d h w -> b d h w c")
        dp = int(np.ceil(d / window_size[0])) * window_size[0]
        hp = int(np.ceil(h / window_size[1])) * window_size[1]
        wp = int(np.ceil(w / window_size[2])) * window_size[2]
        attn_mask = compute_mask([dp, hp, wp], window_size, shift_size, x.device)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        x = x.view(b, d, h, w, -1)
        x = self.downsample(x)

        x = rearrange(x, "b d h w c -> b c d h w")

        return x

class Swin(nn.Module):
    def __init__(self, img_size, in_c, embed_dim, patch_size=2,
                 depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24), window_size=7,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., 
                 norm_layer=nn.LayerNorm, patch_norm=False, act_layer=None):
        super().__init__()

        img_size = ensure_tuple_rep(img_size, 3) #3D
        patch_size = ensure_tuple_rep(patch_size, 3)
        window_size = ensure_tuple_rep(window_size, 3) 

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.in_c = in_c
        self.patch_norm = patch_norm
        self.window_size = window_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            patch_size=self.patch_size,
            in_chans=in_c,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None,  # type: ignore
            spatial_dims=3,
        )
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.mlp_ratio = mlp_ratio

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.pos_drop = nn.Dropout(p=drop_ratio)

        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, sum(depths))]

        self.layers1 = nn.ModuleList()
        self.layers2 = nn.ModuleList()
        self.layers3 = nn.ModuleList()
        self.layers4 = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=self.window_size,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_ratio=drop_ratio,
                attn_drop_ratio=attn_drop_ratio,
                norm_layer=norm_layer,
            )
            if i_layer == 0:
                self.layers1.append(layer)
            elif i_layer == 1:
                self.layers2.append(layer)
            elif i_layer == 2:
                self.layers3.append(layer)
            elif i_layer == 3:
                self.layers4.append(layer)

        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))

    def proj_out(self, x, normalize=False):
        if normalize:
            x_shape = x.size()
            if len(x_shape) == 5:
                n, ch, d, h, w = x_shape
                x = rearrange(x, "n c d h w -> n d h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n d h w c -> n c d h w")
            elif len(x_shape) == 4:
                n, ch, h, w = x_shape
                x = rearrange(x, "n c h w -> n h w c")
                x = F.layer_norm(x, [ch])
                x = rearrange(x, "n h w c -> n c h w")
        return x

    def forward(self, x, normalize=True):
        x0 = self.patch_embed(x)
        x0 = self.pos_drop(x0)
        x0_out = x0
        x1 = self.layers1[0](x0.contiguous())
        x1_out = x1
        x2 = self.layers2[0](x1.contiguous())
        x2_out = x2
        x3 = self.layers3[0](x2.contiguous())
        x3_out = x3
        x4 = self.layers4[0](x3.contiguous())
        x4_out = self.proj_out(x4, normalize)
        return [x0_out, x1_out, x2_out, x3_out, x4_out]
    
class SwinT3D(nn.Module):
    def __init__(self, img_size, num_classes, in_c, embed_dim, out_c):
        
        super().__init__()
    
        self.swin = Swin(img_size=img_size,
                        in_c=in_c,
                        embed_dim=embed_dim,
                        )
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Sequential(
            nn.Linear(out_c, num_classes),
        )
    
    def forward(self, x): 
        x = self.swin(x)
        x = self.global_avg_pool(x[-1])
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## Main
```python
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torchsummary import summary

from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import nibabel as nib
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

import torchio as tio
import warnings
warnings.simplefilter('ignore')

from cnn_models import VGG16_3D, ResNet_3D, DenseNet_3D, AlexNet_3D
from ViTmodel import ViT3D
from SwinTmodel import SwinT3D

def normalize_zero_one(image):
    image = image.astype(np.float32)

    image[image<-1000] = -1000
    image[image>1000] = 1000

    minimum = np.min(image)
    maximum = np.max(image)
    if maximum >minimum:
        ret = (image - minimum)/(maximum - minimum)
    else:
        ret = image * 0.
    return ret

class CTScanDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, target_size=(128, 128, 128)):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = nib.load(self.image_paths[idx])
        image = image.get_fdata() 

        image = normalize_zero_one(image)
        # Convert to a tensor and add channel dimension
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # Shape: (1, D, H, W)
        image = image.unsqueeze(0)
        image = F.interpolate(image, size=self.target_size, mode='trilinear', align_corners=False)
        image = image.squeeze(0)
        
        subject = tio.Subject(
            ct=tio.ScalarImage(tensor = image)
        )

        if self.transform:
            subject = self.transform(subject)
        
        label = self.labels[idx]

        image = subject.ct.tensor

        return image, torch.tensor(label, dtype=torch.long)

def train(model, epoch, train_loader, lr, num_epochs):
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in train_loader:
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs) 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

def validation(model, val_loader):

    model.eval()
    criterion = nn.CrossEntropyLoss()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Loss: {val_loss/len(val_loader):.4f}, Validation Accuracy: {100 * correct / total:.2f}%")    

###################################################################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--mode", default='eval', help="train or eval") # MODE = 'eval'  # or 'train'
parser.add_argument("--image_dir", default='../testSEG', type=str)
parser.add_argument("--label_dir", default='../testlabel.csv', type=str)
parser.add_argument("--model_name", help="VGG_3D, ResNet_3D, DenseNet_3D, AlexNet_3D, ViT3D, SwinT3D",type=str)
parser.add_argument("--learning_rate",default=1e-6, type=float) #learning rate 1e-6
parser.add_argument("--epochs",default=100, type=int)
parser.add_argument("--batch_size",default=1, type=int) 
parser.add_argument("--augmentation",default='True', type=str)
parser.add_argument("--save_path", default='./modelWeights.pth',type=str)
parser.add_argument("--load_path",type=str)
parser.add_argument("--print_net",default=False, type=bool)
parser.add_argument("--file_out",default=False, type=bool)

def get_model(model_name):
    if model_name == 'VGG_3D':
        return VGG16_3D(num_classes=2).to(device)
    elif model_name == 'ResNet_3D':
        return ResNet_3D(num_classes=2).to(device) 
    elif model_name == 'DenseNet_3D':
        return DenseNet_3D(num_classes=2).to(device)
    elif model_name == 'AlexNet_3D':
        return AlexNet_3D(num_classes=2).to(device)
    elif model_name == 'ViT3D':
        return ViT3D(img_size=128, num_classes=2,in_c=1, embed_dim=8*8*8,).to(device)
    elif model_name == 'SwinT3D':
        return SwinT3D(img_size=128, num_classes=2, in_c=1, embed_dim=96, out_c=96*2*2*2*2,).to(device)
    else:
        print("please input correct model name!")
        return None

def main():  
    args = parser.parse_args()
    MODE = args.mode
    lr = args.learning_rate 
    num_epochs = args.epochs
    batch_size = args.batch_size

    dig_labels = pd.read_csv(args.label_dir)
    seg_paths = os.listdir(args.image_dir)
    dig_labels = dig_labels['Diagnosis'].tolist()

    if seg_paths[0].split('.')[0][:5]=='Train':
        seg_paths = sorted(seg_paths, key=lambda x: int((x.split('.')[0])[5:]))
    elif seg_paths[0].split('.')[0][:4]=='Test':
        seg_paths = sorted(seg_paths, key=lambda x: int((x.split('.')[0])[4:]))
    else:
        return

    for i in range(len(seg_paths)):
        seg_paths[i] = os.path.join(args.image_dir, seg_paths[i])

    labels = dig_labels
    image_paths = seg_paths 
    print('N=',len(dig_labels))

    if (MODE == 'train'):
        if get_model(args.model_name):
            model = get_model(args.model_name)
            if args.print_net:
                summary(model, input_size=(1,128,128,128))
        else:
            return
        
        if args.augmentation == 'True':
            transform = tio.Compose([
                tio.RandomFlip(axes=(0, 1, 2), flip_probability=0.5),
                tio.RandomAffine(degrees=15), 
                tio.RandomNoise(mean=0, std=(0, 0.05)),
                tio.RandomBlur(std=(0.5, 1.5)),
                ])
            print('data augmentation: Flip, Affine, Noise and Blur')
        else:
            transform = None

        X_train, X_val, y_train, y_val = train_test_split(image_paths, labels, test_size=0.2)
        train_dataset = CTScanDataset(X_train, y_train, transform=transform) # ViT do not need augmentation
        val_dataset = CTScanDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size,  shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        print('Strat Training', args.model_name)
        print("epochs: {}, learning rate: {}".format(num_epochs, lr))

        for epoch in range(num_epochs):
            train(model, epoch, train_loader, lr, num_epochs)
            validation(model,val_loader)

        model_path = args.save_path
        torch.save(model, model_path)
        print(args.model_name, "training finished")

    if (MODE == 'eval'):

        model = torch.load(args.load_path, map_location=device)
        if args.print_net:
            summary(model, input_size=(1,128,128,128))

        test_dataset = CTScanDataset(seg_paths, dig_labels)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        y_true = []  # Ground truth labels
        y_pred = []  # Predicted labels
        y_probs = []  # Predicted probabilities

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)[:, 1]  
                predictions = torch.argmax(outputs, dim=1)  

                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_probs.extend(probabilities.cpu().numpy())

        print(model.__class__.__name__)

        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2f}")

        precision = precision_score(y_true, y_pred)
        print(f"Precision: {precision:.2f}")

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp)
        print(f"Specificity: {specificity:.2f}")

        sensitivity = tp / (tp + fn)
        print(f"Sensitivity: {sensitivity:.2f}")

        f1 = f1_score(y_true, y_pred)
        print(f"F1-Score: {f1:.2f}")

        auc = roc_auc_score(y_true, y_probs)
        print(f"AUC: {auc:.2f}")

        if args.file_out:
            files=[]
            for file in seg_paths:               
                files.append(file.split('/')[-1].split('.')[0])
        
            fpr, tpr, thresholds = roc_curve(y_true, y_probs)

            cutoff_index = np.argmax(tpr - fpr)
            cutoff = thresholds[cutoff_index]
            cutoff = [cutoff for _ in range(len(files))]

            if files[0][:5]=='Train':
                outFile = '%s_predTRAIN.csv' % model.__class__.__name__
            elif files[0][:4]=='Test':
                outFile = '%s_predTEST.csv' % model.__class__.__name__
            else:
                outFile = '%s_pred.csv' % model.__class__.__name__
                
            pd.DataFrame(
                {
                    'files': files,
                    'True_label': y_true,
                    'Pred_label': y_pred,
                    'Cutoff': cutoff,
                    'Probs_label': y_probs,
                }
            ).to_csv('./predCSVs/'+outFile, index=False)

if __name__ == "__main__":
    main()
    # python main.py --mode=train --save_path=./3dAlexNet.pth --image_dir=../trainSEG --label_dir=../trainlabel.csv --model_name=AlexNet_3D
    # python main.py --mode=eval --load_path=./3dDenseNet.pth --image_dir=../testSEG --label_dir=../testlabel.csv
```

