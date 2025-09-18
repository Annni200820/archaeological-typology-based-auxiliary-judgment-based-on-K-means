import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torch.utils.data import Dataset 
from PIL import Image, ImageFile
import os
from keras.layers import *
from keras.models import Model
from keras import backend as K
import imageio
import tensorflow as tf
tf.config.list_physical_devices('GPU')
#
# class FeatureExtractor(nn.Module):
#     def __init__(self):
#         super(FeatureExtractor, self).__init__()
#
#         # simple convolution
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )

    # def forward(self, x: torch.tensor):
    #     x = self.conv(x)
    #     return x

class Resnet50FeatureExtractor(nn.Module):
    def __init__(
            self,   
            device
    ):
        super(Resnet50FeatureExtractor, self).__init__()

        model = resnet50(pretrained=True).to(device=device)
        model.train(mode=False)
        train_nodes, eval_nodes = get_graph_node_names(model)

        print('train_nodes')
        print(train_nodes)
        print('eval_nodes')
        print(eval_nodes)

        return_nodes = {
            'layer4.2.relu_2': 'layer4',
        }
        self.feature_extractor = create_feature_extractor(model, return_nodes=return_nodes).to(device=device)   

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

# image dataset preprocess
class CustomImageDataset(Dataset):
    def __init__(
            self,
            image_dir,
            transform=None,
    ):
        super(CustomImageDataset, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = os.listdir(image_dir)      

    def __len__(self):
        return len(self.images) 
    
    def __getitem__(self, item):
        image_path = os.path.join(self.image_dir, self.images[item])
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        data = {
            'image': image,
            'path': image_path
        }

        return data