# n-res.py
import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor
from torch.utils.data import Dataset
from PIL import Image
import os

class Resnet50FeatureExtractor(nn.Module):
    def __init__(self, device, return_layers=None):
        super(Resnet50FeatureExtractor, self).__init__()

        if return_layers is None:
            return_layers = {
                'layer1': 'layer1',
                'layer2': 'layer2',
                'layer3': 'layer3',
                'layer4': 'layer4'
            }
        self.return_layers = return_layers

        model = resnet50(pretrained=True).to(device=device)
        model.train(mode=False)
        self.feature_extractor = create_feature_extractor(model, return_nodes=list(return_layers.keys())).to(device=device)

    def forward(self, x):
        x = self.feature_extractor(x)
        return x

class CustomImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
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