import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import Image

# defaults
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def denormalize_resnet152(img, mean=IMAGENET_MEAN, std=IMAGENET_STD):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return img * std + mean


def plot_image_from_path(image_path):
    image = Image.open(image_path).convert("RGB")
    fig0 = plt.figure()
    ax0 = fig0.add_subplot()
    ax0.imshow(image)
    ax0.set_title(f"Sample image: {image_path.name}")
    ax0.axis('off')         
    plt.tight_layout()
    plt.show(block=False)


def plot_image_from_index(data_loader, example_index):
    img, label = data_loader.dataset[example_index]
    img = denormalize_resnet152(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    fig0 = plt.figure()
    ax0 = fig0.add_subplot()
    img_np = img.permute(1, 2, 0).clip(0, 1).cpu().numpy()
    ax0.imshow(img_np.squeeze())
    ax0.set_title(f"Sample image: {label}")
    ax0.axis('off')        
    plt.tight_layout()
    plt.show(block=False)


def plot_images_from_index(data_loader, similar_images_indices, ncols=5):
    if len(similar_images_indices) == 0:
        return

    ncols = int(ncols)
    nrows = max(1, math.ceil(len(similar_images_indices) / ncols))
    figsize = (ncols * 2.0, max(2.5, nrows * 2.5))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    axes = np.array(axes).reshape(-1)

    for ax in axes:
        ax.axis('off')

    for i, idx in enumerate(similar_images_indices):
        if i >= len(axes):
            break
        ax = axes[i]
        img, label = data_loader.dataset[idx]
        img = denormalize_resnet152(img, mean=IMAGENET_MEAN, std=IMAGENET_STD)
        img_np = img.permute(1, 2, 0).clip(0, 1).cpu().numpy()
        ax.imshow(img_np.squeeze())
        ax.set_title(f"{label}", fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    plt.show(block=False)
