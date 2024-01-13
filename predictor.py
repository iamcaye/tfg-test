#!/usr/bin/python3.10
from dataset import HAM10000Dataset
from segment_anything import sam_model_registry, SamPredictor
import torch
import matplotlib.pyplot as plt
from utils.sam_utils import show_mask
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

sam_checkpoint = "weights/sam_vit_h_4b8939.pth"
model_type = "vit_h"

if __name__ == "__main__":
    dataset = HAM10000Dataset()
    dataset.load()
    print(f"Device: {device}")
    print(f"Image_shape: {dataset[0]['image'].shape}")
    px = dataset[0]['image'].shape[0] // 2
    py = dataset[0]['image'].shape[1] // 2

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    for item in dataset:
        predictor.set_image(item["image"])
        masks, _, _ = predictor.predict(
            point_coords=np.array([[px, py]]),
            point_labels=np.array([1]),
            multimask_output=True,
        )

        fig = plt.figure(figsize=(10, 10))
        fig.add_subplot(1, 2, 1)
        plt.imshow(item["image"])
        plt.title("Image")

        fig.add_subplot(1, 2, 2)
        plt.imshow(item["image"])
        for mask in masks:
            show_mask(mask, plt.gca(), random_color=True)
        plt.title("Predicted mask")
        fig.text(0.5, 0.91, f"Lesion type: {item['lesion_type']}", ha="center", fontsize=16)
        plt.show()


        input("Press Enter to continue...")
