#!/usr/bin/python3.10
import os
import cv2
from pandas import read_csv
from tqdm import tqdm

LESION_TYPE = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma"
}


class HAM10000Dataset():
    def __init__(self, base_dir="./data", image_folder="img", metadata_folder="metadata", length=10):
        self.base_dir = base_dir
        self.image_folder = image_folder
        self.metadata_folder = metadata_folder
        self.file_list = os.listdir(f"{base_dir}/{image_folder}")
        self.metadata = read_csv(f"{base_dir}/{metadata_folder}/HAM10000_metadata")
        self.length = length
        self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load(self):
        print("Loading dataset...")
        for file in tqdm(self.file_list[0:1000]):
            image = cv2.imread(f"{self.base_dir}/{self.image_folder}/{file}")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            metadata = self.metadata[self.metadata["image_id"] == file.split(".")[0]]
            item = {
                "image": image,
                "image_id": file.split(".")[0],
                "file_name": file,
                "lesion_type": self.get_lesion_type(metadata["dx"].values[0]),
                "metadata": metadata.to_dict("records")[0]
            }
            self.data.append(item)

    def get_lesion_type(self, lesion_type):
        return LESION_TYPE[lesion_type]


if __name__ == "__main__":
    dataset = HAM10000Dataset()
    dataset.load()
    print(f"Image_shape: {dataset[0]['image'].shape}")
    print(dataset[0])
