# SAM
SAM is a deep learning model for image segmentation
It may only return the class without making any categorization

## Fine-tune SAM using custom data
1. Load de images. Advice: use tifffile instead of opencv to open large number of images at once.
2. Divide the image in patches, try to make it by a multiplier of the image length

