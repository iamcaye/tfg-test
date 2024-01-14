#!/usr/bin/python3.10
import sys
from cv2 import imread
from utils.utils import compare
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: compare.py <image_id>')
        exit(1)

    image_id = sys.argv[1]
    result_f = 'results/' + image_id + '_mask.png'
    segmentation_f = 'data/segmentations/' + image_id + '_segmentation.png'
    img_f = 'data/img/' + image_id + '.jpg'

    result = imread(result_f)
    segmentation = imread(segmentation_f)
    img = imread(img_f)
    score = compare(result, segmentation)

    fig = plt.figure(figsize=(10, 10))

    fig.add_subplot(1, 3, 1)
    plt.imshow(img)
    plt.title('Image')

    fig.add_subplot(1, 3, 2)
    plt.imshow(segmentation)
    plt.title('Validation')

    fig.add_subplot(1, 3, 3)
    plt.imshow(result)
    plt.title('Result')

    fig.text(0.5, 0.01, 'Score: ' + str(score), ha='center', fontsize=18)
    plt.show()
