#!/usr/bin/python3.10

import os
from cv2 import imread
import pandas as pd
from tqdm import tqdm
from utils import compare

results = []
segmentations = []

if __name__ == '__main__':
    result_f = os.listdir('results')
    for f in tqdm(result_f):
        img = imread('results/' + f)
        image_id = f.split('_mask')[0]
        segmentation = imread('data/segmentations/' + image_id + '_segmentation.png')
        results.append({
            'name': f,
            'score': compare(img, segmentation)
        })

    mean = sum([result['score'] for result in results]) / len(results)
    print('Mean score:', mean)
    df = pd.DataFrame(results)
    df.to_csv('scores.csv', index=False, sep=';')
