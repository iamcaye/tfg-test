#!/usr/bin/python3.11

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random as rand
import cv2

fig = plt.figure(figsize=(10, 7))

imgs = os.listdir('./data/img')

n = rand.randint(0, len(imgs) - 1)

img = mpimg.imread(f'./data/img/{imgs[n]}', )
segm = mpimg.imread(f"./data/segmentations/{imgs[n].split('.')[0]}_segmentation.png")

fig.add_subplot(1, 2, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Img')

fig.add_subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(segm, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('segm')


plt.show()
