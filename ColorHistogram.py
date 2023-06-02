import cv2 as cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from array import array

nb_bins = 256
count_r = np.zeros(nb_bins)
count_g = np.zeros(nb_bins)
count_b = np.zeros(nb_bins)

root = './images/yes_aurora/'
for image in os.listdir(root):
	if image.endswith('.png'):
		x = cv2.imread(root+image, cv2.IMREAD_COLOR)
		norm_image = cv2.normalize(x, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		hist_r = np.histogram(norm_image[0], bins=nb_bins, range=[0, 1])
		hist_g = np.histogram(norm_image[1], bins=nb_bins, range=[0, 1])
		hist_b = np.histogram(norm_image[2], bins=nb_bins, range=[0, 1])
		count_r += hist_r[0]
		count_g += hist_g[0]
		count_b += hist_b[0]
		norm_image.tofile('histogram.txt', sep=' ')
		
		f = open('histogram.txt', 'a')
		f.seek(0)
		f.write(", 1")
		f.seek(769)
		f.write(image)
		f.close()

bins = hist_r[1]
fig = plt.figure()
plt.bar(bins[:-1], count_r, color='r', alpha=0.33)
plt.bar(bins[:-1], count_g, color='g', alpha=0.33)
plt.bar(bins[:-1], count_b, color='b', alpha=0.33)

plt.show()