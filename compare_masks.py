import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("midl/data/inpainting/original_hair.png")
mask = cv2.imread("midl/data/inpainting/original_hair_mask.png")

# mask = np.invert(mask)
mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(mask, 127 , 255, cv2.THRESH_BINARY)

hist = np.histogram(mask, bins=256)

cv2.imwrite('original_hair_mask.png', mask)
import pdb; pdb.set_trace()
# mask[mask[:,:,0] == 255] = 0
# mask[mask[:,:,1] == 255] = 0
# plt.hist(mask, bins=256)
# plt.show()

# cv2.imshow("mask", mask)
# cv2.waitKey(0)
