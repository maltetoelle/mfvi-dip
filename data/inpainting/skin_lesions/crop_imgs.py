import numpy as np
import cv2

for i in range(6):
    img = cv2.imread(f"hair_{i}_mask.png")

    size = np.min(np.array(img.shape[:2]))
    center = np.array(img.shape[:2]) / 2
    x = center[1] - size/2
    y = center[0] - size/2

    crop_img = img[int(y):int(y+size), int(x):int(x+size)]
    crop_img = cv2.resize(crop_img, (256, 256))
    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"hair_{i}_res_mask.png", crop_img)
    cv2.imshow("img", crop_img)
    cv2.waitKey(0)
