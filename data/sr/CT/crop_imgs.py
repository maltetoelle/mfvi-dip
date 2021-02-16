import numpy as np
import cv2

for i in range(2):
    img = cv2.imread(f"ct{i}.png")

    new_w = 384




    # size = np.min(np.array(img.shape[:2]))
    #
    # center = np.array(img.shape[:2]) / 2
    # x = center[1] - size/2
    # y = center[0] - size/2

    fy = img.shape[1] / new_w
    newsize = img.shape[0] * (1/fy)
    newsize = int(newsize - (newsize % 32))
    oldsize = int(fy * newsize)

    print(newsize)
    print(oldsize)

    cr_w = int(0.5 * (img.shape[1] - oldsize))
    crop_img = img[:oldsize, :]
    #crop_img = img[int(y):int(y+size), int(x):int(x+size)]
    crop_img = cv2.resize(crop_img, (new_w, newsize))
    # crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(f"ct{i}_res.png", crop_img)
    cv2.imshow("img", crop_img)
    cv2.waitKey(0)
