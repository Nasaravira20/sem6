import cv2
import numpy as np

image = cv2.imread('images/image.png')
mask = np.zeros(image.shape[:2], np.uint8)

bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)

rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)
cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
segmented_image = image * mask2[:, :, np.newaxis]

cv2.imwrite('segmented_image.jpg', segmented_image)