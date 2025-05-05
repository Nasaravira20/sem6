import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('left_image.jpg', cv2.IMREAD_GRAYSCALE)
imgR = cv2.imread('right_image.jpg', cv2.IMREAD_GRAYSCALE)

if imgL is None or imgR is None:
	raise FileNotFoundError("One or both image files could not be loaded. Please check the file paths.")

num_disparities = 16 * 5
block_size = 15
stereo = cv2.StereoBM_create(numDisparities=num_disparities, blockSize=block_size)

disparity = stereo.compute(imgL, imgR)

norm_type=cv2.NORM_MINMAX
disparity_normalized = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
disparity_normalized = np.uint8(disparity_normalized)
plt.figure(figsize=(10, 7))
plt.imshow(disparity_normalized, cmap='gray')
plt.title('Disparity Map')
plt.axis('off')
plt.show()
