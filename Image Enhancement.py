import cv2
import numpy as np
from matplotlib import pyplot as plt

image = cv2.imread('images/image.png')

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

hist_eq_image = cv2.equalizeHist(gray_image)

kernel = np.ones((5, 5), np.float32) / 25
smoothed_image = cv2.filter2D(image, -1, kernel)

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(sobelx, sobely)

edges = cv2.Canny(gray_image, 100, 200)

plt.subplot(231), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original')
plt.subplot(232), plt.imshow(gray_image, cmap='gray'), plt.title('Grayscale')
plt.subplot(233), plt.imshow(hist_eq_image, cmap='gray'), plt.title('Histogram Equalized')
plt.subplot(234), plt.imshow(cv2.cvtColor(smoothed_image, cv2.COLOR_BGR2RGB)), plt.title('Smoothed')
plt.subplot(235), plt.imshow(gradient_magnitude, cmap='gray'), plt.title('Gradient Magnitude')
plt.subplot(236), plt.imshow(edges, cmap='gray'), plt.title('Edge Detection')
plt.show()