import cv2
import numpy as np

image_path = 'images/image.png'
image = cv2.imread(image_path)

cv2.imshow('Original Image', image)

cropped_image = image[50:200, 100:300]
cv2.imshow('Cropped Image', cropped_image)

resized_image = cv2.resize(image, (300, 300))
cv2.imshow('Resized Image', resized_image)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

_, thresholded_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('Thresholded Image', thresholded_image)

contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_image = image.copy()
cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
cv2.imshow('Contours', contour_image)

params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 100
blob_detector = cv2.SimpleBlobDetector_create(params)
keypoints = blob_detector.detect(gray_image)
blob_image = cv2.drawKeypoints(image, keypoints, np.array([]), (0, 0, 255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow('Blob Detection', blob_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
