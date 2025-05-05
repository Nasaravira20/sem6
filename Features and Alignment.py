import cv2
import numpy as np

def extract_orb_features(image):
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(descriptors1, descriptors2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors1, descriptors2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def align_images(image1, image2):
    keypoints1, descriptors1 = extract_orb_features(image1)
    keypoints2, descriptors2 = extract_orb_features(image2)
    matches = match_features(descriptors1, descriptors2)
    points1 = np.float32([keypoints1[m.queryIdx].pt for m in matches])
    points2 = np.float32([keypoints2[m.trainIdx].pt for m in matches])
    matrix, _ = cv2.findHomography(points1, points2, cv2.RANSAC, 5.0)
    aligned_image = cv2.warpPerspective(image1, matrix, (image2.shape[1], image2.shape[0]))
    return aligned_image

image1 = cv2.imread('images/image copy.png', cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread('images/image copy 2.png', cv2.IMREAD_GRAYSCALE)
aligned_image = align_images(image1, image2)
cv2.imwrite('aligned_image.jpg', aligned_image)