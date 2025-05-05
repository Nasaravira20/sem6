import numpy as np
import cv2 as cv
import glob

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
obj3d = np.zeros((44, 3), np.float32)
a = [0, 36, 72, 108, 144, 180, 216, 252, 288, 324, 360]
b = [0, 72, 144, 216, 36, 108, 180, 252]
for i in range(44):
    obj3d[i] = (a[i // 4], b[i % 8], 0)

obj_points = []
img_points = []
images = glob.glob('images/image copy 3.png')
print(f"Found images: {images}")

if not images:
    print("No images found in the specified path.")
    exit()

for f in images:
    img = cv.imread(f)
    if img is None:
        print(f"Failed to load image: {f}")
        continue
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('Input Image', gray)
    cv.waitKey(0)
    cv.destroyAllWindows()

    ret, corners = cv.findCirclesGrid(gray, (4, 11), None, flags=cv.CALIB_CB_ASYMMETRIC_GRID)
    if ret:
        print(f"Circles grid detected in image: {f}")
        obj_points.append(obj3d)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        cv.drawChessboardCorners(img, (4, 11), corners2, ret)
        cv.imshow('Detected Grid', img)
        cv.waitKey(0)
        cv.destroyAllWindows()
    else:
        print(f"Circles grid NOT detected in image: {f}")

if not obj_points or not img_points:
    print("No valid circle grids detected. Exiting.")
    exit()

ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("Error in projection: \n", ret)
print("\nCamera matrix: \n", camera_mat)
print("\nDistortion coefficients: \n", distortion)
print("\nRotation vectors: \n", rotation_vecs)
print("\nTranslation vectors: \n", translation_vecs)