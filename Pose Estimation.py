import cv2
import numpy as np
import os
import urllib.request

# URLs for the model files
proto_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/pose_deploy_linevec.prototxt"
weights_url = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"

# Paths to save the model files
proto_file = "pose_deploy_linevec.prototxt"
weights_file = "pose_iter_440000.caffemodel"

# Download the files if they don't exist
if not os.path.exists(proto_file):
    print(f"Downloading {proto_file}...")
    urllib.request.urlretrieve(proto_url, proto_file)

if not os.path.exists(weights_file):
    print(f"Downloading {weights_file}...")
    urllib.request.urlretrieve(weights_url, weights_file)

# Load OpenPose pre-trained model
net = cv2.dnn.readNetFromCaffe(proto_file, weights_file)

# Specify the number of points and pairs for pose estimation
n_points = 18
pose_pairs = [
    (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)
]

# Read the input image
image_path = "images/image.png"  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    raise Exception("Failed to load image. Check the image path.")

frame_width = frame.shape[1]
frame_height = frame.shape[0]
in_width = 368
in_height = 368
inp_blob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (in_width, in_height),
                                 (0, 0, 0), swapRB=False, crop=False)
net.setInput(inp_blob)
output = net.forward()

h, w = output.shape[2], output.shape[3]
points = []

for i in range(n_points):
    prob_map = output[0, i, :, :]
    min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

    x = int((frame_width * point[0]) / w)
    y = int((frame_height * point[1]) / h)

    if prob > 0.1:  # Threshold
        points.append((x, y))
        cv2.circle(frame, (x, y), 5, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
    else:
        points.append(None)

for pair in pose_pairs:
    part_a = pair[0]
    part_b = pair[1]

    if points[part_a] and points[part_b]:
        cv2.line(frame, points[part_a], points[part_b], (0, 255, 0), 2)
        cv2.circle(frame, points[part_a], 5, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

cv2.imshow('Pose Estimation', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
