import cv2
import numpy as np

image = np.zeros((500, 500, 3), dtype="uint8")

cv2.line(image, (50, 50), (450, 50), (255, 0, 0), 2)
cv2.rectangle(image, (50, 100), (450, 200), (0, 255, 0), 2)
cv2.circle(image, (250, 350), 50, (0, 0, 255), 2)
cv2.ellipse(image, (250, 350), (100, 50), 0, 0, 180, (255, 255, 0), 2)
cv2.putText(image, "Image Annotation", (100, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

cv2.imshow("Annotated Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()