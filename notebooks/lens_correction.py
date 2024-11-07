import numpy as np
import cv2

src = cv2.imread("/media/galiold/Storage/unprocessed_videos/frames-fhd/17/17_img00010.png")
width = src.shape[1]
height = src.shape[0]

distCoeff = np.zeros((4, 1), np.float64)

k1 = -1.5e-5;  # negative to remove barrel distortion
k2 = 0;
p1 = 0.0;
p2 = 0.0;

distCoeff[0, 0] = k1;
distCoeff[1, 0] = k2;
distCoeff[2, 0] = p1;
distCoeff[3, 0] = p2;

# assume unit matrix for camera
cam = np.eye(3, dtype=np.float32)

cam[0, 2] = width / 2.0  # define center x
cam[1, 2] = height / 2.0  # define center y
cam[0, 0] = 10.  # define focal length x
cam[1, 1] = 10.  # define focal length y

# here the undistortion will be computed
dst = cv2.undistort(src, cam, distCoeff)
cv2.imwrite('17_img00010_corrected.png', dst)
# cv2.imshow('dst', dst)
# while True:
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()