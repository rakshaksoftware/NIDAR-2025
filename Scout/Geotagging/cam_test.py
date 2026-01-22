import numpy as np
import cv2
import os

# ==========================================
# CONFIGURATION
# ==========================================
CHESSBOARD_SIZE = (8, 5)

CAM_WIDTH = 1280
CAM_HEIGHT = 720
FLIP_METHOD = 0

SAVE_DIR = "calib_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# ==========================================
# GSTREAMER PIPELINE
# ==========================================
def get_gstreamer_pipeline(width, height, framerate=30):
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width={width}, height={height}, "
        f"format=NV12, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={FLIP_METHOD} ! "
        f"video/x-raw, format=BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! appsink"
    )

# ==========================================
# PREPARE OBJECT POINTS
# ==========================================
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0],
                       0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

objpoints = []
imgpoints = []

# ==========================================
# START CAMERA
# ==========================================
cap = cv2.VideoCapture(
    get_gstreamer_pipeline(CAM_WIDTH, CAM_HEIGHT),
    cv2.CAP_GSTREAMER
)

if not cap.isOpened():
    print("Error: Camera not opened")
    exit()

print("\n--- CAMERA CALIBRATION ---")
print("Press 's' to save a frame")
print("Press 'c' to calibrate")
print("Press 'q' to quit\n")

count = 0

# ==========================================
# MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()

    cv2.putText(display, f"Saved Images: {count}",
                (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 255), 2)

    cv2.imshow("Calibration", display)
    key = cv2.waitKey(1) & 0xFF

    # --------------------------------------
    # SAVE FRAME (ONLY DETECT WHEN PRESSED)
    # --------------------------------------
    if key == ord('s'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE)

        if found:
            objpoints.append(objp)
            imgpoints.append(corners)

            filename = f"{SAVE_DIR}/img_{count:02d}.png"
            cv2.imwrite(filename, frame)

            count += 1
            print(f"Saved: {filename}")

            cv2.drawChessboardCorners(display, CHESSBOARD_SIZE, corners, found)
            cv2.imshow("Calibration", display)
            cv2.waitKey(200)
        else:
            print("Chessboard NOT detected")

    # --------------------------------------
    # CALIBRATE
    # --------------------------------------
    elif key == ord('c'):
        if count < 10:
            print("Need at least 10 images")
            continue

        print("Calibrating...")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )

        print("\n" + "=" * 40)
        print("CALIBRATION COMPLETE")
        print(f"RMS Error: {ret:.4f}")
        print("\nCamera Matrix:\n", mtx)
        print("\nDistortion Coefficients:\n", dist)
        print("=" * 40)
        break

    # --------------------------------------
    # QUIT
    # --------------------------------------
    elif key == ord('q'):
        break

# ==========================================
# CLEANUP
# ==========================================
cap.release()
cv2.destroyAllWindows()
