# Background

We can get the coordinates of our drone using GPS and have image from camera. After identifying human we now need a faster way to get the exact coordinates of the human.

Given a drone flying at a known height with a camera capturing images, we need to determine the real-world coordinates (X, Y, Z) of a point on the ground when we know its pixel coordinates (u, v) in the image.

## Required Parameters

- **Camera intrinsics:** Focal length, principal point, sensor dimensions
- **Camera extrinsics:** Drone position, camera orientation
- **Flight parameters:** Drone height above ground level
- **Image coordinates:** Pixel location (u, v) of the point of interest

## Analysis

### Step 1: Camera Model Understanding

We need to start with the pinhole camera model which relates 3D world points to 2D image points:

$$
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = K [R|t] \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix}

$$

Where:

- K is the camera intrinsic matrix containing focal length and principal point
- [R|t] represents the extrinsic parameters (rotation and translation)
- (X, Y, Z) are the world coordinates we seek
- (u, v) are the image pixel coordinates we know

### Step 2: Inverse Projection

Since the drone height is known, we can use it to solve the under constrained system:

$$
\begin{bmatrix} X \\ Y \\ Z \end{bmatrix} = R^{-1} \left( K^{-1} \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \cdot s - t \right)

$$

Where s is a scaling factor determined by the known height of the drone above ground.

### Step 3: Ground Plane Constraint

Assuming the point lies on the ground plane (Z = 0 in world coordinates), we can simplify the equations further.

# References

1. Wikipedia
    - Perspective n point - https://en.wikipedia.org/wiki/Perspective-n-Point
    - WGS - https://en.wikipedia.org/wiki/World_Geodetic_System
    - ENU coordinate system - https://en.wikipedia.org/wiki/Local_tangent_plane_coordinates
    - ECF coordinates system - https://en.wikipedia.org/wiki/Earth-centered,_Earth-fixed_coordinate_system
    - Geodetic coordinates - https://en.wikipedia.org/wiki/Geodetic_coordinates
2. Open CV documentations 
    - OpenCV Camera Calibration Docs : https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
    - Brown Conrady Camera Model (Distortion) :https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
    

# Implementation

### Step 0 — Setup, conventions, and required inputs

### Coordinate Systems

- **Camera frame** (Xc, Yc, Zc): +Xc = right, +Yc = down, +Zc = forward (out of image plane)
- **World frame** = local ENU: (E, N, U) = East, North, Up
- **Drone geodetic position** (φ0, λ0, h0) = (lat, lon, alt)

### Inputs required

**Pixel coordinates:** u, v (pixels) — origin top-left, u→right, v→down.

**Image size:** width W, height H (pixels).

**Camera intrinsics:**

- fx, fy (focal lengths in pixels) or horizontal/vertical FOVs.
- Principal point cx, cy (pixels).
- Skew s

**Camera mounting:** rotation from camera to body Rc→b (3×3) or equivalent yaw/pitch/roll (mount).

**Drone state:**

- Geodetic position (φ0, λ0, h0) — latitude, longitude (deg), altitude (m).
- Attitude (from IMU): roll φ, pitch θ, yaw ψ. Using Tait–Bryan Z–Y–X order.
- Ground altitude hg at the location where the ray intersects.

### Step 1 — Pixel → camera ray (intrinsic inversion)

What it does: Converts a pixel coordinate (u,v) into a 3D direction in the camera coordinate system.

### Equations

1. Form homogeneous pixel vector:

$$
u_{\sim} = \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
$$

1. Compute camera-ray by inverting intrinsics:

$$
r_c = K^{-1}u_{\sim}
$$

1. If s = 0 this expands to:

$$
x_c = \frac{u - c_x}{f_x}, \quad y_c = \frac{v - c_y}{f_y}, \quad r_c = \begin{bmatrix} x_c \\ y_c \\ 1 \end{bmatrix}
$$

1. (Optional) Normalize direction:

$$
\hat{r}_c = \frac{r_c}{\|r_c\|}
$$

### Step 2 — Camera frame → Body (drone) frame (mount transform)

What it does: Transforms the ray direction from camera coordinates to the drone's body coordinates.

### Equation

$$
r_b = R_{c \rightarrow b} \, r_c
$$

### Step 3 — Body (drone) → World (ENU) frame (attitude transform)

What it does: Rotate the ray from the drone's body coordinates into the world frame (local ENU).

### Rotation matrices

$$
R_{x}(\phi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix}
$$

$$
R_{y}(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}
$$

$$
R_{z}(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}
$$

Combined rotation (Z–Y–X convention):

$$
R_{b \rightarrow w} = R_{z}(\psi) \, R_{y}(\theta) \, R_{x}(\phi)
$$

### Equation

$$
r_w = R_{b \rightarrow w} \, r_b
$$

### Step 4 — Parametric ray in world coordinates (line equation)

What it does: Writes a parametric equation for the 3D ray starting at the camera location.

### Parametric equation

$$
p(t) = p_d + t \, r_w, \quad t \in \mathbb{R}
$$

Where pd = [Ed, Nd, Ud]T is the drone's position in world frame.

### Step 5 — Ray ↔ Ground intersection (solve for t)

What it does: Finds where the ray hits the ground plane or terrain.

### Equation (flat-plane case)

$$
U_d + t \, r_U = h_g \implies t = \frac{h_g - U_d}{r_U}
$$

Intersection point:

$$
p_{hit} = p_d + t \, r_w = \begin{bmatrix} E_d + t r_E \\ N_d + t r_N \\ h_g \end{bmatrix}
$$

### Step 6 — ENU hit point → Geodetic (lat, lon, alt)

What it does: Converts the intersection coordinates (ENU) into geodetic latitude, longitude, and altitude.

### ENU → ECEF → Geodetic

1. Convert ENU origin to ECEF:

$$
N(\phi_0) = \frac{a}{\sqrt{1 - e^2 \sin^2 \phi_0}}
$$

$$
X_0 = (N(\phi_0) + h_0) \cos\phi_0 \cos\lambda_0
$$

$$
Y_0 = (N(\phi_0) + h_0) \cos\phi_0 \sin\lambda_0
$$

$$
Z_0 = (N(\phi_0)(1 - e^2) + h_0) \sin\phi_0
$$

1. Convert ENU to ECEF using rotation matrix:

$$
R_{e \rightarrow enu} = \begin{bmatrix} -\sin\lambda_0 & \cos\lambda_0 & 0 \\ -\sin\phi_0\cos\lambda_0 & -\sin\phi_0\sin\lambda_0 & \cos\phi_0 \\ \cos\phi_0\cos\lambda_0 & \cos\phi_0\sin\lambda_0 & \sin\phi_0 \end{bmatrix}
$$

$$
X = X_0 + R_{e \rightarrow enu}^{\top} \, p_{enu}
$$

1. Convert ECEF to geodetic coordinates using standard algorithms.

# Code

Below is the code to implement the above concepts also incorporating the values of roll, pitch and yaw.

```python
import numpy as np
from dataclasses import dataclass

@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    s: float = 0.0

@dataclass
class Angles:
    roll: float   # ϕ (rad) (x)
    pitch: float  # θ (rad) (y)
    yaw: float    # ψ (rad) (z)

def build_K(intr: Intrinsics) -> np.ndarray:
    K = np.array([
        [intr.fx, intr.s,   intr.cx],
        [0.0,     intr.fy,  intr.cy],
        [0.0,     0.0,      1.0    ]
    ], dtype=float)
    return K

def inv_K(K: np.ndarray) -> np.ndarray:
    return np.linalg.inv(K)

def fov_to_focal_pixels(W: int, H: int, fov_x, fov_y):
    fx = W / (2.0 * np.tan(fov_x * 0.5))
    fy = H / (2.0 * np.tan(fov_y * 0.5))

    return float(fx), float(fy)

W, H = 1600, 1200
fov_deg = 84
d = (W**2 + H**2)**0.5
f = (d/2)/np.tan(np.deg2rad(fov_deg)/2)
fov_x = 2*np.arctan(W/(2*f))
fov_y = 2*np.arctan(H/(2*f))
fx, fy = fov_to_focal_pixels(W, H, fov_x, fov_y)
intr = Intrinsics(fx, fy, cx=W/2, cy=H/2, s=0)
cam_angles = Angles(roll=0, pitch=0, yaw=0)

u, v = 1250, 340
height = 96

def Rx(phi: float) -> np.ndarray:
    c, s = np.cos(phi), np.sin(phi)
    return np.array([[1, 0, 0],
                     [0, c,-s],
                     [0, s, c]], dtype=float)

def Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[ c, 0,-s],
                     [ 0, 1, 0],
                     [ s, 0, c]], dtype=float)

def Rz(psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return np.array([[ c,-s, 0],
                     [ s, c, 0],
                     [ 0, 0, 1]], dtype=float)

def RotationMatrix(cam_angles) -> np.ndarray:
    return Rz(-cam_angles.yaw) @ Ry(-cam_angles.pitch) @ Rx(-cam_angles.roll)

print(RotationMatrix(cam_angles)@inv_K(build_K(intr))@np.array([u, v, 1])*height)
```

# Testing

Since we didn’t have the access to the camera to be used in the drone for testing. We used our phone to test the above code. Keeping the roll, pitch and yaw zero gave us perfect results. (Testing a point with known coordinates on a grid with respect to an assumed reference) Now, we wanted to test for rotation of camera i.e. non-zero roll, pitch and yaw. We tried to use the gyroscope inbuilt in the phone for the values. However, there was a large difference in values for the roll, pitch and yaw for images captured by keeping the phone orientation similar.
Images were captured using the *Open Camera* application on an Android smartphone. The captured files were in **JPEG format** by default. To examine the embedded metadata (EXIF information such as timestamp, GPS coordinates, and camera parameters), the images were uploaded to **Metadata2Go**, an online metadata extraction tool where we got the roll, pitch and yaw.

Sources:

- Metadata extractor - MetaData2Go [Metadata2go.com](https://www.metadata2go.com/)
- Open Camera app - [Open Camera - Apps on Google Play](https://play.google.com/store/apps/details?id=net.sourceforge.opencamera&hl=en&pli=1)

# Depth Camera

We also tried to use Intel-RealSense 435i depth camera to help us in geotagging. The camera has an inbuilt accelerometer and, gyroscope. We were able to extract the accelerations and roll, pitch and yaw angular velocities using the below script. the camera was successfully detecting the depth of objects till 3.5-4 metres. However, it is highly unlikely that we will use this camera because we need to measure depth of around 8m which we are not able to detect till now.

Links-

- Depth Camera Used: [Intel® RealSense™ Depth Camera D435i](https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html)
- Library to install RealSense Viewer and required dependencies to run the camera: https://github.com/IntelRealSense/librealsense

```python
import pyrealsense2 as rs
import time

# Configure pipeline
pipeline = rs.pipeline()
config = rs.config()

# Enable gyro + accelerometer streams
config.enable_stream(rs.stream.gyro)
config.enable_stream(rs.stream.accel)

# Start pipeline
pipeline.start(config)

try:
    while True:
        # Wait for the next set of frames
        frames = pipeline.wait_for_frames()

        # Gyro frame
        if frames.first_or_default(rs.stream.gyro):
            gyro = frames.first_or_default(rs.stream.gyro).as_motion_frame().get_motion_data()
            print(f"Gyro: x={gyro.x:.4f}, y={gyro.y:.4f}, z={gyro.z:.4f}")

        # Accel frame
        if frames.first_or_default(rs.stream.accel):
            accel = frames.first_or_default(rs.stream.accel).as_motion_frame().get_motion_data()
            print(f"Accel: x={accel.x:.4f}, y={accel.y:.4f}, z={accel.z:.4f}")

        time.sleep(0.05)  # limit print speed a bit

except KeyboardInterrupt:
    print("Stopped streaming")

finally:
    pipeline.stop()

```

# To Do

Require images of so that we can test our code and make the required changes. Can do this by simulating a virtual drone environment and capturing images from there or using the actual camera to be used in the competition whenever it is functional.
