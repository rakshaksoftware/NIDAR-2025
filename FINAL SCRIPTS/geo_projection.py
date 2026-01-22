import numpy as np

R_earth = 6378137.0

# ---------------- ROTATION ----------------
def rpy_to_R_body_to_NED(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    R_roll = np.array([
        [1, 0, 0],
        [0, cr, -sr],
        [0, sr, cr]
    ])

    R_pitch = np.array([
        [cp, 0, sp],
        [0, 1, 0],
        [-sp, 0, cp]
    ])

    R_yaw = np.array([
        [cy, -sy, 0],
        [sy, cy, 0],
        [0, 0, 1]
    ])

    return R_yaw @ R_pitch @ R_roll

def pixel_to_camera_ray(u, v, fx, fy, cx, cy):
    x = (u - cx) / fx
    y = (v - cy) / fy
    d = np.array([x, y, 1.0], dtype=np.float32)
    return d / np.linalg.norm(d)

def ned_offset_to_latlon(lat0_deg, lon0_deg, north, east):
    lat0 = np.deg2rad(lat0_deg)
    lon0 = np.deg2rad(lon0_deg)

    dlat = north / R_earth
    dlon = east / (R_earth * np.cos(lat0))

    lat = lat0 + dlat
    lon = lon0 + dlon

    return np.rad2deg(lat), np.rad2deg(lon)

def geolocate_target_from_pixel(u, v, K, R_cam_to_body, telem):
    fx, fy, cx, cy = K

    # 1) pixel -> cam ray
    d_cam = pixel_to_camera_ray(u, v, fx, fy, cx, cy)

    # 2) cam -> body
    d_body = R_cam_to_body @ d_cam

    # 3) body -> NED
    R_b2n = rpy_to_R_body_to_NED(telem["roll"], telem["pitch"], telem["yaw"])
    d_ned = R_b2n @ d_body
    d_ned = d_ned / np.linalg.norm(d_ned)
    dN, dE, dD = d_ned

    # must be pointing down
    if dD <= 0 or telem["h_agl"] is None:
        return None

    # 4) intersect with ground plane (D = h_agl)
    t = telem["h_agl"] / dD
    N = t * dN
    E = t * dE

    # 5) N,E offset + drone geodetic -> target geodetic
    if telem["lat"] is None or telem["lon"] is None:
        return None

    lat_t, lon_t = ned_offset_to_latlon(telem["lat"], telem["lon"], N, E)
    return lat_t, lon_t
