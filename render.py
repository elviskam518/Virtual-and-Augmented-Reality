import numpy as np
import time
import matplotlib.pyplot as plt
import cv2
from engine import triangle, line, obj_load, lookat, perspective, viewport
import engine

def translate(tx, ty, tz):
    mat = np.eye(4, dtype=float)
    mat[0, 3] = tx
    mat[1, 3] = ty
    mat[2, 3] = tz
    return mat


def scale(sx, sy, sz):
    mat = np.eye(4, dtype=float)
    mat[0, 0] = sx
    mat[1, 1] = sy
    mat[2, 2] = sz
    return mat


def rotate_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    mat = np.eye(4, dtype=float)
    mat[1, 1] = c
    mat[1, 2] = -s
    mat[2, 1] = s
    mat[2, 2] = c
    return mat


def rotate_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    mat = np.eye(4, dtype=float)
    mat[0, 0] = c
    mat[0, 2] = s
    mat[2, 0] = -s
    mat[2, 2] = c
    return mat


def rotate_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    mat = np.eye(4, dtype=float)
    mat[0, 0] = c
    mat[0, 1] = -s
    mat[1, 0] = s
    mat[1, 1] = c
    return mat



def load_imu_data(filename):
    rows = []

    with open(filename, "r") as f:
        f.readline()  
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append([float(x) for x in line.split(",")])

    rows = np.array(rows)

    time_values = rows[:, 0]
    gyro_deg = rows[:, 1:4]
    accel = rows[:, 4:7]
    mag = rows[:, 7:10]

    gyro_rad = np.deg2rad(gyro_deg)

    return {
        "time": time_values,
        "gyro": gyro_rad,
        "accel": accel,
        "mag": mag
    }


def quat_normalize(q):
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def quat_conjugate(q):
    w, x, y, z = q
    return np.array([w, -x, -y, -z])


def quat_multiply(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def euler_to_quat(roll, pitch, yaw):
    cr = np.cos(roll / 2)
    sr = np.sin(roll / 2)
    cp = np.cos(pitch / 2)
    sp = np.sin(pitch / 2)
    cy = np.cos(yaw / 2)
    sy = np.sin(yaw / 2)

    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy
    ])


def quat_to_euler(q):
    w, x, y, z = q

    roll = np.arctan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))

    sin_pitch = 2 * (w*y - z*x)
    if abs(sin_pitch) >= 1:
        pitch = np.copysign(np.pi / 2, sin_pitch)
    else:
        pitch = np.arcsin(sin_pitch)

    yaw = np.arctan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))

    return roll, pitch, yaw

#https://www.songho.ca/opengl/gl_quaternion.html
def quat_to_rotation_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q

    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),         1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),         2*(y*z + w*x),     1 - 2*(x*x + y*y)]
    ], dtype=float)


def quat_to_matrix_4x4(q):
    rot = quat_to_rotation_matrix(q)
    mat = np.eye(4, dtype=float)
    mat[:3, :3] = rot
    return mat


def rotate_vector_by_quat(q, v):
    v_quat = np.array([0.0, v[0], v[1], v[2]])
    rotated = quat_multiply(quat_multiply(q, v_quat), quat_conjugate(q))
    return rotated[1:4]

def compute_tilt_error_series(orientations, accel_data, up=np.array([0.0, 0.0, 1.0])):
    n = len(accel_data)
    tilt = np.zeros(n)

    for i in range(n):
        ag = rotate_vector_by_quat(orientations[i], accel_data[i])

        norm_ag = np.linalg.norm(ag)
        if norm_ag < 1e-10:
            tilt[i] = np.nan
            continue

        ag = ag / norm_ag
        dot_val = np.clip(np.dot(ag, up), -1.0, 1.0)
        tilt[i] = np.degrees(np.arccos(dot_val))

    return tilt


def get_stats(arr):
    arr = arr[~np.isnan(arr)]
    return {
        "mean": np.mean(arr),
        "median": np.median(arr),
        "p95": np.percentile(arr, 95),
        "max": np.max(arr)
    }


def print_comparison_table(stats_gyro, stats_fused, alpha=0.02):
    print(f"{'Metric':<10}{'Gyro only':<15}{f'Gyro + Accel (α={alpha})':<22}{'Changed'}")

    for key, label in [("mean", "Mean"), ("median", "Median"), ("p95", "P95"), ("max", "Max")]:
        g = stats_gyro[key]
        f = stats_fused[key]
        reduction = (g - f) / g * 100
        print(f"{label:<10}{g:>6.2f}°{'':<8}{f:>6.2f}°{'':<10}{reduction:>4.0f}% ↓")
def axis_angle_to_quat(axis, angle):
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12 or abs(angle) < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])

    axis = axis / axis_norm
    half = angle / 2.0

    return np.array([
        np.cos(half),
        axis[0] * np.sin(half),
        axis[1] * np.sin(half),
        axis[2] * np.sin(half)
    ])
    
#3.1
def dead_reckoning(imu):
    n = len(imu["time"])
    result = np.zeros((n, 4))
    result[0] = [1, 0, 0, 0]

    q = np.array([1.0, 0.0, 0.0, 0.0])

    for i in range(1, n):
        dt = imu["time"][i] - imu["time"][i - 1]
        omega = imu["gyro"][i - 1]

        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            dq = axis_angle_to_quat(axis, angle)
            q = quat_normalize(quat_multiply(q, dq))

        result[i] = q

    return result

#3.2
def gyro_accel_complementary(imu, alpha=0.02):
    n = len(imu["time"])
    result = np.zeros((n, 4))
    result[0] = [1, 0, 0, 0]

    q = np.array([1.0, 0.0, 0.0, 0.0])
    global_up = np.array([0.0, 0.0, 1.0])

    for i in range(1, n):
        dt = imu["time"][i] - imu["time"][i - 1]
        omega = imu["gyro"][i - 1]
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            dq = axis_angle_to_quat(axis, angle)
            q = quat_normalize(quat_multiply(q, dq))

        accel_local = imu["accel"][i]
        accel_global = rotate_vector_by_quat(q, accel_local)

        accel_norm = np.linalg.norm(accel_global)
        if accel_norm < 1e-10:
            result[i] = q
            continue

        accel_dir = accel_global / accel_norm

        tilt_axis = np.cross(accel_dir, global_up)
        axis_norm = np.linalg.norm(tilt_axis)
        if axis_norm < 1e-8:
            result[i] = q
            continue

        tilt_axis = tilt_axis / axis_norm
        cos_angle = np.clip(np.dot(accel_dir, global_up), -1.0, 1.0)
        tilt_angle = np.arccos(cos_angle)

        correction = axis_angle_to_quat(tilt_axis, alpha * tilt_angle)
        q = quat_normalize(quat_multiply(correction, q))

        result[i] = q

    return result
#4.1
def get_heading(accel, mag):
    gravity = accel / max(np.linalg.norm(accel), 1e-10)

    east = np.cross(gravity, mag)
    east_norm = np.linalg.norm(east)
    if east_norm < 1e-10:
        return None
    east = east / east_norm

    north = np.cross(east, gravity)
    north_norm = np.linalg.norm(north)
    if north_norm < 1e-10:
        return None
    north = north / north_norm

    return np.arctan2(np.dot(mag, east), np.dot(mag, north))


def gyro_accel_mag_complementary(imu, alpha=0.02, alpha_mag=0.01):
    n = len(imu["time"])
    result = np.zeros((n, 4))
    result[0] = [1, 0, 0, 0]

    q = np.array([1.0, 0.0, 0.0, 0.0])
    global_up = np.array([0.0, 0.0, 1.0])
    mag_norms = np.linalg.norm(imu["mag"], axis=1)
    expected_mag_norm = np.median(mag_norms)

    heading_ref = get_heading(imu["accel"][0], imu["mag"][0])

    for i in range(1, n):
        dt = imu["time"][i] - imu["time"][i - 1]

        omega = imu["gyro"][i - 1]
        angle = np.linalg.norm(omega) * dt
        if angle > 1e-10:
            axis = omega / np.linalg.norm(omega)
            dq = axis_angle_to_quat(axis, angle)
            q = quat_normalize(quat_multiply(q, dq))

        accel_local = imu["accel"][i]
        accel_global = rotate_vector_by_quat(q, accel_local)

        accel_norm = np.linalg.norm(accel_global)
        if accel_norm > 1e-10:
            accel_dir = accel_global / accel_norm

            tilt_axis = np.cross(accel_dir, global_up)
            tilt_axis_norm = np.linalg.norm(tilt_axis)

            if tilt_axis_norm >= 1e-8:
                tilt_axis = tilt_axis / tilt_axis_norm
                dot_val = np.clip(np.dot(accel_dir, global_up), -1.0, 1.0)
                tilt_angle = np.arccos(dot_val)

                q_acc = axis_angle_to_quat(tilt_axis, alpha * tilt_angle)
                q = quat_normalize(quat_multiply(q_acc, q))
        if heading_ref is not None:
            mag_i = imu["mag"][i]
            mag_norm_i = np.linalg.norm(mag_i)
            if abs(mag_norm_i - expected_mag_norm) / max(expected_mag_norm, 1e-10) < 0.3:
                heading_mag = get_heading(imu["accel"][i], mag_i)

                if heading_mag is not None:    
                    mag_yaw = heading_mag - heading_ref
                    while mag_yaw > np.pi:
                        mag_yaw -= 2 * np.pi
                    while mag_yaw < -np.pi:
                        mag_yaw += 2 * np.pi

                    _, _, yaw_est = quat_to_euler(q)

                    yaw_error = mag_yaw - yaw_est
                    while yaw_error > np.pi:
                        yaw_error -= 2 * np.pi
                    while yaw_error < -np.pi:
                        yaw_error += 2 * np.pi

                    yaw_fix = alpha_mag * yaw_error
                    q_mag = np.array([
                        np.cos(yaw_fix / 2),
                        0.0,
                        0.0,
                        np.sin(yaw_fix / 2)
                    ])

                    q = quat_normalize(quat_multiply(q_mag, q))

        result[i] = q

    return result


def render_model(vertices, faces, model_mat, view_mat, proj_mat, vp_mat, light_dir):
    vertices_h = np.c_[vertices, np.ones(len(vertices))]

    world_vertices = vertices_h @ model_mat.T
    cam_vertices_h = world_vertices @ view_mat.T
    cam_vertices = cam_vertices_h[:, :3]

    clip = cam_vertices_h @ proj_mat.T
    ndc = clip[:, :3] / clip[:, 3:4]

    ndc_h = np.c_[ndc, np.ones(len(ndc))]
    screen_vertices = (ndc_h @ vp_mat.T)[:, :3]

    tri_cam = cam_vertices[faces]
    tri_screen = screen_vertices[faces]

    T = np.transpose(tri_screen, axes=[0, 2, 1]).copy()
    T[:, 2, :] = 1

    det = np.linalg.det(T)
    valid = np.abs(det) > 1e-10

    T_inv = np.zeros_like(T)
    T_inv[valid] = np.linalg.inv(T[valid])

    normals = np.cross(tri_cam[:, 2] - tri_cam[:, 0], tri_cam[:, 1] - tri_cam[:, 0])
    norm_len = np.linalg.norm(normals, axis=1).reshape(len(normals), 1)
    norm_len[norm_len == 0] = 1
    normals = normals / norm_len

    intensity = np.dot(normals, light_dir) * 255

    h, w = engine.image.shape[0], engine.image.shape[1]
    visible = np.argwhere((intensity >= 0) & valid)[:, 0]
    for i in visible:
        v0, v1, v2 = tri_screen[i]
        if (v0[0] < 0 or v0[0] >= w or v0[1] < 0 or v0[1] >= h or
            v1[0] < 0 or v1[0] >= w or v1[1] < 0 or v1[1] >= h or
            v2[0] < 0 or v2[0] >= w or v2[1] < 0 or v2[1] >= h):
            continue
        triangle(T_inv[i], v0, v1, v2, intensity[i])


def clear_frame():
    engine.image[:] = 0
    engine.zbuffer[:] = -1e18
#5.1 5.2
def Task5_physics(orientations, imu, vertices, faces, view_mat, proj_mat, vp_mat,
                 light_dir, title, samples_per_frame):

    total_samples = len(imu["time"])

    gravity = -9.8
    Cd = 0.5
    air_density = 1.3
    area = 0.2
    mass = 1.0
    bunny_radius = 0.3
    falling = [
        {"pos": np.array([ 0.5, 3.0, 0.0]), "vel": np.array([0.0, 0.0, 0.0])},
        {"pos": np.array([-0.3, 4.0, 0.2]), "vel": np.array([0.0, 0.0, 0.0])},
        {"pos": np.array([ 0.0, 5.0,-0.1]), "vel": np.array([0.0, 0.0, 0.0])},
        {"pos": np.array([ 0.2, 3.5, 0.3]), "vel": np.array([0.0, 0.0, 0.0])},
    ]
    center_pos = np.array([0.0, 0.0, 0.0])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(np.zeros((engine.image.shape[0], engine.image.shape[1], 4), dtype=np.uint8))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    h, w = engine.image.shape[0], engine.image.shape[1]
    filename = title.replace(" ", "_").replace(":", "") + ".mp4"
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 3, (w, h))
    print(f"Saving video to {filename}")

    frame_count = 0
    start_total = time.time()
    idx = 0

    try:
        while idx < total_samples:
            frame_start = time.time()
            clear_frame()
            dt_physics = imu["time"][min(idx + samples_per_frame, total_samples-1)] - imu["time"][idx]

            q = orientations[idx]
            render_model(vertices, faces, quat_to_matrix_4x4(q), view_mat, proj_mat, vp_mat, light_dir)

            for b in falling:
                f_gravity = mass * gravity
                speed = np.linalg.norm(b["vel"])
                if speed > 1e-10:
                    f_drag = 0.5 * Cd * air_density * area * speed * speed
                    drag_force = (-b["vel"] / speed) * f_drag
                else:
                    drag_force = np.array([0.0, 0.0, 0.0])

                b["vel"][1] += (f_gravity / mass) * dt_physics
                b["vel"] += (drag_force / mass) * dt_physics
                b["pos"] += b["vel"] * dt_physics

                if b["pos"][1] < -1.5:
                    b["pos"][1] = -1.5
                    b["vel"][1] = -b["vel"][1] * 0.5

                dist = np.linalg.norm(b["pos"] - center_pos)
                if dist < bunny_radius * 2 and dist > 1e-10:
                    bounce_dir = (b["pos"] - center_pos) / dist
                    b["vel"] = bounce_dir * np.linalg.norm(b["vel"]) * 0.8
                    b["pos"] = center_pos + bounce_dir * bunny_radius * 2.1

                if -3 < b["pos"][0] < 3 and -2 < b["pos"][1] < 6 and -3 < b["pos"][2] < 3:
                    model_fall = translate(b["pos"][0], b["pos"][1], b["pos"][2]) @ scale(0.4, 0.4, 0.4)
                    render_model(vertices, faces, model_fall, view_mat, proj_mat, vp_mat, light_dir)

            frame_rgba = engine.image[::-1, :, :]
            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            writer.write(frame_bgr)

            img.set_data(frame_rgba)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            frame_time = time.time() - frame_start
            frame_count += 1
            idx += samples_per_frame

            if frame_count % 10 == 0:
                print(f"Frame {frame_count} | t={imu['time'][min(idx, total_samples-1)]:.2f}s | {frame_time:.2f}s")

    except KeyboardInterrupt:
        pass

    writer.release()
    print("Video saved.")

    total_time = time.time() - start_total
    print(f"\n{title} finished | frames={frame_count} | time={total_time:.1f}s")
    plt.ioff()
    plt.close()
    
def play_sequence(orientations, imu, vertices, faces, view_mat, proj_mat, vp_mat,
                  light_dir, title, samples_per_frame):

    total_samples = len(imu["time"])

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(np.zeros((engine.image.shape[0], engine.image.shape[1], 4), dtype=np.uint8))
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()

    h, w = engine.image.shape[0], engine.image.shape[1]
    filename = title.replace(" ", "_").replace(":", "") + ".mp4"
    writer = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    print(f"Saving video to {filename}")

    frame_count = 0
    start_total = time.time()
    idx = 0

    try:
        while idx < total_samples:
            frame_start = time.time()
            clear_frame()

            q = orientations[idx]
            render_model(vertices, faces, quat_to_matrix_4x4(q), view_mat, proj_mat, vp_mat, light_dir)
            
            frame_rgba = engine.image[::-1, :, :]

            frame_bgr = cv2.cvtColor(frame_rgba, cv2.COLOR_RGBA2BGR)
            writer.write(frame_bgr)

            img.set_data(frame_rgba)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

            frame_time = time.time() - frame_start
            frame_count += 1
            idx += samples_per_frame

            if frame_count % 10 == 0:
                roll, pitch, yaw = quat_to_euler(q)
                t_now = imu["time"][min(idx, total_samples - 1)]
                print(
                    f"Frame {frame_count} | t={t_now:.2f}s | "
                    f"roll={np.degrees(roll):.1f}° "
                    f"pitch={np.degrees(pitch):.1f}° "
                    f"yaw={np.degrees(yaw):.1f}° | "
                    f"{frame_time:.2f}s"
                )

    except KeyboardInterrupt:
        pass

    writer.release()
    print("Video saved.")

    total_time = time.time() - start_total
    fps = frame_count / max(total_time, 1e-6)
    print(f"\n{title} finished | frames={frame_count} | time={total_time:.1f}s | FPS={fps:.1f}")

    plt.ioff()
    plt.close()


if __name__ == "__main__":
    width, height = 800, 800

    light_dir = np.array([0, 0, -1], dtype=float)
    eye = np.array([-1.5, 1.0, 2.0], dtype=float)

    engine.image = np.zeros((height, width, 4), dtype=np.uint8)
    engine.zbuffer = -1e18 * np.ones((height, width), dtype=float)
    engine.coords = np.mgrid[0:width, 0:height].astype(int)

    view_mat = lookat(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    proj_mat = perspective(fov_deg=60, aspect=1, near=0.1, far=100.0)
    vp_mat = viewport(32, 32, width - 64, height - 64, 1000)

    bunny_v, bunny_f = obj_load("bunny.obj")
    print(f"Vertices: {len(bunny_v)}, Faces: {len(bunny_f)}")
    vmin, vmax = bunny_v.min(), bunny_v.max()
    bunny_v = (2 * (bunny_v - vmin) / (vmax - vmin) - 1) * 1.25
    bunny_v[:, 0] -= (bunny_v[:, 0].min() + bunny_v[:, 0].max()) / 2
    bunny_v[:, 1] -= (bunny_v[:, 1].min() + bunny_v[:, 1].max()) / 2

    imu = load_imu_data("IMUData.csv")
    print(f"{len(imu['time'])} samples loaded, duration = {imu['time'][-1]:.1f}s")

    samples_per_frame = 10

    print("\nTask3.1: Gyro Only")
    orient_gyro = dead_reckoning(imu)
    r, p, y = quat_to_euler(orient_gyro[-1])
    print(f"Final: roll={np.degrees(r):.1f}°, pitch={np.degrees(p):.1f}°, yaw={np.degrees(y):.1f}°")
    play_sequence(orient_gyro, imu, bunny_v, bunny_f, view_mat, proj_mat, vp_mat, light_dir,"3.1 Gyro Only", samples_per_frame)

    print("\nTask3.2: Gyro + Accel")
    orient_accel = gyro_accel_complementary(imu, alpha=0.02)
    r, p, y = quat_to_euler(orient_accel[-1])
    print(f"Final: roll={np.degrees(r):.1f}°, pitch={np.degrees(p):.1f}°, yaw={np.degrees(y):.1f}°")
    play_sequence(orient_accel, imu, bunny_v, bunny_f,view_mat, proj_mat, vp_mat, light_dir,"3.2 Gyro + Accel", samples_per_frame)
    
    print("\nTilt Error Analysis")
    alpha = 0.02
    tilt_gyro = compute_tilt_error_series(orient_gyro, imu["accel"])
    tilt_accel = compute_tilt_error_series(orient_accel, imu["accel"])

    stats_gyro = get_stats(tilt_gyro)
    stats_accel = get_stats(tilt_accel)
    print_comparison_table(stats_gyro, stats_accel, alpha=alpha)

    plt.figure(figsize=(6, 5))
    plt.boxplot(
        [tilt_gyro[~np.isnan(tilt_gyro)], tilt_accel[~np.isnan(tilt_accel)]],
        tick_labels=["Gyro only", f"Gyro + Accel\n(α={alpha})"]
    )
    plt.ylabel("Tilt Error (deg)")
    plt.title("Tilt Error Comparison (Box Plot)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("tilt_boxplot.png", dpi=150)
    plt.show()
    
    print("\nTask4: Gyro + Accel + Mag")
    orient_full = gyro_accel_mag_complementary(imu, alpha=0.02, alpha_mag=0.05)
    r, p, y = quat_to_euler(orient_full[-1])
    print(f"Final: roll={np.degrees(r):.1f}°, pitch={np.degrees(p):.1f}°, yaw={np.degrees(y):.1f}°")
    play_sequence(orient_full, imu, bunny_v, bunny_f,view_mat, proj_mat, vp_mat, light_dir,"4.Gyro + Accel + Mag", samples_per_frame)
    
    print("\nTask5: Physics Simulation")
    #using same orientation as 3.2 but add physics simulation of falling bunnies and collisions
    Task5_physics(orient_accel, imu, bunny_v, bunny_f,view_mat, proj_mat, vp_mat, light_dir,"5.Physics", samples_per_frame=10)