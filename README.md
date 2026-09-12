# IMU Orientation Tracking and 3D Physics Visualisation

This **COMP3751 Virtual & Augmented Reality coursework project** turns recorded motion-sensor data into the orientation of a rendered 3D bunny.

I extended the coursework's supplied software renderer with transformations, quaternion-based orientation tracking, sensor-fusion experiments and a simple multi-object physics simulation. The project connects the mathematics of motion tracking with a visible 3D result.

**[Read the report](VR.pdf)** · **[Gyroscope-only video](3.1_Gyro_Only.mp4)** · **[Gyroscope + accelerometer video](3.2_Gyro_%2B_Accel.mp4)**

## What I implemented

### 1. Transformations and rendering

Added translation, scaling and rotation to place and animate models in the scene. The rendering pipeline combines model, view, perspective and viewport transformations, with depth handling, flat shading and outline edges.

The supplied bunny mesh and rendering foundation support the coursework extensions; this is not a claim that the original renderer or 3D asset was created entirely from scratch.

### 2. Orientation tracking from IMU data

Compared three tracking approaches using the included `IMUData.csv`:

| Approach | Purpose |
| --- | --- |
| Gyroscope-only dead reckoning | Integrates angular velocity using quaternions to update orientation. |
| Gyroscope + accelerometer | Adds complementary tilt correction using the measured gravity direction. |
| Gyroscope + accelerometer + magnetometer | Adds a heading-correction stage and a magnetic-field magnitude check. |

The magnetometer check applies correction only when the measured field magnitude is within 30% of a reference magnitude. The report explores different accelerometer and magnetometer correction coefficients.

### 3. A physics demonstration

Rendered multiple falling bunnies around the central IMU-driven model, with gravity, ground interaction and bounding-sphere collision detection. The ground in this sequence is an implicit plane at `y = -1.5`, rather than the supplied `floor.obj` mesh.

## Evaluation reported in the coursework

For the recorded sequence, the report compares gyroscope-only tracking with accelerometer fusion at **α = 0.02**:

| Tilt diagnostic | Gyroscope only | Gyroscope + accelerometer |
| --- | ---: | ---: |
| Mean | 3.01° | 1.73° |
| Median | 2.66° | 1.26° |
| 95th percentile | 6.24° | 4.71° |

The mean decreased by approximately **42%** in this diagnostic.

The metric measures how the transformed accelerometer direction aligns with the global up vector. It is **not an independent ground-truth orientation measurement**: the accelerometer also participates in the fused estimate, and motion-induced acceleration can affect the comparison.

## Skills demonstrated

**Python · NumPy · OpenCV · Matplotlib · Quaternions · 3D transformations · Sensor fusion · Collision detection**

The project demonstrates implementing mathematical models, visualising their behaviour and examining the trade-off between correction strength, drift and sensitivity to sensor disturbances.

## Scope and limitations

The application replays recorded IMU data in a desktop visualisation; it is not a complete headset application or a positional tracking system.

Accelerometer correction cannot determine yaw from gravity alone. Bounding spheres are efficient but fit the bunny mesh loosely, so collisions can occur while visible parts of the meshes still appear separated. The report discusses tighter bounding volumes and the difficulty of estimating position by integrating noisy acceleration.

## Repository guide

- [`render.py`](render.py): transformations, orientation estimation, diagnostics, animation and physics.
- [`engine.py`](engine.py): software-rendering utilities.
- [`IMUData.csv`](IMUData.csv): recorded sensor sequence.
- [`bunny.obj`](bunny.obj) and [`floor.obj`](floor.obj): supplied scene assets.
- [`VR.pdf`](VR.pdf): experiments, figures and discussion.
- The two MP4 files demonstrate gyroscope-only and accelerometer-fused tracking.

<details>
<summary>Running the visualisation</summary>

The original environment used **Python 3.11.1**. Install NumPy, OpenCV and Matplotlib, then run from the repository root:

```bash
python render.py
```

The program runs gyroscope-only tracking, accelerometer fusion, full sensor fusion and the physics sequence. It requires a graphical desktop environment to display the figures.

</details>
