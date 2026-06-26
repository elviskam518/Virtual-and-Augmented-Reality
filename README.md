COMP3751 VR Coursework
CIS Username: cqst66

Files included
- render.py
- engine.py
- bunny.obj
- floor.obj (In Problem 5, the ground is implemented as an implicit plane at y = -1.5 rather than using the provided floor.obj model.)
- IMUData.csv
- VR.pdf
- demo video file(3.1 , 3.2)

Requirements
- Python 3.11.1
- NumPy
- OpenCV (cv2)

By default, running:

    python3 render.py

will execute the four main required sequences:
- Problem 3.1: Gyro-only dead reckoning  
- Problem 3.2: Gyro + accelerometer fusion  
- Problem 4: Gyro + accelerometer + magnetometer  
- Problem 5: Physics simulation  