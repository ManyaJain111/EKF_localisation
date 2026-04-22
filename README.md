# EKF Localization (ROS2)
## Overview
This package implements an **Extended Kalman Filter (EKF)** for localizing a mobile robot in a **2D environment**.  
The robot moves in a **10 m × 10 m map** containing landmarks and obstacles. Using **noisy odometry and landmark measurements**, the EKF continuously estimates the robot’s pose:
(x, y, theta)
The system also provides **visualization, performance monitoring, noise analysis tools, and automated evaluation plots** to study how EKF behavior changes under different conditions.

## Project Structure

```
ros2_ws/
└── src/
    └── ekf_localization/
        └── ekf_localization/
            └── ekf_node.py          ← Main EKF node (self-contained simulation)
```

## Running the Project

### Step 1 — Start the Docker container

From your laptop terminal:

```bash
docker start ros2_ekf_container
```


### Step 2 — Enter the container

```bash
docker exec -it ros2_ekf_container bash
```


### Step 3 — Source ROS and workspace

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_ws
source install/setup.bash
```


### Step 4 — Run the EKF node

```bash
ros2 run ekf_localization ekf_node
```

You should see output similar to:

```
EKF Node initialized with Q and R analysis capabilities
Available Q scales: low_noise(0.1), medium_noise(1.0), high_noise(5.0)
Available R configs: low_noise, medium_noise, high_noise
Covariance shrunk: trace=0.005
```


## RViz Visualization

Open a **second terminal** and enter the container again:

```bash
docker exec -it ros2_ekf_container bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
rviz2
```

### RViz configuration

Add the following topics:

| Topic | Display Type |
|---|---|
| `/ekf_path` | Path |
| `/true_path` | Path |
| `/odom_path` | Path |
| `/ekf_covariance` | Marker |
| `/landmarks` | Marker |
| `/map_boundary` | Marker |


## Kidnap Robot Feature

The kidnap feature teleports the robot's **true position** to a new location while the EKF does not know about the move. This demonstrates how the EKF recovers after a sudden displacement.

### How to kidnap the robot

Open a **third terminal**:

```bash
docker exec -it ros2_ekf_container bash
source /opt/ros/humble/setup.bash
source ~/ros2_ws/install/setup.bash
```

Run:

```bash
ros2 topic pub --once /kidnap_robot std_msgs/msg/String "{data: '8.0,8.0'}"
```

You will see in the EKF node terminal:

```
🚨 ROBOT KIDNAPPED to (8.0, 8.0)! EKF does not know...
```

In RViz:

- The **true robot path** jumps to the new location
- The **EKF estimate remains incorrect initially**
- The covariance ellipse increases
- The EKF gradually re-localizes using landmark observations

To kidnap to another location:

```bash
ros2 topic pub --once /kidnap_robot std_msgs/msg/String "{data: '2.0,2.0'}"
```


## Q/R Noise Analysis

The EKF node supports switching between three noise configurations during runtime.

### Change process noise (Q)

```bash
ros2 topic pub --once /set_q_config std_msgs/msg/String "{data: 'low_noise'}"
ros2 topic pub --once /set_q_config std_msgs/msg/String "{data: 'medium_noise'}"
ros2 topic pub --once /set_q_config std_msgs/msg/String "{data: 'high_noise'}"
```

| Config | Q Scale |
|---|---|
| `low_noise` | 0.1 |
| `medium_noise` | 1.0 |
| `high_noise` | 5.0 |


### Change measurement noise (R)

```bash
ros2 topic pub --once /set_r_config std_msgs/msg/String "{data: 'low_noise'}"
ros2 topic pub --once /set_r_config std_msgs/msg/String "{data: 'medium_noise'}"
ros2 topic pub --once /set_r_config std_msgs/msg/String "{data: 'high_noise'}"
```

Performance metrics are printed every few seconds in the EKF terminal.


## Automatic Performance Plotting

An additional script `ekf_auto_plotter.py` is provided to automatically:

- switch between **Q noise configurations**
- switch between **R noise configurations**
- trigger **kidnapped robot events**
- collect EKF performance data
- generate a **performance analysis plot**

The script produces a PNG file showing:

1. EKF position error over time
2. Covariance ellipse size
3. Q noise configuration comparison
4. R noise configuration comparison


### Step 1 — Enter the container

```bash
docker exec -it ros2_ekf_container bash
```


### Step 2 — Go to the workspace

```bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
```


### Step 3 — Install plotting dependencies (only once)

```bash
apt update
apt install python3-matplotlib -y
```


### Step 4 — Run the EKF node

In the first terminal run:

```bash
ros2 run ekf_localization ekf_node
```

Leave this terminal running.


### Step 5 — Run the plotter script

Open a **second terminal** and run:

```bash
docker exec -it ros2_ekf_container bash
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
python3 ekf_auto_plotter.py
```

### What the script does automatically
1. Waits **15 seconds** for EKF stabilization
2. Cycles through **Q noise configurations**
3. Cycles through **R noise configurations**
4. Performs **two robot kidnapping events**
5. Collects EKF performance data
6. Saves a plot image


This image contains **four graphs**:

- EKF Position Error vs Time
- Covariance Ellipse Size
- Q Noise Configuration Comparison
- R Noise Configuration Comparison

## Published Topics
| Topic | Type | Description |
|---|---|---|
| `/true_path` | `nav_msgs/Path` | Ground truth robot path |
| `/odom_path` | `nav_msgs/Path` | Noisy odometry path |
| `/ekf_path` | `nav_msgs/Path` | EKF estimated path |
| `/robotkit_ekf_path` | `nav_msgs/Path` | Simulated robotkit_ekf comparison |
| `/ekf_covariance` | `visualization_msgs/Marker` | Covariance ellipse (3-sigma) |
| `/landmarks` | `visualization_msgs/Marker` | Landmark positions |
| `/obstacles` | `visualization_msgs/Marker` | Obstacle positions |
| `/map_boundary` | `visualization_msgs/Marker` | Map boundary |
| `/ekf_position_error` | `std_msgs/Float64` | EKF position error |

## Subscribed Topics
| Topic | Type | Description |
|---|---|---|
| `/set_q_config` | `std_msgs/String` | Switch process noise configuration |
| `/set_r_config` | `std_msgs/String` | Switch measurement noise configuration |
| `/kidnap_robot` | `std_msgs/String` | Teleport robot to `x,y` |

## Simulation Parameters
| Parameter | Value |
|---|---|
| Map size | 10 × 10 metres |
| Robot start position | (5, 5) |
| Landmarks | 7 |
| Obstacles | 4 |
| Robot speed | 0.5 m/s |
| Update rate | 10 Hz |
| Landmark observation range | 7 metres |



