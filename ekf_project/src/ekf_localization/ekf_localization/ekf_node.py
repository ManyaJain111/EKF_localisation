#!/usr/bin/env python3

import math
import random
import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point
from visualization_msgs.msg import Marker
from builtin_interfaces.msg import Time
from std_msgs.msg import Float32MultiArray, String, Float64
import time
from std_srvs.srv import Trigger


def normalize_angle(a):
    """Normalize angle to [-pi, pi]"""
    return (a + math.pi) % (2 * math.pi) - math.pi


def ellipse_points(mean, P, n_sigma=3):
    """Generate points for covariance ellipse"""
    cov = P[:2, :2]
    cov = 0.5 * (cov + cov.T)

    eig_vals, eig_vecs = np.linalg.eig(cov)

    order = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[order]
    eig_vecs = eig_vecs[:, order]

    eig_vals = np.maximum(eig_vals, 1e-6)

    angle = math.atan2(eig_vecs[1, 0], eig_vecs[0, 0])

    # Use n_sigma ellipse
    width = 2 * n_sigma * math.sqrt(eig_vals[0])
    height = 2 * n_sigma * math.sqrt(eig_vals[1])

    pts = []
    for t in np.linspace(0, 2 * math.pi, 50):
        x = width / 2 * math.cos(t)
        y = height / 2 * math.sin(t)

        xr = x * math.cos(angle) - y * math.sin(angle)
        yr = x * math.sin(angle) + y * math.cos(angle)

        pts.append((mean[0] + xr, mean[1] + yr))

    return pts, eig_vals, eig_vecs, angle


class EKFNode(Node):

    def __init__(self):

        super().__init__('ekf_node')

        self.dt = 0.1

        # Initial poses
        self.x_true = np.array([5.0, 5.0, 0.0])
        self.x_odom = np.array([5.0, 5.0, 0.0])
        self.x_ekf = np.array([5.0, 5.0, 0.0])

        self.v = 0.5  # Reduced speed for smoother motion
        self.w = 0.0

        # Initial covariance
        self.P = np.diag([0.1, 0.1, 0.05])  # Smaller initial covariance

        # Base process noise
        self.Q_base = np.diag([0.02, 0.02, 0.01])  # Smaller base noise
        
        # Q configurations for analysis (Q_scale = 0.1, 1.0, 5.0)
        self.Q_configs = {
            'low_noise': self.Q_base * 0.1,      # Q_scale = 0.1
            'medium_noise': self.Q_base * 1.0,   # Q_scale = 1.0
            'high_noise': self.Q_base * 5.0       # Q_scale = 5.0
        }
        
        # R configurations for analysis
        self.R_configs = {
            'low_noise': np.diag([0.002, 0.001]),
            'medium_noise': np.diag([0.01, 0.005]),
            'high_noise': np.diag([0.05, 0.02])
        }
        
        # Start with medium noise for both
        self.current_Q_config = 'medium_noise'
        self.current_R_config = 'medium_noise'
        self.Q = self.Q_configs[self.current_Q_config]
        self.R = self.R_configs[self.current_R_config]

        # For performance tracking
        self.performance_metrics = {
            'Q_low': {'errors': [], 'timestamps': [], 'eig_vals': []},
            'Q_medium': {'errors': [], 'timestamps': [], 'eig_vals': []},
            'Q_high': {'errors': [], 'timestamps': [], 'eig_vals': []},
            'R_low': {'errors': [], 'timestamps': [], 'eig_vals': []},
            'R_medium': {'errors': [], 'timestamps': [], 'eig_vals': []},
            'R_high': {'errors': [], 'timestamps': [], 'eig_vals': []}
        }
        
        self.metric_start_time = time.time()
        self.config_change_time = self.metric_start_time
        self.step_count = 0
        
        # Store current eigenvalues for tracking
        self.current_eig_vals = np.array([0.3, 0.3])

        # Simulate robotkit_ekf package for comparison
        self.x_robotkit = np.array([5.0, 5.0, 0.0])
        self.P_robotkit = np.diag([0.1, 0.1, 0.05])

        # Smoothing for EKF path
        self.ekf_path_buffer = []
        self.smoothing_window = 10  # Increased for smoother path
        
        # Add low-pass filter for EKF updates
        self.alpha = 0.3  # Filter gain (lower = smoother)
        self.x_ekf_filtered = self.x_ekf.copy()

        # Landmarks
        self.landmarks = {
            0: (2, 2),
            1: (8, 2),
            2: (8, 8),
            3: (2, 8),
            4: (5, 1),
            5: (9, 5),
            6: (1, 5)
        }

        # Obstacles for reactive control
        self.obstacles = [
            (4, 4),
            (6, 3),
            (3, 7),
            (7, 6)
        ]

        # Publishers
        self.true_pub = self.create_publisher(Path, '/true_path', 10)
        self.odom_pub = self.create_publisher(Path, '/odom_path', 10)
        self.ekf_pub = self.create_publisher(Path, '/ekf_path', 10)
        self.ekf_raw_pub = self.create_publisher(Path, '/ekf_raw_path', 10)  # For comparison
        self.robotkit_pub = self.create_publisher(Path, '/robotkit_ekf_path', 10)

        self.landmark_pub = self.create_publisher(Marker, '/landmarks', 10)
        self.obs_pub = self.create_publisher(Marker, '/obstacles', 10)
        self.map_pub = self.create_publisher(Marker, '/map_boundary', 10)
        self.cov_pub = self.create_publisher(Marker, '/ekf_covariance', 10)
        
        # Publishers for analysis
        self.metrics_pub = self.create_publisher(Float32MultiArray, '/ekf_metrics', 10)
        self.config_pub = self.create_publisher(String, '/ekf_config', 10)
        self.eig_pub = self.create_publisher(Float32MultiArray, '/ekf_eigenvalues', 10)
        self.error_pub = self.create_publisher(Float64, '/ekf_position_error', 10)
        
        # Subscribers for dynamic config changes
        self.create_subscription(String, '/set_q_config', self.q_config_callback, 10)
        self.create_subscription(String, '/set_r_config', self.r_config_callback, 10)
        self.create_subscription(String, '/kidnap_robot', self.kidnap_callback, 10)
        
        # Services for data logging
        self.create_service(Trigger, '/save_analysis_data', self.save_analysis_callback)

        # Path messages
        self.true_path = Path()
        self.true_path.header.frame_id = "map"

        self.odom_path = Path()
        self.odom_path.header.frame_id = "map"

        self.ekf_path = Path()
        self.ekf_path.header.frame_id = "map"
        
        self.ekf_raw_path = Path()
        self.ekf_raw_path.header.frame_id = "map"
        
        self.robotkit_path = Path()
        self.robotkit_path.header.frame_id = "map"

        # Timers
        self.timer = self.create_timer(self.dt, self.loop)
        self.analysis_timer = self.create_timer(2.0, self.publish_analysis)  # Every 2 seconds

        self.target_heading = random.uniform(-math.pi, math.pi)
        self.last_landmark_count = 0
        
        self.get_logger().info('EKF Node initialized with Q and R analysis capabilities')
        self.get_logger().info('Available Q scales: low_noise(0.1), medium_noise(1.0), high_noise(5.0)')
        self.get_logger().info('Available R configs: low_noise, medium_noise, high_noise')

    def kidnap_callback(self, msg):
        """Teleport the true robot to a new position"""
        try:
            parts = msg.data.split(',')
            x, y = float(parts[0]), float(parts[1])

            # Move the TRUE robot only
            self.x_true = np.array([x, y, self.x_true[2]])

            self.get_logger().info(
                f'🚨 ROBOT KIDNAPPED to ({x}, {y})! EKF does not know...'
            )

        except Exception as e:
            self.get_logger().warn('Invalid kidnap format. Use: x,y')

    def get_time(self):
        """Get current time stamp"""
        t = self.get_clock().now().seconds_nanoseconds()
        msg = Time()
        msg.sec = int(t[0])
        msg.nanosec = int(t[1])
        return msg

    def q_config_callback(self, msg):
        """Dynamically change Q configuration for analysis"""
        if msg.data in self.Q_configs:
            self.current_Q_config = msg.data
            self.Q = self.Q_configs[self.current_Q_config]
            scale = {'low_noise': 0.1, 'medium_noise': 1.0, 'high_noise': 5.0}[msg.data]
            self.get_logger().info(f'🔄 Switched to Q_scale = {scale} ({self.current_Q_config})')
            
            config_msg = String()
            config_msg.data = f"Current config - Q: {self.current_Q_config} (scale={scale}), R: {self.current_R_config}"
            self.config_pub.publish(config_msg)
            
            self.config_change_time = time.time()

    def r_config_callback(self, msg):
        """Dynamically change R configuration for analysis"""
        if msg.data in self.R_configs:
            self.current_R_config = msg.data
            self.R = self.R_configs[self.current_R_config]
            self.get_logger().info(f'🔄 Switched to R: {self.current_R_config}')
            
            config_msg = String()
            config_msg.data = f"Current config - Q: {self.current_Q_config}, R: {self.current_R_config}"
            self.config_pub.publish(config_msg)
            
            self.config_change_time = time.time()

    def save_analysis_callback(self, request, response):
        """Save analysis data to file"""
        try:
            filename = f"ekf_analysis_{time.strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, 'w') as f:
                f.write("EKF Analysis Results\n")
                f.write("=" * 50 + "\n\n")
                
                f.write("Q Configuration Analysis (scales: 0.1, 1.0, 5.0):\n")
                q_scales = {'Q_low': 0.1, 'Q_medium': 1.0, 'Q_high': 5.0}
                for q_config, scale in q_scales.items():
                    data = self.performance_metrics[q_config]
                    if data['errors']:
                        avg_error = np.mean(data['errors'])
                        avg_eig = np.mean(data['eig_vals']) if data['eig_vals'] else 0
                        f.write(f"  Q_scale={scale}: Avg Error={avg_error:.3f}m, Avg Eigenvalue={avg_eig:.3f}\n")
                
                f.write("\nR Configuration Analysis:\n")
                for r_config in ['R_low', 'R_medium', 'R_high']:
                    data = self.performance_metrics[r_config]
                    if data['errors']:
                        avg_error = np.mean(data['errors'])
                        avg_eig = np.mean(data['eig_vals']) if data['eig_vals'] else 0
                        f.write(f"  {r_config}: Avg Error={avg_error:.3f}m, Avg Eigenvalue={avg_eig:.3f}\n")
                        
                f.write("\nEllipse Size History:\n")
                f.write("Step\tQ_Config\tR_Config\tEig_Major\tEig_Minor\tError\n")
                # Could add more detailed history here
            
            response.success = True
            response.message = f"Data saved to {filename}"
            self.get_logger().info(f'💾 Analysis saved to {filename}')
        except Exception as e:
            response.success = False
            response.message = str(e)
        
        return response

    def simulate_control(self):
        """Simulate control input with obstacle avoidance"""
        x, y, th = self.x_true

        # Smoother heading changes
        if random.random() < 0.08:  # Reduced from 0.03
            self.target_heading = random.uniform(-math.pi, math.pi)

        steer = normalize_angle(self.target_heading - th)

        # Obstacle avoidance
        for ox, oy in self.obstacles:
            dx = x - ox
            dy = y - oy
            d = math.sqrt(dx * dx + dy * dy)

            if d < 1.5:
                avoid = math.atan2(dy, dx)
                steer += 1.5 * normalize_angle(avoid - th)  # Reduced from 2.0

        # Boundary avoidance
        margin = 1.0
        if x < margin:
            steer += 1.0  # Reduced from 1.5
        if x > 10 - margin:
            steer -= 1.0
        if y < margin:
            steer += 1.0
        if y > 10 - margin:
            steer -= 1.0

        # Smooth velocity changes
        self.w += 0.25 * (steer - self.w)  # Reduced from 0.2

        return np.array([self.v, self.w])

    def simulate_true(self, u):
        """Simulate true robot motion"""
        v, w = u
        x, y, th = self.x_true

        x += v * self.dt * math.cos(th)
        y += v * self.dt * math.sin(th)
        th += w * self.dt

        # Keep within bounds
        x = max(0.3, min(9.7, x))
        y = max(0.3, min(9.7, y))

        self.x_true = np.array([x, y, normalize_angle(th)])

    def simulate_odom(self, u):
        """Simulate noisy odometry"""
        v, w = u

        # Add noise to control inputs
        v += random.gauss(0, 0.05)  # Reduced from 0.08
        w += random.gauss(0, 0.03)  # Reduced from 0.05

        x, y, th = self.x_odom

        x += v * self.dt * math.cos(th)
        y += v * self.dt * math.sin(th)
        th += w * self.dt

        x = max(0.3, min(9.7, x))
        y = max(0.3, min(9.7, y))

        self.x_odom = np.array([x, y, normalize_angle(th)])

    def get_measurements(self):
        """Simulate landmark measurements"""
        z = []

        for lx, ly in self.landmarks.values():
            dx = lx - self.x_true[0]
            dy = ly - self.x_true[1]
            d = math.sqrt(dx ** 2 + dy ** 2)

            if d < 7:  # Only observe landmarks within range
                r = d + random.gauss(0, self.R[0, 0])
                b = math.atan2(dy, dx) - self.x_true[2]
                b = normalize_angle(b + random.gauss(0, self.R[1, 1]))
                z.append((lx, ly, r, b))

        return z

    def ekf_predict(self, u):
        """EKF prediction step"""
        v, w = u
        x, y, th = self.x_ekf
        th_prev = th

        # State prediction
        x += v * self.dt * math.cos(th)
        y += v * self.dt * math.sin(th)
        th += w * self.dt

        # Jacobian
        F = np.array([
            [1, 0, -v * self.dt * math.sin(th_prev)],
            [0, 1, v * self.dt * math.cos(th_prev)],
            [0, 0, 1]
        ])

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q
        self.P = 0.5 * (self.P + self.P.T)  # Ensure symmetry
        self.x_ekf = np.array([x, y, normalize_angle(th)])

    def ekf_update(self, z):
        """EKF update step with landmarks"""
        for lx, ly, r, b in z:
            dx = lx - self.x_ekf[0]
            dy = ly - self.x_ekf[1]
            q = dx ** 2 + dy ** 2
            d = math.sqrt(q)

            if d < 0.01:
                continue

            # Predicted measurement
            zhat = np.array([
                d,
                normalize_angle(math.atan2(dy, dx) - self.x_ekf[2])
            ])

            # Innovation
            y = np.array([r, b]) - zhat
            y[1] = normalize_angle(y[1])

            # Measurement Jacobian
            H = np.array([
                [-dx / d, -dy / d, 0],
                [dy / q, -dx / q, -1]
            ])

            # Kalman gain
            S = H @ self.P @ H.T + self.R
            K = self.P @ H.T @ np.linalg.inv(S)

            # State update
            self.x_ekf = self.x_ekf + K @ y
            self.x_ekf[2] = normalize_angle(self.x_ekf[2])

            # Covariance update
            self.P = (np.eye(3) - K @ H) @ self.P
            self.P = 0.5 * (self.P + self.P.T)

    def robotkit_ekf_predict_update(self, u, z):
        """Simulate robotkit_ekf package behavior for comparison"""
        v, w = u
        x, y, th = self.x_robotkit
        th_prev = th

        # Predict
        x += v * self.dt * math.cos(th)
        y += v * self.dt * math.sin(th)
        th += w * self.dt

        F = np.array([
            [1, 0, -v * self.dt * math.sin(th_prev)],
            [0, 1, v * self.dt * math.cos(th_prev)],
            [0, 0, 1]
        ])

        self.P_robotkit = F @ self.P_robotkit @ F.T + self.Q

        # Update with gating (simulating package behavior)
        for lx, ly, r, b in z:
            dx = lx - x
            dy = ly - y
            d = math.sqrt(dx ** 2 + dy ** 2)

            if d < 0.01:
                continue

            zhat_d = d
            zhat_b = normalize_angle(math.atan2(dy, dx) - th)
            
            innovation_r = r - zhat_d
            innovation_b = normalize_angle(b - zhat_b)
            
            # Only update if innovation is within 3 sigma (gating)
            if abs(innovation_r) < 3 * math.sqrt(self.R[0, 0]) and \
               abs(innovation_b) < 3 * math.sqrt(self.R[1, 1]):
                
                H = np.array([
                    [-dx / d, -dy / d, 0],
                    [dy / (d**2), -dx / (d**2), -1]
                ])

                S = H @ self.P_robotkit @ H.T + self.R
                K = self.P_robotkit @ H.T @ np.linalg.inv(S)

                correction = K @ np.array([innovation_r, innovation_b])
                x += correction[0]
                y += correction[1]
                th += correction[2]

                self.P_robotkit = (np.eye(3) - K @ H) @ self.P_robotkit

        self.x_robotkit = np.array([x, y, normalize_angle(th)])
        self.P_robotkit = 0.5 * (self.P_robotkit + self.P_robotkit.T)

    def smooth_ekf_pose(self, raw_pose):
        """Apply smoothing to EKF path"""
        self.ekf_path_buffer.append(raw_pose)
        
        if len(self.ekf_path_buffer) > self.smoothing_window:
            self.ekf_path_buffer.pop(0)
        
        if len(self.ekf_path_buffer) < 3:
            return raw_pose
        
        # Use Gaussian-like weighting
        n = len(self.ekf_path_buffer)
        weights = np.exp(-0.5 * ((np.arange(n) - n/2) / (n/4)) ** 2)
        weights /= weights.sum()
        
        smoothed = np.zeros(3)
        for i, pose in enumerate(self.ekf_path_buffer):
            smoothed += weights[i] * pose
        
        # Apply low-pass filter
        self.x_ekf_filtered = self.alpha * smoothed + (1 - self.alpha) * self.x_ekf_filtered
        self.x_ekf_filtered[2] = normalize_angle(self.x_ekf_filtered[2])
        
        return self.x_ekf_filtered

    def track_performance(self):
        """Track EKF performance metrics for both Q and R configs"""
        # Calculate position error
        error = np.linalg.norm(self.x_ekf[:2] - self.x_true[:2])
        
        # Store error for current configs
        q_key = f'Q_{self.current_Q_config.split("_")[0]}'
        r_key = f'R_{self.current_R_config.split("_")[0]}'
        
        self.performance_metrics[q_key]['errors'].append(error)
        self.performance_metrics[q_key]['timestamps'].append(
            time.time() - self.config_change_time
        )
        self.performance_metrics[q_key]['eig_vals'].append(np.mean(self.current_eig_vals))
        
        self.performance_metrics[r_key]['errors'].append(error)
        self.performance_metrics[r_key]['timestamps'].append(
            time.time() - self.config_change_time
        )
        self.performance_metrics[r_key]['eig_vals'].append(np.mean(self.current_eig_vals))
        
        # Publish current error
        error_msg = Float64()
        error_msg.data = error
        self.error_pub.publish(error_msg)

    def publish_analysis(self):
        """Publish performance analysis with numerical ellipse data"""
        # Calculate average errors for each config
        avg_errors = Float32MultiArray()
        avg_errors.data = [
            np.mean(self.performance_metrics['Q_low']['errors'][-50:]) if self.performance_metrics['Q_low']['errors'] else 0.0,
            np.mean(self.performance_metrics['Q_medium']['errors'][-50:]) if self.performance_metrics['Q_medium']['errors'] else 0.0,
            np.mean(self.performance_metrics['Q_high']['errors'][-50:]) if self.performance_metrics['Q_high']['errors'] else 0.0,
            np.mean(self.performance_metrics['R_low']['errors'][-50:]) if self.performance_metrics['R_low']['errors'] else 0.0,
            np.mean(self.performance_metrics['R_medium']['errors'][-50:]) if self.performance_metrics['R_medium']['errors'] else 0.0,
            np.mean(self.performance_metrics['R_high']['errors'][-50:]) if self.performance_metrics['R_high']['errors'] else 0.0
        ]
        self.metrics_pub.publish(avg_errors)
        
        # Publish current eigenvalues (ellipse size)
        eig_msg = Float32MultiArray()
        eig_msg.data = [
            float(self.current_eig_vals[0]),  # Major axis
            float(self.current_eig_vals[1]),  # Minor axis
            float(np.mean(self.current_eig_vals))  # Average size
        ]
        self.eig_pub.publish(eig_msg)
        
        # Log detailed analysis
        self.get_logger().info('=' * 60)
        self.get_logger().info('📊 EKF PERFORMANCE ANALYSIS:')
        q_scale = {'low_noise': 0.1, 'medium_noise': 1.0, 'high_noise': 5.0}[self.current_Q_config]
        self.get_logger().info(f'Current: Q={self.current_Q_config} (scale={q_scale}), R={self.current_R_config}')
        
        # Q analysis
        q_scales = {'low_noise': 0.1, 'medium_noise': 1.0, 'high_noise': 5.0}
        self.get_logger().info('\n🔵 Q CONFIGURATION ANALYSIS (scales: 0.1, 1.0, 5.0):')
        for q_config, scale in q_scales.items():
            key = f'Q_{q_config.split("_")[0]}'
            if self.performance_metrics[key]['errors']:
                avg_error = np.mean(self.performance_metrics[key]['errors'][-50:])  # Last 50 samples
                avg_eig = np.mean(self.performance_metrics[key]['eig_vals'][-50:]) if self.performance_metrics[key]['eig_vals'] else 0
                self.get_logger().info(f'  Q_scale={scale:.1f}: Error={avg_error:.3f}m, Ellipse Size={avg_eig:.3f}')
        
        # R analysis
        self.get_logger().info('\n🔴 R CONFIGURATION ANALYSIS:')
        for r_config in ['low_noise', 'medium_noise', 'high_noise']:
            key = f'R_{r_config.split("_")[0]}'
            if self.performance_metrics[key]['errors']:
                avg_error = np.mean(self.performance_metrics[key]['errors'][-50:])
                avg_eig = np.mean(self.performance_metrics[key]['eig_vals'][-50:]) if self.performance_metrics[key]['eig_vals'] else 0
                self.get_logger().info(f'  R_{r_config}: Error={avg_error:.3f}m, Ellipse Size={avg_eig:.3f}')
        
        # Current ellipse size
        self.get_logger().info(f'\n📐 Current Ellipse: major={self.current_eig_vals[0]:.3f}, minor={self.current_eig_vals[1]:.3f}, avg={np.mean(self.current_eig_vals):.3f}')
        
        # Landmark info
        landmarks_visible = len(self.get_measurements())
        self.get_logger().info(f'👁️ Landmarks visible: {landmarks_visible}')
        
        # Compare with robotkit_ekf
        robotkit_error = np.linalg.norm(self.x_robotkit[:2] - self.x_true[:2])
        self.get_logger().info(f'🤖 robotkit_ekf error: {robotkit_error:.3f}m')
        self.get_logger().info('=' * 60)

    def publish_path(self, path, pose, pub, is_ekf=False, is_raw=False):
        """Publish path for visualization"""
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_time()

        if is_ekf:
            if is_raw:
                # Publish raw EKF path for comparison
                msg.pose.position.x = float(pose[0])
                msg.pose.position.y = float(pose[1])
            else:
                # Publish smoothed EKF path
                smoothed_pose = self.smooth_ekf_pose(pose)
                msg.pose.position.x = float(smoothed_pose[0])
                msg.pose.position.y = float(smoothed_pose[1])
        else:
            msg.pose.position.x = float(pose[0])
            msg.pose.position.y = float(pose[1])

        path.poses.append(msg)

        if len(path.poses) > 400:
            path.poses.pop(0)

        pub.publish(path)

    def publish_covariance(self):
        """Publish covariance ellipse"""
        # Get ellipse points and eigenvalues
        pts_3sigma, eig_vals, eig_vecs, angle = ellipse_points(self.x_ekf, self.P, n_sigma=3)
        self.current_eig_vals = eig_vals

        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_time()
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.03

        # Color based on covariance size
        cov_trace = np.trace(self.P[:2, :2])
        if cov_trace < 0.5:
            m.color.g = 1.0
            m.color.r = 0.0
        elif cov_trace < 1.0:
            m.color.g = 1.0
            m.color.r = 1.0
        else:
            m.color.r = 1.0
            m.color.g = 0.0
        m.color.b = 0.0
        m.color.a = 1.0

        # Publish ellipse
        for x, y in pts_3sigma + [pts_3sigma[0]]:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            m.points.append(p)

        self.cov_pub.publish(m)
        
        # Log covariance changes with numerical values
        current_landmarks = len(self.get_measurements())
        if current_landmarks > self.last_landmark_count:
            self.get_logger().info(f'📉 Covariance shrunk: trace={cov_trace:.3f}, eigenvalues=[{eig_vals[0]:.3f}, {eig_vals[1]:.3f}] (observed {current_landmarks} landmarks)')
        elif current_landmarks < self.last_landmark_count:
            self.get_logger().info(f'📈 Covariance growing: trace={cov_trace:.3f}, eigenvalues=[{eig_vals[0]:.3f}, {eig_vals[1]:.3f}] (lost landmarks)')
        
        self.last_landmark_count = current_landmarks

    def publish_landmarks(self):
        """Publish landmarks for visualization"""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_time()
        m.type = Marker.POINTS
        m.scale.x = 0.25
        m.scale.y = 0.25

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        for x, y in self.landmarks.values():
            p = Point()
            p.x = float(x)
            p.y = float(y)
            m.points.append(p)

        self.landmark_pub.publish(m)

    def publish_obstacles(self):
        """Publish obstacles for visualization"""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_time()
        m.type = Marker.POINTS
        m.scale.x = 0.35
        m.scale.y = 0.35

        m.color.r = 1.0
        m.color.a = 1.0

        for x, y in self.obstacles:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            m.points.append(p)

        self.obs_pub.publish(m)

    def publish_map(self):
        """Publish map boundary"""
        m = Marker()
        m.header.frame_id = "map"
        m.header.stamp = self.get_time()
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.05

        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 1.0

        for x, y in [(0,0), (10,0), (10,10), (0,10), (0,0)]:
            p = Point()
            p.x = float(x)
            p.y = float(y)
            m.points.append(p)

        self.map_pub.publish(m)

    def loop(self):
        """Main loop"""
        self.step_count += 1
        u = self.simulate_control()

        self.simulate_true(u)
        self.simulate_odom(u)

        self.ekf_predict(u)

        z = self.get_measurements()
        self.ekf_update(z)
        
        # Run robotkit_ekf for comparison
        self.robotkit_ekf_predict_update(u, z)
        
        # Track performance
        self.track_performance()

        # Publish paths
        self.publish_path(self.true_path, self.x_true, self.true_pub)
        self.publish_path(self.odom_path, self.x_odom, self.odom_pub)
        self.publish_path(self.ekf_path, self.x_ekf, self.ekf_pub, is_ekf=True, is_raw=False)
        self.publish_path(self.ekf_raw_path, self.x_ekf, self.ekf_raw_pub, is_ekf=True, is_raw=True)
        self.publish_path(self.robotkit_path, self.x_robotkit, self.robotkit_pub)

        # Publish markers
        self.publish_landmarks()
        self.publish_obstacles()
        self.publish_map()
        self.publish_covariance()


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Keyboard interrupt, shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()