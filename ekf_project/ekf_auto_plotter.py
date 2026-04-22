#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray, Float64, String
import matplotlib.pyplot as plt
from collections import deque
import threading
import time

MAX_POINTS = 1000

class EKFPlotter(Node):
    def __init__(self):
        super().__init__('ekf_plotter')

        self.times = deque(maxlen=MAX_POINTS)
        self.errors = deque(maxlen=MAX_POINTS)
        self.eig_major = deque(maxlen=MAX_POINTS)
        self.eig_minor = deque(maxlen=MAX_POINTS)
        self.latest_metrics = [0.0] * 6
        self.start_time = time.time()
        self.kidnap_times = []

        self.create_subscription(Float64, '/ekf_position_error', self.error_cb, 10)
        self.create_subscription(Float32MultiArray, '/ekf_eigenvalues', self.eig_cb, 10)
        self.create_subscription(Float32MultiArray, '/ekf_metrics', self.metrics_cb, 10)

        self.q_pub = self.create_publisher(String, '/set_q_config', 10)
        self.r_pub = self.create_publisher(String, '/set_r_config', 10)
        self.kidnap_pub = self.create_publisher(String, '/kidnap_robot', 10)

        self.get_logger().info('EKF Plotter ready!')

    def error_cb(self, msg):
        t = time.time() - self.start_time
        self.times.append(t)
        self.errors.append(msg.data)
        if len(self.errors) > 1 and msg.data - list(self.errors)[-2] > 0.3:
            self.kidnap_times.append(t)

    def eig_cb(self, msg):
        if len(msg.data) >= 2:
            self.eig_major.append(msg.data[0])
            self.eig_minor.append(msg.data[1])

    def metrics_cb(self, msg):
        if len(msg.data) >= 6:
            self.latest_metrics = list(msg.data)

    def publish_string(self, publisher, data):
        msg = String()
        msg.data = data
        publisher.publish(msg)

    def run_sequence(self):
        print("\n Starting automated sequence...\n")

        print("Stabilizing for 15 seconds...")
        time.sleep(15)

        for config in ['low_noise', 'medium_noise', 'high_noise']:
            print(f"Switching Q to {config}...")
            self.publish_string(self.q_pub, config)
            time.sleep(20)

        self.publish_string(self.q_pub, 'medium_noise')

        for config in ['low_noise', 'medium_noise', 'high_noise']:
            print(f"Switching R to {config}...")
            self.publish_string(self.r_pub, config)
            time.sleep(20)

        self.publish_string(self.r_pub, 'medium_noise')
        time.sleep(10)

        print("Kidnapping robot to (8.0, 8.0)...")
        self.publish_string(self.kidnap_pub, '8.0,8.0')
        time.sleep(10)

        print("Kidnapping robot to (2.0, 2.0)...")
        self.publish_string(self.kidnap_pub, '2.0,2.0')
        time.sleep(10)

        print("\nSequence complete! Saving plots...")
        self.save_plots()

    def save_plots(self):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('EKF Localization Performance Analysis', fontsize=16, fontweight='bold')

        times = list(self.times)
        errors = list(self.errors)
        eig_maj = list(self.eig_major)
        eig_min = list(self.eig_minor)

        ax1 = axes[0, 0]
        if times and errors:
            ax1.plot(times, errors, color='crimson', linewidth=1.5, label='Position Error')
            added_label = False
            for kt in self.kidnap_times:
                label = 'Kidnap Event' if not added_label else ''
                ax1.axvline(x=kt, color='orange', linestyle='--', linewidth=2, label=label)
                added_label = True
            ax1.set_title('EKF Position Error Over Time')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Error (m)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)

        ax2 = axes[0, 1]
        if eig_maj and eig_min:
            t_eig = list(range(len(eig_maj)))
            ax2.plot(t_eig, eig_maj, color='blue', linewidth=1.5, label='Major Axis')
            ax2.plot(t_eig, eig_min, color='green', linewidth=1.5, label='Minor Axis')
            ax2.fill_between(t_eig, eig_min, eig_maj, alpha=0.2, color='skyblue')
            ax2.set_title('Covariance Ellipse Size Over Time')
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Eigenvalue')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        q_labels = ['Q low\n(scale=0.1)', 'Q medium\n(scale=1.0)', 'Q high\n(scale=5.0)']
        q_errors = self.latest_metrics[:3]
        colors_q = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax3.bar(q_labels, q_errors, color=colors_q, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, q_errors):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'{val:.4f}m', ha='center', va='bottom', fontsize=9)
        ax3.set_title('Q Process Noise Analysis\n(Avg Position Error)')
        ax3.set_ylabel('Error (m)')
        ax3.grid(True, alpha=0.3, axis='y')

        ax4 = axes[1, 1]
        r_labels = ['R low\n(low noise)', 'R medium\n(med noise)', 'R high\n(high noise)']
        r_errors = self.latest_metrics[3:]
        colors_r = ['#2ecc71', '#f39c12', '#e74c3c']
        bars = ax4.bar(r_labels, r_errors, color=colors_r, edgecolor='black', linewidth=0.5)
        for bar, val in zip(bars, r_errors):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001,
                    f'{val:.4f}m', ha='center', va='bottom', fontsize=9)
        ax4.set_title('R Measurement Noise Analysis\n(Avg Position Error)')
        ax4.set_ylabel('Error (m)')
        ax4.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        filename = f'ekf_analysis_{time.strftime("%H%M%S")}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f'Plot saved as {filename}')
        plt.show()


def main():
    rclpy.init()
    node = EKFPlotter()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    seq_thread = threading.Thread(target=node.run_sequence, daemon=True)
    seq_thread.start()

    seq_thread.join()

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
