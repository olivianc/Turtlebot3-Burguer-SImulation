import rclpy
import rclpy.qos
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from math import atan2, hypot, pi
import numpy as np
from tf_transformations import euler_from_quaternion

class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        self.declare_parameter('linear_speed', 0.15)
        self.declare_parameter('k_p', 0.5)
        self.declare_parameter('k_i', 0.0)
        self.declare_parameter('k_d', 0.1)

        self.linear_speed = self.get_parameter('linear_speed').value
        self.k_p = self.get_parameter('k_p').value
        self.k_i = self.get_parameter('k_i').value
        self.k_d = self.get_parameter('k_d').value

        self.path_sub = self.create_subscription(Path, 'planned_path', self.path_callback, rclpy.qos.qos_profile_sensor_data)
        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.pose_callback, rclpy.qos.qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, rclpy.qos.qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.path_points = []
        self.robot_pose = None
        self.scan_data = None

        self.error_integral = 0.0
        self.prev_error = 0.0

        self.timer = self.create_timer(0.1, self.control_loop)

    def path_callback(self, msg):
        self.path_points = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.get_logger().info(f'Trayectoria recibida con {len(self.path_points)} puntos.')

    def pose_callback(self, msg):
        pos = msg.pose.pose.position
        ori = msg.pose.pose.orientation
        # Convertir cuaternión a ángulo yaw
        (_, _, yaw) = euler_from_quaternion([ori.x, ori.y, ori.z, ori.w])
        self.robot_pose = (pos.x, pos.y, yaw)

    def scan_callback(self, msg):
        self.scan_data = msg.ranges

    def control_loop(self):
        if not self.path_points or not self.robot_pose or not self.scan_data:
            return

        robot_x, robot_y, robot_yaw = self.robot_pose

        # Encontrar punto objetivo más cercano
        distances = [hypot(px - robot_x, py - robot_y) for px, py in self.path_points]
        min_index = np.argmin(distances)

        if min_index >= len(self.path_points) - 1:
            self.get_logger().info('Ruta finalizada.')
            self.cmd_pub.publish(Twist())  # Detener robot
            return

        target_x, target_y = self.path_points[min_index + 1]

        # Error transversal (ángulo entre orientación actual y punto objetivo)
        angle_to_target = atan2(target_y - robot_y, target_x - robot_x)
        error = angle_to_target - robot_yaw

        # Normalizar error entre -pi y pi
        error = (error + pi) % (2 * pi) - pi

        # PID
        self.error_integral += error * 0.1
        error_derivative = (error - self.prev_error) / 0.1
        angular_z = self.k_p * error + self.k_i * self.error_integral + self.k_d * error_derivative
        angular_z = max(min(angular_z, 1.0), -1.0)

        # Evitar obstáculos
        min_distance = min(self.scan_data)
        if min_distance < 0.7:  # Umbral de distancia para evitar obstáculos
            angular_z = 1.0 if angular_z > 0 else -1.0
            self.linear_speed = 0.05  # Reducir la velocidad lineal en lugar de detenerse
        else:
            self.linear_speed = self.get_parameter('linear_speed').value

        self.prev_error = error

        # Publicar velocidades
        cmd = Twist()
        cmd.linear.x = self.linear_speed
        cmd.angular.z = angular_z
        self.cmd_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollower()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
