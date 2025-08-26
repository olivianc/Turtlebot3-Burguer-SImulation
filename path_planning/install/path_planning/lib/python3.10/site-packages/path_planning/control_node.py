import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import Twist
import math


class TrajectoryFollower(Node):
    def __init__(self):
        super().__init__('trajectory_follower')

        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, qos_profile_sensor_data)
        self.path_sub = self.create_subscription(Path, 'planned_path', self.path_callback, qos_profile_sensor_data)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.path = []
        self.current_index = 0

        self.kp_lin = 0.8   # Ganancia proporcional lineal
        self.kp_ang = 3.0   # Ganancia proporcional angular

        self.max_lin = 0.3  # Velocidad lineal máxima (m/s)
        self.max_ang = 1.0  # Velocidad angular máxima (rad/s)

        self.lookahead_distance = 0.3  # Distancia de anticipación (m)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

    def path_callback(self, msg):
        self.path = [(pose.pose.position.x, pose.pose.position.y) for pose in msg.poses]
        self.current_index = 0
        self.get_logger().info(f'Nueva trayectoria recibida con {len(self.path)} puntos.')

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y

        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.theta = math.atan2(siny_cosp, cosy_cosp)

        if self.path:
            self.control()

    def control(self):
        if not self.path:
            return

        # Encuentra el punto más cercano al lookahead_distance
        min_dist = float('inf')
        target_index = None

        for i in range(self.current_index, len(self.path)):
            dx = self.path[i][0] - self.x
            dy = self.path[i][1] - self.y
            dist = math.hypot(dx, dy)
            if abs(dist - self.lookahead_distance) < min_dist:
                min_dist = abs(dist - self.lookahead_distance)
                target_index = i

        if target_index is None:
            self.stop_robot()
            return

        self.current_index = target_index

        target_x, target_y = self.path[target_index]

        dx = target_x - self.x
        dy = target_y - self.y
        distance = math.hypot(dx, dy)
        angle_to_goal = math.atan2(dy, dx)
        angle_error = angle_to_goal - self.theta
        angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

        cmd = Twist()

        cmd.linear.x = self.kp_lin * distance
        cmd.angular.z = self.kp_ang * angle_error

        # Limitar velocidades
        cmd.linear.x = max(-self.max_lin, min(cmd.linear.x, self.max_lin))
        cmd.angular.z = max(-self.max_ang, min(cmd.angular.z, self.max_ang))

        self.cmd_pub.publish(cmd)

        # Parar si estamos en el último punto y muy cerca
        if target_index >= len(self.path) - 1 and distance < 0.05:
            self.stop_robot()

    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.get_logger().info("Trayectoria completada.")


def main(args=None):
    rclpy.init(args=args)
    node = TrajectoryFollower()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
