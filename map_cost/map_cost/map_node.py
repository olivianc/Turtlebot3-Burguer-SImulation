import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformListener, Buffer
import numpy as np
import math

MAP_SIZE = 401  # 401x401 grid → centro exacto en (200,200)
RESOLUTION = 0.05  # 5cm por celda
MAX_PROB = 1.0
MIN_PROB = 0.0

class OccupancyGridNode(Node):
    def __init__(self):
        super().__init__('map_node')
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map = np.full((MAP_SIZE, MAP_SIZE), 0.5)  # Desconocido

        # 🔄 Origen centrado en el mundo real
        self.origin_x = -MAP_SIZE // 2
        self.origin_y = -MAP_SIZE // 2

    def scan_callback(self, msg):
        try:
            tf = self.tf_buffer.lookup_transform('base_link', msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"Transform not available: {e}")
            return

        angle = msg.angle_min
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x_lidar = r * math.cos(angle)
                y_lidar = r * math.sin(angle)

                # Transformación a base_link
                dx = tf.transform.translation.x
                dy = tf.transform.translation.y
                yaw = self.get_yaw_from_quaternion(tf.transform.rotation)
                x_robot = math.cos(yaw) * x_lidar - math.sin(yaw) * y_lidar + dx
                y_robot = math.sin(yaw) * x_lidar + math.cos(yaw) * y_lidar + dy

                self.update_map_ray(0.0, 0.0, x_robot, y_robot)

            angle += msg.angle_increment

        self.publish_map()

    def update_map_ray(self, x0, y0, x1, y1):
        def to_cell(x, y):
            i = int((x / RESOLUTION)) + MAP_SIZE // 2
            j = int((y / RESOLUTION)) + MAP_SIZE // 2
            return i, j

        x0_i, y0_i = to_cell(x0, y0)
        x1_i, y1_i = to_cell(x1, y1)

        points = self.bresenham(x0_i, y0_i, x1_i, y1_i)
        for i, j in points[:-1]:
            if 0 <= i < MAP_SIZE and 0 <= j < MAP_SIZE:
                self.map[j, i] = max(self.map[j, i] - 0.1, MIN_PROB)

        i, j = points[-1]
        if 0 <= i < MAP_SIZE and 0 <= j < MAP_SIZE:
            self.map[j, i] = min(self.map[j, i] + 0.1, MAX_PROB)

    def bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x1, y1))
        return points

    def get_yaw_from_quaternion(self, q):
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        return math.atan2(siny_cosp, cosy_cosp)

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.info.resolution = RESOLUTION
        msg.info.width = MAP_SIZE
        msg.info.height = MAP_SIZE
        # 📍 Publicar el origen en coordenadas del mundo
        msg.info.origin.position.x = self.origin_x * RESOLUTION
        msg.info.origin.position.y = self.origin_y * RESOLUTION
        msg.data = [int(p * 100) if 0.0 <= p <= 1.0 else -1 for p in self.map.flatten()]
        self.map_pub.publish(msg)
        
        output = ''
        for row in self.map:
            line = ' '.join([f"{p:.2f}" for p in row])  # Print the raw probabilities
            output += line + '\n'
        self.get_logger().info('\n' + output)

        # Also print the raw probability values in a clearer format
        prob_output = '\n'.join([', '.join([f"{p:.2f}" for p in row]) for row in self.map])
        self.get_logger().info(f"\nProbabilities:\n{prob_output}")


def main():
    rclpy.init()
    node = OccupancyGridNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()