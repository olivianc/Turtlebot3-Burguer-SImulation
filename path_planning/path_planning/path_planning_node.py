import rclpy
import rclpy.qos
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path, Odometry
from geometry_msgs.msg import PoseStamped, Point, PointStamped, PoseWithCovarianceStamped
from std_msgs.msg import Header
from visualization_msgs.msg import Marker
import numpy as np
import heapq

class AStarPlanner(Node):
    def __init__(self):
        super().__init__('a_star_planner')

        # Subscripciones
        self.map_sub = self.create_subscription(OccupancyGrid, 'map', self.map_callback, rclpy.qos.qos_profile_sensor_data)
        self.goal_sub = self.create_subscription(PointStamped, 'clicked_point', self.goal_callback, rclpy.qos.qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', self.pose_callback, rclpy.qos.qos_profile_sensor_data)

        # Publicadores
        self.mark_pub = self.create_publisher(Marker, 'mark_planned_path', 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', rclpy.qos.qos_profile_sensor_data)

        # Variables
        self.map_data = None
        self.map_resolution = 0.05
        self.map_origin = (0.0, 0.0)
        self.map_width = 0
        self.map_height = 0
        self.robot_position = None
        self.initial_pose_received = False

    def map_callback(self, msg):
        self.map_data = msg
        self.map_resolution = msg.info.resolution
        self.map_origin = (msg.info.origin.position.x, msg.info.origin.position.y)
        self.map_width = msg.info.width
        self.map_height = msg.info.height
        self.get_logger().info('Mapa recibido.')

    def pose_callback(self, msg):
        if not self.initial_pose_received:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.robot_x = x
            self.robot_y = y
            self.initial_pose_received = True
            self.get_logger().info(f" Posición inicial guardada: ({x:.2f}, {y:.2f})")


    def goal_callback(self, msg):
        self.goal = msg
        self.get_logger().info('Meta recibida.')

        if self.map_data is None:
            self.get_logger().error('No se ha recibido el mapa todavía.')
            return
        
        if self.robot_x is None or self.robot_y is None:
            self.get_logger().error('No se ha recibido la odometría todavía.')
            return
        
        self.plan_path()
        self.get_logger().info('Trayectoria planeada.')

    def plan_path(self):
        #self.get_logger().info('Entrando a plan_path.')

        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin = self.map_data.info.origin


        # Convertir meta a grid
        gx = int((self.goal.point.x - origin.position.x) / resolution)
        gy = int((self.goal.point.y - origin.position.y) / resolution)

        # Convertir posición actual del robot a grid
        sx = int((self.robot_x - self.map_origin[0]) / self.map_resolution)
        sy = int((self.robot_y - self.map_origin[1]) / self.map_resolution)

        self.get_logger().info(f'Robot en grid: ({sx}, {sy}) - Meta en grid: ({gx}, {gy})')

        grid = np.array(self.map_data.data).reshape((height, width))

        # Validar que el inicio y el goal estén en celdas libres
        if grid[sy, sx] > 10:
            self.get_logger().error('La posición inicial del robot está en un obstáculo.')
            return
        
        if grid[gy, gx] > 10:
            self.get_logger().error('El punto de destino está en un obstáculo.')
            return

        def heuristic(a, b):
            return np.linalg.norm(np.array(a) - np.array(b))

        open_set = []
        heapq.heappush(open_set, (0, (sx, sy)))
        came_from = {}
        g_score = {(sx, sy): 0}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if current == (gx, gy):
                break

            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < width and 0 <= neighbor[1] < height:
                    if grid[neighbor[1], neighbor[0]] > 50:
                        continue
                    tentative_g = g_score[current] + 1
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score = tentative_g + heuristic(neighbor, (gx, gy))
                        heapq.heappush(open_set, (f_score, neighbor))

        # Reconstruir el camino
        path = []
        current = (gx, gy)
        while current in came_from:
            path.append(current)
            current = came_from[current]
        path.append((sx, sy))  # agregar el punto inicial
        path.reverse()

        self.publish_path(path, resolution, origin)
        self.get_logger().info(f'Robot va a: {path}')

    def publish_path(self, path, resolution, origin):
        self.get_logger().info('Publicando trayectoria.')

        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "a_star"
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        for (x, y) in path:
            p = Point()
            p.x = origin.position.x + x * resolution
            p.y = origin.position.y + y * resolution
            p.z = 0.0
            marker.points.append(p)

        ros_path = Path()
        ros_path.header.frame_id = "map"
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for (x, y) in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = origin.position.x + x * resolution
            pose.pose.position.y = origin.position.y + y * resolution
            pose.pose.position.z = 0.0
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)
        self.get_logger().info(f'Ruta publicada {ros_path.poses}')
        self.mark_pub.publish(marker)
        #self.get_logger().info(f'Ruta publicada con {len(ros_path.poses)} puntos.')

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()