import rclpy
import rclpy.qos
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, PointStamped, Point, PoseWithCovarianceStamped
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
        self.initialpose_sub = self.create_subscription(PoseWithCovarianceStamped, 'initialpose', self.initialpose_callback, rclpy.qos.qos_profile_sensor_data)

        # Publicadores
        self.mark_pub = self.create_publisher(Marker, 'mark_planned_path', 10)
        self.path_pub = self.create_publisher(Path, 'planned_path', rclpy.qos.qos_profile_sensor_data)

        # Variables
        self.map_data = None
        self.goal = None
        self.start_x = None
        self.start_y = None

    def map_callback(self, msg):
        self.map_data = msg
        self.get_logger().info('Mapa recibido.')

    def initialpose_callback(self, msg):
        self.start_x = msg.pose.pose.position.x
        self.start_y = msg.pose.pose.position.y
        self.get_logger().info('Posición inicial recibida.')

    def goal_callback(self, msg):
        self.goal = msg
        self.get_logger().info('Meta recibida.')

        if self.map_data is None:
            self.get_logger().error('No se ha recibido el mapa todavía.')
            return
        
        if self.start_x is None or self.start_y is None:
            self.get_logger().error('No se ha recibido la posición inicial todavía.')
            return
        
        self.plan_path()
        self.get_logger().info('Trayectoria planeada.')

    def plan_path(self):
        self.get_logger().info('Entrando a plan_path.')

        width = self.map_data.info.width
        height = self.map_data.info.height
        resolution = self.map_data.info.resolution
        origin = self.map_data.info.origin

        # Usar directamente las coordenadas crudas
        gx = self.goal.point.x
        gy = self.goal.point.y

        sx = self.start_x
        sy = self.start_y

        self.get_logger().info(f'Posición inicial: ({sx}, {sy}) - Meta: ({gx}, {gy})')

        grid = np.array(self.map_data.data).reshape((height, width))

        # Validar que el inicio y el goal estén en celdas libres
        if grid[int(sy / resolution), int(sx / resolution)] > 50:
            self.get_logger().error('La posición inicial está en un obstáculo.')
            return
        
        if grid[int(gy / resolution), int(gx / resolution)] > 50:
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

            for dx, dy in [(-resolution, 0), (resolution, 0), (0, -resolution), (0, resolution)]:
                neighbor = (current[0] + dx, current[1] + dy)
                if 0 <= neighbor[0] < width * resolution and 0 <= neighbor[1] < height * resolution:
                    if grid[int(neighbor[1] / resolution), int(neighbor[0] / resolution)] > 50:
                        continue
                    tentative_g = g_score[current] + resolution
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
            p.x = x
            p.y = y
            p.z = 0.0
            marker.points.append(p)

        ros_path = Path()
        ros_path.header.frame_id = "map"
        ros_path.header.stamp = self.get_clock().now().to_msg()

        for (x, y) in path:
            pose = PoseStamped()
            pose.header.frame_id = "map"
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = 0.0
            ros_path.poses.append(pose)

        self.path_pub.publish(ros_path)
        self.mark_pub.publish(marker)
        self.get_logger().info(f'Ruta publicada con {len(ros_path.poses)} puntos.')

def main(args=None):
    rclpy.init(args=args)
    node = AStarPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
