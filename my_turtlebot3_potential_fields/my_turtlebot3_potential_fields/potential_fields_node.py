#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PointStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import numpy as np
import math
import time

class RobustObstacleAvoidance(Node):
    def __init__(self):
        super().__init__('robust_obstacle_avoidance')

        # Subscribers
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.goal_sub = self.create_subscription(PointStamped, '/clicked_point', self.goal_callback, 10)

        # Publisher
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # Timer
        self.timer = self.create_timer(0.1, self.control_loop)

        # Robot state
        self.position = np.array([0.0, 0.0])
        self.yaw = 0.0
        self.goal = None
        self.scan_data = None
        self.goal_index = 0  

        self.x_in = -1.98
        self.y_in = 0.00
        self.recarga = 1


        # Navigation params
        self.safe_distance = 0.2
        self.max_speed = 0.3
        self.rotation_speed = 0.8
        self.lookahead_distance = 1.0
        self.rate = self.create_rate(10)
        self.time_incio = time.perf_counter()
        

        #Square 
        self.square_pos = []

        # Counter for logging
        self.counter = 0
        
    def goal_callback(self, msg):

        self.x = msg.point.x 
        self.y = msg.point.y 

        self.square_pos.append(self.x)
        self.square_pos.append(self.y)

        
        self.goal = np.array((self.square_pos[self.goal_index],self.square_pos[self.goal_index+1]))


        


    def odom_callback(self, msg):
        self.position = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y])
        q = msg.pose.pose.orientation
        self.yaw = math.atan2(2.0*(q.w*q.z + q.x*q.y), 1.0-2.0*(q.y*q.y + q.z*q.z))
        #self.get_logger().info(f'Odometry -> Posicion: {self.position}, Yaw: {math.degrees(self.yaw):.2f}')

    def scan_callback(self, msg):
        self.scan_data = msg

    def get_data(self):
        return self.scan_data

    def get_Rep_Force(self):
        data = self.get_data()
        lec = np.asarray(data.ranges)
        lec[np.isinf(lec)] = 13.5
        deltaang = (data.angle_max - data.angle_min) / len(data.ranges)
        laserdegs = np.arange(data.angle_min, data.angle_max, deltaang)
        Fx = 0.0
        Fy = 0.0
        for i, deg in enumerate(laserdegs):
            if i >= 360:
                break
            if lec[i] < 1.75:
                Fx += (1 / lec[i]) ** 2 * np.cos(deg)
                Fy += (1 / lec[i]) ** 2 * np.sin(deg)
        Fth = np.arctan2(Fy, Fx) + np.pi
        Fmag = np.linalg.norm((Fx, Fy))
        return Fx, Fy, Fmag, Fth

    def get_Att_Force(self):
        xy = np.array((self.position[0], self.position[1]))
        xycl = np.array((self.goal[0], self.goal[1]))
        euclD = np.linalg.norm(xy - xycl)
        Fatrx = (self.goal[0] - self.position[0]) / euclD
        Fatry = (self.goal[1] - self.position[1]) / euclD
        Fatrth = np.arctan2(Fatry, Fatrx) - self.yaw
        Fmagat = np.linalg.norm((Fatrx, Fatry))
        return Fatrx, Fatry, Fmagat, Fatrth, euclD

    def get_Speed(self, Ftotx, Ftoty, Ftotth):
        speed = Twist()
        if abs(Ftotth) < 0.1:
            speed.linear.x = self.max_speed
            speed.angular.z = 0.0
        else:
            if Ftotth < 0:
                if abs(Ftotth) < np.pi / 2:
                    speed.linear.x = 0.025
                    speed.angular.z = -0.15
                else:
                    speed.linear.x = 0.0
                    speed.angular.z = -0.15
            if Ftotth > 0:
                if abs(Ftotth) < np.pi / 2:
                    speed.linear.x = 0.05
                    speed.angular.z = 0.15
                else:
                    speed.linear.x = 0.0
                    speed.angular.z = 0.15
        return speed

    def control_loop(self):
        

        if self.goal is None or self.scan_data is None:
            return
        goal_dist = np.linalg.norm(self.goal - self.position)
        #self.get_logger().info(f'Distancia al objetivo: {goal_dist:.2f}')


        if goal_dist < self.safe_distance:
            self.get_logger().info(f'Objetivo {self.goal_index+1} alcanzado')

            self.time_fin = time.perf_counter() 

            self.tiempo_total = self.time_fin - self.time_incio
            self.get_logger().info(f'tiempo total -> {self.tiempo_total}')


            # Si hay más puntos en la secuencia, pasar al siguiente
            if self.goal_index < len(self.square_pos)+1:

                if (self.tiempo_total > 50.0) & (self.tiempo_total < 65.0):
                    
                    if self.recarga == 1:
                        self.square_pos.insert(0, self.y_in)
                        self.square_pos.insert(0, self.x_in)
                        self.recarga = 2
                        
                    self.get_logger().info(f'TIEMPO DE RECARGAR!')
                    
                self.goal_index += 1
                self.goal = np.array((self.square_pos.pop(0),self.square_pos.pop(0)))
                self.get_logger().info(f'Siguiente objetivo: {self.goal}')

            else:
                self.stop_robot()
                return


        Fx, Fy, Fmag, Fth = self.get_Rep_Force()
        Fatx, Faty, Fmagat, Fatrth, euclD = self.get_Att_Force()
        Ftotx = Fmag * np.cos(Fth) * 0.005 + Fmagat * np.cos(Fatrth)
        Ftoty = Fmag * np.sin(Fth) * 0.005 + Fmagat * np.sin(Fatrth)
        Ftotth = np.arctan2(Ftoty, Ftotx)
        if Ftotth > np.pi:
            Ftotth = -np.pi - (Ftotth - np.pi)
        if Ftotth < -np.pi:
            Ftotth = Ftotth + 2 * np.pi

        if self.counter == 1000000:
            #self.get_logger().info(f'Ftotxy: {Ftotx}, {Ftoty}, {Ftotth * 180 / np.pi}, euclD: {euclD}')
            self.counter = 0

        speed = self.get_Speed(Ftotx, Ftoty, Ftotth)
        self.cmd_pub.publish(speed)
        self.rate.sleep()

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle



    def stop_robot(self):
        cmd = Twist()
        self.cmd_pub.publish(cmd)
        self.get_logger().info("Objetivo alcanzado - Robot detenido")


def main(args=None):
    rclpy.init(args=args)
    node = RobustObstacleAvoidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
