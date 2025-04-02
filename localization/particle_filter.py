from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

assert rclpy

import numpy as np


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        ###################
        # State Variables
        self.prev_time = self.get_clock().now()
        self.pose_estimate = PoseWithCovarianceStamped()

        # TODO: FIGURE OUT HOW TO INITIALIZE PARTICLES


    ###################

    # Implement the MCL algorithm
    # using the sensor model and the motion model
    #
    # Make sure you include some way to initialize
    # your particles, ideally with some sort
    # of interactive interface in rviz
    #
    # Publish a transformation frame between the map
    # and the particle_filter_frame.

    def laser_callback(self, laser_msg):
        ranges = laser_msg.ranges
        probabilities = self.sensor_model.evaluate(self.particles, ranges)

        mask = probabilities > 0.1 # curr arbitrary
        filtered_particles = self.particles[mask]
        filtered_probabilities = probabilities[mask]

        # TODO: add noise as drawn
        self.particles = np.random.choice(filtered_particles, size=len(ranges), replace=True, p=filtered_probabilities)        


    def odom_callback(self, odom_msg):
        
        curr_time = self.get_clock().now()
        delta_time = curr_time - self.prev_time

        delta_x = odom_msg.twist.twist.linear.x * delta_time
        delta_y = odom_msg.twist.twist.linear.y * delta_time
        delta_angle = odom_msg.twist.twist.angular.z * delta_time

        odom_robot = np.array([delta_x, delta_y, delta_angle])

        # transform odom from robot frame to world frame using current pose estimate
        theta = self.pose_estimate.pose.pose.orientation.z  # get current orientation
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ])
        odom_world = R @ odom_robot

        self.particles = self.motion_model.evaluate(self.particles, odom_world)
        
        self.prev_time = curr_time
            



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
