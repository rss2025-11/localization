from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

assert rclpy

import numpy as np

from threading import Lock

from visualization_msgs.msg import Marker
from tf_transformations import quaternion_from_euler

from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Point


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter("particle_filter_frame", "default")
        self.particle_filter_frame = (
            self.get_parameter("particle_filter_frame")
            .get_parameter_value()
            .string_value
        )

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.

        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("scan_topic", "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(
            LaserScan, scan_topic, self.laser_callback, 1
        )

        self.odom_sub = self.create_subscription(
            Odometry, odom_topic, self.odom_callback, 1
        )

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.pose_callback, 1
        )

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        self.declare_parameter("num_particles", 200)
        self.num_particles = (
            self.get_parameter("num_particles").get_parameter_value().integer_value
        )

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        ###################
        # State Variables
        self.prev_time = self.get_clock().now()
        self.particles_lock = Lock()
        self.particles = np.zeros((self.num_particles, 3))

        self.viz_particle_pub = self.create_publisher(Marker, "/viz/particles", 1)
        self.viz_pose_pub = self.create_publisher(Marker, "/viz/pose_estimate", 1)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.last_scan_process_time = self.get_clock().now()
        self.scan_process_period = 1 / 10  # 10 Hz

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

    def pose_callback(self, pose_msg):
        with self.particles_lock:
            x = pose_msg.pose.pose.position.x
            y = pose_msg.pose.pose.position.y
            quaternion = pose_msg.pose.pose.orientation
            theta = 2 * np.arctan2(quaternion.z, quaternion.w)
            variance = 0.1
            position_noise = np.random.normal(0, variance, (self.num_particles, 2))
            # # TODO: why normal vs gaussian noise?
            heading_noise = np.random.normal(
                0, 0.08 * np.pi / 2, (self.num_particles, 1)
            )
            pose_matrix = np.stack([[x, y, theta] for _ in range(self.num_particles)])
            self.particles = pose_matrix + np.hstack((position_noise, heading_noise))
            # Ensure angles are wrapped to [-π, π]
            self.particles[:, 2] = np.arctan2(
                np.sin(self.particles[:, 2]), np.cos(self.particles[:, 2])
            )

    def laser_callback(self, laser_msg):
        seconds_since_last_scan = (
            self.get_clock().now() - self.last_scan_process_time
        ).nanoseconds / 1e9
        if seconds_since_last_scan < self.scan_process_period:
            return

        self.last_scan_process_time = self.get_clock().now()

        # Convert ranges from array.array to numpy array
        ranges = np.array(laser_msg.ranges)
        probabilities = self.sensor_model.evaluate(self.particles, ranges)

        if probabilities is None:
            # map not up yet
            return

        probabilities /= np.sum(probabilities)

        indices = np.random.choice(
            self.particles.shape[0],
            size=self.particles.shape[0],
            replace=True,
            p=probabilities,
        )
        with self.particles_lock:
            self.particles = self.particles[indices]

        particles_copy = self.particles.copy()
        self.viz_particles(particles_copy)
        self.update_pose_estimate(particles_copy)

    def odom_callback(self, odom_msg):

        curr_time = self.get_clock().now()
        delta_time = (curr_time - self.prev_time).nanoseconds / 1e9

        delta_x = odom_msg.twist.twist.linear.x * delta_time
        delta_y = odom_msg.twist.twist.linear.y * delta_time
        delta_angle = odom_msg.twist.twist.angular.z * delta_time

        odom_robot = np.array([delta_x, delta_y, delta_angle])

        self.motion_model.update_noise(
            odom_msg.twist.twist.linear.x,
            odom_msg.twist.twist.linear.y,
            odom_msg.twist.twist.angular.z,
        )

        with self.particles_lock:
            self.particles = self.motion_model.evaluate(self.particles, odom_robot)

        self.prev_time = curr_time

        particles_copy = self.particles.copy()
        self.viz_particles(particles_copy)
        self.update_pose_estimate(particles_copy)

    def update_pose_estimate(self, particles_copy):
        """
        Aggregate particles into a single pose estimate.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

        returns:
            pose_estimate: A PoseWithCovarianceStamped message
        """

        mean_x, mean_y, mean_theta = self.aggregate_particles(particles_copy)

        # Create odometry message instead of PoseWithCovarianceStamped
        new_pose_estimate = Odometry()
        new_pose_estimate.header.stamp = self.get_clock().now().to_msg()
        new_pose_estimate.header.frame_id = "map"
        new_pose_estimate.child_frame_id = self.particle_filter_frame

        # Set position
        new_pose_estimate.pose.pose.position.x = mean_x
        new_pose_estimate.pose.pose.position.y = mean_y
        new_pose_estimate.pose.pose.position.z = 0.0

        # Convert angle to quaternion
        q = quaternion_from_euler(0, 0, mean_theta)
        new_pose_estimate.pose.pose.orientation.x = q[0]
        new_pose_estimate.pose.pose.orientation.y = q[1]
        new_pose_estimate.pose.pose.orientation.z = q[2]
        new_pose_estimate.pose.pose.orientation.w = q[3]

        # Compute covariance using a fifth of the particles
        sample_size = len(particles_copy) // 5
        sampled_particles = particles_copy[
            np.random.choice(len(particles_copy), sample_size, replace=False)
        ]

        cov = np.zeros((6, 6))
        xy_cov = np.cov(sampled_particles[:, 0:2].T)
        cov[0:2, 0:2] = xy_cov

        # Angular variance using circular statistics
        angles = sampled_particles[:, 2]
        R = np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2)
        angular_var = -2 * np.log(R)  # von Mises circular variance
        cov[5, 5] = angular_var

        new_pose_estimate.pose.covariance = cov.flatten().tolist()

        # Set twist to zero since we don't compute it
        new_pose_estimate.twist.twist.linear.x = 0.0
        new_pose_estimate.twist.twist.linear.y = 0.0
        new_pose_estimate.twist.twist.linear.z = 0.0
        new_pose_estimate.twist.twist.angular.x = 0.0
        new_pose_estimate.twist.twist.angular.y = 0.0
        new_pose_estimate.twist.twist.angular.z = 0.0
        new_pose_estimate.twist.covariance = [0.0] * 36  # 6x6 covariance matrix

        # Publish the new pose estimate
        self.odom_pub.publish(new_pose_estimate)
        self.viz_pose_estimate([mean_x, mean_y, mean_theta])

        # Create and publish transform
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = "map"
        transform.child_frame_id = self.particle_filter_frame

        # Set translation
        transform.transform.translation.x = mean_x
        transform.transform.translation.y = mean_y
        transform.transform.translation.z = 0.0

        # Set rotation
        transform.transform.rotation.x = q[0]
        transform.transform.rotation.y = q[1]
        transform.transform.rotation.z = q[2]
        transform.transform.rotation.w = q[3]

        # Publish transform
        self.tf_broadcaster.sendTransform(transform)

    def aggregate_particles(self, particles):
        """
        Computes the mean pose estimate from particles.
        Uses arithmetic mean for x,y and circular mean for theta.

        args:
            particles: An Nx3 matrix of [x y theta]
        returns:
            mean_x, mean_y, mean_theta: Components of the pose estimate
        """
        # Simple arithmetic mean for x and y
        mean_x = np.mean(particles[:, 0])
        mean_y = np.mean(particles[:, 1])

        # Circular mean for theta using atan2 of mean sin and cos
        mean_sin = np.mean(np.sin(particles[:, 2]))
        mean_cos = np.mean(np.cos(particles[:, 2]))
        mean_theta = np.arctan2(mean_sin, mean_cos)

        return mean_x, mean_y, mean_theta

    def viz_particles(self, particles):
        """Visualize particles using MarkerArray of points for better efficiency"""
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.POINTS
        marker.action = Marker.ADD

        # Set marker properties
        marker.scale.x = 0.05  # Point size
        marker.scale.y = 0.05  # Point size
        marker.color.a = 1.0  # Opacity
        marker.color.r = 0.0  # Make points blue for better visibility
        marker.color.g = 0.0
        marker.color.b = 1.0

        # Only visualize every 5th particle for performance
        for particle in particles[::5]:
            point = Point()
            point.x = particle[0]
            point.y = particle[1]
            point.z = 0.0
            marker.points.append(point)

        self.viz_particle_pub.publish(marker)

    def viz_pose_estimate(self, pose_estimate):
        # Create a marker for the pose estimate
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.id = 0
        marker.type = Marker.ARROW
        marker.action = Marker.ADD

        # Set pose estimate position
        marker.pose.position.x = pose_estimate[0]
        marker.pose.position.y = pose_estimate[1]
        marker.pose.position.z = 0.0

        # Set pose estimate orientation
        q = quaternion_from_euler(0, 0, pose_estimate[2])
        marker.pose.orientation.x = q[0]
        marker.pose.orientation.y = q[1]
        marker.pose.orientation.z = q[2]
        marker.pose.orientation.w = q[3]

        # Set marker properties
        marker.scale.x = 0.2  # Arrow length
        marker.scale.y = 0.1  # Arrow width
        marker.scale.z = 0.1  # Arrow height
        marker.color.a = 1.0  # Fully opaque
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0

        self.viz_pose_pub.publish(marker)

    def param_update_callback(self, params):
        self.motion_model.update_noise(
            params.linear_x,
            params.linear_y,
            params.angular_z,
        )


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
