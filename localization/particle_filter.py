from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan

from rclpy.node import Node
import rclpy

assert rclpy

import tf

import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from threading import Lock


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

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        ###################
        # State Variables
        self.prev_time = self.get_clock().now()
        self.pose_estimate = PoseWithCovarianceStamped()
        self.particles_lock = Lock()
        self.num_particles = 200
        self.particles = np.zeros((self.num_particles, 3))

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
            x = pose_msg.pose.position.x
            y = pose_msg.pose.position.y
            theta = 0
            variance = 1
            position_noise = np.random.normal(0, variance, (self.num_particles, 2))
            heading_noise = np.random.uniform(0, 2*np.pi, (self.num_particles, 1))
            pose_matrix = np.stack([[x, y, theta] for _ in range(self.num_particles)])
            self.particles = pose_matrix + np.hstack(position_noise, heading_noise)

    def laser_callback(self, laser_msg):
        ranges = laser_msg.ranges
        probabilities = self.sensor_model.evaluate(self.particles, ranges)

        with self.particles_lock:
            self.particles = np.random.choice(self.particles, size=self.particles.size, replace=True, p=probabilities)        


    def odom_callback(self, odom_msg):

        curr_time = self.get_clock().now()
        delta_time = curr_time - self.prev_time

        delta_x = odom_msg.twist.twist.linear.x * delta_time
        delta_y = odom_msg.twist.twist.linear.y * delta_time
        delta_angle = odom_msg.twist.twist.angular.z * delta_time

        odom_robot = np.array([delta_x, delta_y, delta_angle])

        # transform odom from robot frame to world frame using current pose estimate
        theta = self.pose_estimate.pose.pose.orientation.z  # get current orientation
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        odom_world = R @ odom_robot

        with self.particles_lock:
            self.particles = self.motion_model.evaluate(self.particles, odom_world)

        self.prev_time = curr_time

        self.update_pose_estimate()

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

        mean_x, mean_y, mean_theta = self.run_k_means(particles_copy)

        # Create pose estimate message
        new_pose_estimate = PoseWithCovarianceStamped()
        new_pose_estimate.header.stamp = self.get_clock().now()
        new_pose_estimate.header.frame_id = "map"

        # Set position
        new_pose_estimate.pose.pose.position.x = mean_x
        new_pose_estimate.pose.pose.position.y = mean_y

        # Convert angle to quaternion
        q = tf.transformations.quaternion_from_euler(0, 0, mean_theta)
        new_pose_estimate.pose.pose.orientation.x = q[0]
        new_pose_estimate.pose.pose.orientation.y = q[1]
        new_pose_estimate.pose.pose.orientation.z = q[2]
        new_pose_estimate.pose.pose.orientation.w = q[3]

        # Compute covariance using all particles
        cov = np.zeros((6, 6))
        xy_cov = np.cov(particles_copy[:, 0:2].T)
        cov[0:2, 0:2] = xy_cov

        # Angular variance using circular statistics
        angles = particles_copy[:, 2]
        R = np.sqrt(np.mean(np.cos(angles)) ** 2 + np.mean(np.sin(angles)) ** 2)
        angular_var = -2 * np.log(R)  # von Mises circular variance
        cov[5, 5] = angular_var

        new_pose_estimate.pose.covariance = cov.flatten().tolist()

        # Publish the new pose estimate
        self.pose_estimate = new_pose_estimate
        self.odom_pub.publish(self.pose_estimate)

    def run_k_means(self, particles):
        """
        Runs k-means clustering on the particles to find the mean pose estimate. Tries k=1,2,3 and chooses the k with the lowest inertia.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

        returns:
            mean_x: The mean x coordinate of the particles
            mean_y: The mean y coordinate of the particles
            mean_theta: The mean theta coordinate of the particles
        """
        angles_2d = np.column_stack((np.cos(particles[:, 2]), np.sin(particles[:, 2])))
        clustering_data = np.column_stack((particles[:, 0:2], angles_2d))

        # Standardize the features for fair comparison
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clustering_data)

        # Try different numbers of clusters
        best_score = float("inf")
        best_kmeans = None

        for k in range(1, 4):  # Try k=1,2,3
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(scaled_data)

            # Normalize inertia by number of clusters to prevent favoring more clusters
            normalized_score = kmeans.inertia_ / k

            if normalized_score < best_score:
                best_score = normalized_score
                best_kmeans = kmeans

        # Get labels for best clustering
        labels = best_kmeans.labels_

        largest_cluster = np.argmax(np.bincount(labels))

        # Transform cluster center back to original scale
        center = scaler.inverse_transform(best_kmeans.cluster_centers_[largest_cluster])
        mean_x = center[0]
        mean_y = center[1]
        mean_theta = np.arctan2(center[3], center[2])

        return mean_x, mean_y, mean_theta


def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
