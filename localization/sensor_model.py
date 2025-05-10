import numpy as np
from scan_simulator_2d import PyScanSimulator2D

# Try to change to just `from scan_simulator_2d import PyScanSimulator2D`
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys
import math

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter("map_topic", "default")
        node.declare_parameter("num_beams_per_particle", 1)
        node.declare_parameter("scan_theta_discretization", 1.0)
        node.declare_parameter("scan_field_of_view", 1.0)
        node.declare_parameter("lidar_scale_to_map_scale", 1.0)

        self.map_topic = (
            node.get_parameter("map_topic").get_parameter_value().string_value
        )
        self.num_beams_per_particle = (
            node.get_parameter("num_beams_per_particle")
            .get_parameter_value()
            .integer_value
        )
        self.scan_theta_discretization = (
            node.get_parameter("scan_theta_discretization")
            .get_parameter_value()
            .double_value
        )
        self.scan_field_of_view = (
            node.get_parameter("scan_field_of_view").get_parameter_value().double_value
        )
        self.lidar_scale_to_map_scale = (
            node.get_parameter("lidar_scale_to_map_scale")
            .get_parameter_value()
            .double_value
        )

        self.got_map = False
        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        ####################################
        # Adding state variables; can change eta

        self.eta = 1
        self.d_init = 0
        self.d_max = 200
        self.z_init = 0
        self.z_max = 200
        self.d_vals = np.linspace(self.d_init, self.d_max, self.table_width)
        self.z_vals = np.linspace(self.z_init, self.z_max, self.table_width)

        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization,
        )

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid, self.map_topic, self.map_callback, 1
        )

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

    def find_prob_hit(self, z_k, d):
        mask = (0 <= z_k) & (z_k <= self.z_max)
        return np.where(
            mask,
            1
            / np.sqrt(2 * np.pi * self.sigma_hit**2)
            * np.exp((-((z_k - d) ** 2)) / (2 * self.sigma_hit**2)),
            0,
        )

    def find_prob_short(self, z_k, d):
        mask = (0 <= z_k) & (z_k <= d) & (d != 0)
        # TODO: might be doing too much
        safe_div = lambda x, y: np.divide(
            x, y, where=y != 0, out=np.zeros(y.shape, dtype=np.float64)
        )
        return np.where(mask, 2 * (safe_div(1, d) - safe_div(z_k, d * d)), 0)

    def find_prob_max(self, z_k, d):
        mask = z_k == self.z_max
        return np.where(mask, 1, 0)

    def find_prob_rand(self, z_k, d):
        mask = (0 <= z_k) & (z_k <= self.z_max)
        return np.where(mask, 1 / self.z_max, 0)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        # Initialize tables
        hit_table = np.zeros((self.table_width, self.table_width))
        other_table = np.zeros((self.table_width, self.table_width))

        # For each possible ground truth distance (columns)
        for j, d_val in enumerate(self.d_vals):
            # Create vector of ground truth distances
            d_vec = np.tile(d_val, (1, self.table_width))

            # Calculate hit probabilities for all possible measurements
            hit_probs = self.find_prob_hit(self.z_vals, d_vec)
            hit_table[:, j] = hit_probs  # Fill column j

            # Calculate and combine other probabilities
            short_probs = self.find_prob_short(self.z_vals, d_vec)
            max_probs = self.find_prob_max(self.z_vals, d_vec)
            rand_probs = self.find_prob_rand(self.z_vals, d_vec)

            other_table[:, j] = (
                self.alpha_short * short_probs
                + self.alpha_max * max_probs
                + self.alpha_rand * rand_probs
            )

        # Normalize hit_table across d values (columns)
        d_sums = np.sum(hit_table, axis=0)  # Sum across columns
        # Prevent division by zero by setting the denominator equal to 1 if it was 0
        # (numerator will still be 0 since probabilities are non-negative)
        d_sums[d_sums == 0] = 1
        hit_table /= d_sums[
            np.newaxis, :
        ]  # Normalize each column by dividing it by its sum

        # Combine tables with alpha_hit
        self.sensor_model_table = self.alpha_hit * hit_table + other_table

        # Final normalization only if necessary
        z_sums = np.sum(self.sensor_model_table, axis=0)
        if not np.allclose(z_sums, 1):
            self.sensor_model_table /= z_sums[np.newaxis, :]

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            print("Map not set")
            return

        scans = self.scan_sim.scan(particles)
        scale_factor = self.resolution * self.lidar_scale_to_map_scale
        
        scaled_scans = scans / scale_factor
        clipped_scans = np.clip(scaled_scans, 0, self.z_max)
        
        scaled_observation = observation / scale_factor
        clipped_observation = np.clip(scaled_observation, 0, self.z_max)
        # downsampled_observation = clipped_observation[::self.num_beams_per_particle]
        num_beams = self.num_beams_per_particle
        indices = np.linspace(0, len(clipped_observation) - 1, num_beams).astype(int)
        downsampled_observation = clipped_observation[indices]

        # Get indices for the lookup table - subtract 1 from digitize results since we want 0-based indices
        d_indices = np.digitize(clipped_scans, self.d_vals) - 1
        d_indices = np.clip(d_indices, 0, self.table_width - 1)
        
        z_indices = np.digitize(downsampled_observation, self.z_vals) - 1
        z_indices = np.clip(z_indices, 0, self.table_width - 1)
        
        # Vectorized lookup using broadcasting
        probabilities_table = self.sensor_model_table[z_indices[:, np.newaxis], d_indices.T]
        
        return np.prod(probabilities_table.T, axis=1)

    def map_callback(self, map_msg):
        if self.got_map:
            return
        
        self.got_map = True
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.0
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution  # number pixels per meter

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion(
            (origin_o.x, origin_o.y, origin_o.z, origin_o.w)
        )
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5,
        )  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
