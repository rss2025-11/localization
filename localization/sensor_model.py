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
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

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

        self.d_vals = np.linspace(0, self.table_width-1, self.table_width)
        self.z_vals = np.linspace(0, self.table_width-1, self.table_width)
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)



        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()
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


        p_hit = lambda z_k, d : self.eta/np.sqrt(2*np.pi*self.sigma_hit**2)* np.exp((-(z_k - d)**2)/(2*self.sigma_hit**2)) if np.all((0<=z_k) & (z_k<=self.z_max)) else 0
        p_short = lambda z_k, d : 2/d*(1 - z_k/d) if np.all((0 <= z_k) & (z_k <= d) & (d != 0)) else 0
        p_max = lambda z_k, d : 1 if np.all(z_k == self.z_max) else 0
        p_rand = lambda z_k, d : 1/self.z_max if np.all((0<=z_k) & (z_k <= self.z_max)) else 0

        for i in range(len(self.d_vals)):
            d_vec = np.tile(self.d_vals[i], (1, self.table_width))
            p_h = p_hit(self.z_vals, d_vec)
            p_h /= np.sum(p_h) #Normalizing p_hit
            p_s = p_short(self.z_vals, d_vec)
            p_m = p_max(self.z_vals, d_vec)
            p_r = p_rand(self.z_vals, d_vec)
            p = self.alpha_hit * p_h + self.alpha_short * p_s + self.alpha_max*p_m + self.alpha_rand*p_r
            p /= np.sum(p) #Normalizing column
            self.sensor_model_table[i, :] = p #If not row vector, get Transpose

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
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        # simulation is ground truth; d
        # what car is seeing; array of all distances
        scans = self.scan_sim.scan(particles) #The ray-tracing done for us
        scans /= (self.resolution * self.lidar_scale_to_map_scale) # convert scan results from meters to pixels
        # z_max is defined in pixels
        clipped_scans = np.clip(scans, 0, self.z_max) # convert scan results from meters to pixels


        # simulated scans given a particle; z_k
        observation /= (self.resolution * self.lidar_scale_to_map_scale) # convert observation results from meters to pixels
        clipped_observation = np.clip(observation, 0, self.z_max)

        # for each particle, we are determining how likely the observation is based on the scan
        probabilities = np.empty(len(particles))

        print("Made to line 169")
        for i in range(len(clipped_scans)):
            # find probability of observation given particle_scan
            particle_scan = clipped_scans[i]
            print("Made to line 173")
            p = 1
            for j in range(len(particle_scan)):
                ray_truth = particle_scan[j]
                d_index = (np.abs(self.d_vals - ray_truth)).argmin()
                z_index= (np.abs(self.z_vals - clipped_observation[j])).argmin()
                p *= self.sensor_model_table[z_index][d_index]
                print("Made to line 180")


            probabilities[j] = p
        
        return probabilities

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution # number pixels per meter

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")
