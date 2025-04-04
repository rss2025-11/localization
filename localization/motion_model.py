import numpy as np
import rclpy

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        # noise params
        self.mu = 0
        self.var = 0.1

        # TODO: make the noise params variable on control commands
        ####################################
        node.declare_parameter("deterministic", False)
        node.declare_parameter("k_vel_trans", 0.001)
        node.declare_parameter("k_vel_rot", 0.002)
        node.declare_parameter("var", 0.1)

        self.deterministic = (
            node.get_parameter("deterministic").get_parameter_value().bool_value
        )
        try:
            self.num_particles = (
                node.get_parameter("num_particles").get_parameter_value().integer_value
            )
        except:
            node.declare_parameter("num_particles", 1)
            self.num_particles = (
                node.get_parameter("num_particles").get_parameter_value().integer_value
            )

        # Set up parameter callback
        self.param_callback = node.add_on_set_parameters_callback(self.parameters_callback)

        self.node = node

    def parameters_callback(self, params):
        for param in params:
            if param.name == "k_vel_trans":
                self.k_vel_trans = param.value
            elif param.name == "k_vel_rot":
                self.k_vel_rot = param.value
            elif param.name == "var":
                self.var = param.value
            elif param.name == "deterministic":
                self.deterministic = param.value
        return rclpy.node.SetParametersResult(successful=True)

    def update_noise(self, linear_x, linear_y, angular_z):
        translation_velocity = np.sqrt(linear_x**2 + linear_y**2)
        rotation_velocity = angular_z

        self.k_vel_trans = self.node.get_parameter("k_vel_trans").get_parameter_value().double_value
        self.k_vel_rot = self.node.get_parameter("k_vel_rot").get_parameter_value().double_value
        self.var = self.node.get_parameter("var").get_parameter_value().double_value

        self.var += (self.k_vel_trans * translation_velocity + self.k_vel_rot * rotation_velocity)

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        # For odometry, all time calculations are assumed done in the particle_filter.py
        # and the odometry data passed in is in the world frame.

        ####################################
        # odometry asumed to be an np.array

        # transform odom from robot frame to world frame using current pose estimate

        x_particles = particles[:, 0]
        y_particles = particles[:, 1]
        thetas = particles[:, 2]

        # Create rotation matrices for each particle's theta
        cos_thetas = np.cos(thetas)
        sin_thetas = np.sin(thetas)

        # Rotate odometry translation
        dx_world = cos_thetas * odometry[0] - sin_thetas * odometry[1]
        dy_world = sin_thetas * odometry[0] + cos_thetas * odometry[1]

        # Compute new positions
        x_world = x_particles + dx_world
        y_world = y_particles + dy_world
        theta_world = thetas + odometry[2]

        # Clip angles to [-π, π]
        theta_world = np.arctan2(np.sin(theta_world), np.cos(theta_world))

        # Stack results to get the new particles in world coordinates
        updated_particles = np.column_stack((x_world, y_world, theta_world))

        if not self.deterministic:
            # Add noise
            N = particles.shape[0]
            noise = np.random.normal(self.mu, np.sqrt(self.var), (N, 3))
            updated_particles += noise

            # Renormalize angles after adding noise
            updated_particles[:, 2] = np.arctan2(
                np.sin(updated_particles[:, 2]), np.cos(updated_particles[:, 2])
            )

        return updated_particles
        ####################################
