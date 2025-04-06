import numpy as np
import rclpy


class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        # noise params
        self.param1 = 0.0
        self.param2 = 0.001  # this is standard deviation
        self.x_noise = None
        self.y_noise = None
        self.theta_noise = None
        self.noise_model = self.gaussian_noise

        # TODO: make the noise params variable on control commands
        ####################################
        node.declare_parameter("deterministic", False)
        node.declare_parameter("k_vel_trans", 0.01)
        node.declare_parameter("k_vel_rot", 0.02)

        self.deterministic = (
            node.get_parameter("deterministic").get_parameter_value().bool_value
        )
        self.k_vel_trans = (
            node.get_parameter("k_vel_trans").get_parameter_value().double_value
        )
        self.k_vel_rot = (
            node.get_parameter("k_vel_rot").get_parameter_value().double_value
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
        self.param_callback = node.add_on_set_parameters_callback(
            self.parameters_callback
        )

        self.node = node

    def parameters_callback(self, params):
        for param in params:
            if param.name == "k_vel_trans":
                self.k_vel_trans = param.value
            elif param.name == "k_vel_rot":
                self.k_vel_rot = param.value
            elif param.name == "deterministic":
                self.deterministic = param.value
        return rclpy.node.SetParametersResult(successful=True)

    def gaussian_noise(self, velocity, k_vel):
        """
        Gaussian distribution for noise. Good for modelling sensor noise, in
        which the sensor tends to report values that are close to the true value.
        """
        # Update variance based on velocities
        std = self.param2 + np.abs((k_vel * velocity))
        return np.random.normal(self.param1, std, size=(self.num_particles,))

    def uniform_noise(self, velocity, k_vel):
        """
        Uniform distribution for noise. Good prior for any noise, can equally
        overestimate or underestimate the true value.
        """
        # Update range based on velocities
        range_size = self.param2 + (k_vel * velocity)
        min_val = self.param1 - range_size / 2
        max_val = self.param1 + range_size / 2
        return np.random.uniform(min_val, max_val, size=(self.num_particles,))

    def exponential_noise(self, velocity, k_vel):
        """
        Exponential distribution for noise. One-sided and decays exponentially.
        Good for modelling wheel-slip, in which the car tends to travel less than
        reported by the odometry.

        param1 is the mean of the exponential distribution.
        param2 is the scale parameter of the exponential distribution.
        """
        # Update scale parameter based on velocities
        scale = self.param2 + np.abs((k_vel * velocity))
        return np.random.exponential(scale, size=(self.num_particles,)) + self.param1

    def update_noise(self, linear_x, linear_y, angular_z):

        self.x_noise = self.noise_model(linear_x, self.k_vel_trans)
        self.y_noise = self.noise_model(linear_y, self.k_vel_trans)
        self.theta_noise = self.noise_model(angular_z, self.k_vel_rot)

    def apply_noise(self, particles):
        """
        Applies translation and rotation noise to particles.
        Uses pre-computed self.x_noise, self.y_noise, and self.theta_noise.
        """
        # If any noises are None, then update with 0 velocities
        if self.x_noise is None or self.y_noise is None or self.theta_noise is None:
            self.update_noise(0, 0, 0)

        # Apply noise component-wise
        particles[:, 0] += self.x_noise  # x noise
        particles[:, 1] += self.y_noise  # y noise
        particles[:, 2] += self.theta_noise  # theta noise

        # Renormalize angles after adding noise
        particles[:, 2] = np.arctan2(np.sin(particles[:, 2]), np.cos(particles[:, 2]))

        return particles

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

        # Multiply odometry by -1 for real world
        odometry = odometry.copy()  # * -1
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
            updated_particles = self.apply_noise(updated_particles)

        return updated_particles

        ####################################
