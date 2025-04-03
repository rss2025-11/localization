import numpy as np


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
        node.declare_parameter("num_particles", 200)
        node.declare_parameter("deterministic", False)

        self.deterministic = node.get_parameter("deterministic").get_parameter_value().bool_value
        self.num_particles = node.get_parameter("num_particles").get_parameter_value().integer_value
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
        theta_world = np.arctan2(
            np.sin(theta_world), np.cos(theta_world)
        )

        # Stack results to get the new particles in world coordinates
        updated_particles = np.column_stack((x_world, y_world, theta_world))
    
        return updated_particles
        ####################################
