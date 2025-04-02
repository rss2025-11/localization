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
        N = odometry.shape[0]
        tiled_odom = np.tile(odometry, (N, 1))
        noise = np.random.normal(self.mu, self.var, (N, 3))

        updated_particles = particles + tiled_odom + noise

        # Normalize angles to [-π, π]
        updated_particles[:, 2] = np.arctan2(
            np.sin(updated_particles[:, 2]), np.cos(updated_particles[:, 2])
        )

        return updated_particles
        ####################################
