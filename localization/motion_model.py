import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################

    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        #For odometry, all time calculations are assumed done in the particle_filter.py
        #and the odometry data passed in is in the world frame.

        ####################################
        # TODO

        #odometry asumed to be an np.array
        #TODO Add Noise
        N = odometry.shape[0]
        tiled_odom = np.tile(odometry, (N,1))
        return particles + tiled_odom

        ####################################
