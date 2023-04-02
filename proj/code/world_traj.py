from copy import deepcopy

import numpy as np
import scipy

from .graph_search import graph_search


class WorldTraj(object):
    """
    Trajectory planner
    """

    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!

        # self.resolution = np.array([0.1, 0.1, 0.1])
        # self.resolution = np.array([0.15, 0.15, 0.15])
        # self.resolution = np.array([0.2, 0.2, 0.2])
        # self.resolution = np.array([0.25, 0.25, 0.25])
        self.resolution = np.array([0.35, 0.35, 0.35])
        self.margin = 0.35

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, self.nodes_num = graph_search(world, self.resolution, self.margin, start, goal, astar=True)
        self.world = world

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.

        self.points = np.zeros((1, 3))  # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        # self.points = self.path[3:-3:6, :]
        # self.points = np.vstack((self.path[0, :], self.points, self.path[-1, :]))  # make sure contain the goal point

        # self.points = self.sparse_waypoints(self.path, 6)  # this does not work well!

        # # self.interval = 10
        # self.interval = int(self.nodes_num / 22)
        # if self.nodes_num % self.interval >= self.interval * 2 / 3:
        #     self.points = self.path[:-1:self.interval, :]
        # elif self.nodes_num % self.interval < self.interval * 2 / 3:
        #     self.points = self.path[:int(np.floor(-self.interval * 2 / 3 - 1)):self.interval, :]
        #
        # # self.points = self.path[:-1:8, :]
        # self.points = np.vstack((self.points, self.path[-1, :]))  # make sure contain the goal point

        self.points = self.sparse_waypoints2()

        self.seg_num = self.points.shape[0] - 1  # number of trajectory segments
        self.time = self._allocate_time(self.points,
                                        scale_factor=2.8)  # time to run through each trajectory segments, shape: (m_pts - 1,)
        self.time_point = np.append(0, np.cumsum(self.time))  # time point at each waypoint, shape: (m_pts,)
        self.A, self.p = self._init_constraint(self.time, self.points, minimize_method="snap")
        c = scipy.linalg.solve(self.A, self.p)
        self.coeff = np.reshape(c, (-1, 8, 3))  # using "snap" minimization

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x = np.zeros((3,))
        x_dot = np.zeros((3,))
        x_ddot = np.zeros((3,))
        x_dddot = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE
        if t >= self.time_point[-1]:
            x_dot = np.zeros((3,))
            x = self.points[-1, :]
        else:
            for i in range(self.time_point.size - 1):
                if self.time_point[i] <= t < self.time_point[i] + self.time[i]:
                    t = t - self.time_point[i]
                    T0 = np.array([pow(t, 7), pow(t, 6), pow(t, 5), pow(t, 4), pow(t, 3), pow(t, 2), t, 1])
                    T1 = np.array([pow(t, 6), pow(t, 5), pow(t, 4), pow(t, 3), pow(t, 2), t, 1, 0])
                    T2 = np.array([pow(t, 5), pow(t, 4), pow(t, 3), pow(t, 2), t, 1, 0, 0])
                    T3 = np.array([pow(t, 4), pow(t, 3), pow(t, 2), t, 1, 0, 0, 0])
                    T4 = np.array([pow(t, 3), pow(t, 2), t, 1, 0, 0, 0, 0])
                    c1 = np.array([7, 6, 5, 4, 3, 2, 1, 0])
                    c2 = np.array([42, 30, 20, 12, 6, 2, 0, 0])
                    c3 = np.array([210, 120, 60, 24, 6, 0, 0, 0])
                    c4 = np.array([840, 360, 120, 24, 0, 0, 0, 0])
                    x = T0 @ self.coeff[i]
                    x_dot = T1 * c1 @ self.coeff[i]
                    x_ddot = T2 * c2 @ self.coeff[i]
                    x_dddot = T3 * c3 @ self.coeff[i]
                    x_ddddot = T4 * c4 @ self.coeff[i]

        flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                       'yaw': yaw, 'yaw_dot': yaw_dot}
        return flat_output

    def _init_constraint(self, time, points, minimize_method="snap"):
        """
        Construct the constraint matrix of A for and the trajectory point vector p for solve the
        all segments stacked polynomial coefficients vector c.
        A * c = p

        Parameters:
            time: a vector of shape (m_pts - 1, )
            points: pruned trajectory points of shape (m_pts, 3)
            minimize_method: default to be "snap" minimization, can also use "jerk".
        Output:
            A: full rank constraint matrix to solve for coefficients vector c
               shape: (8 * (m_pts - 1), 8 * (m_pts - 1)) --- "snap"
                      (6 * (m_pts - 1), 6 * (m_pts - 1)) --- "jerk"
            p: trajectory point vector
        """
        if minimize_method == "snap":
            # end boundary condition 2 * n (snap, n = 4)
            # start point
            a = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
                          [0, 0, 0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 2, 0, 0],
                          [0, 0, 0, 0, 3, 0, 0, 0]])
            a = np.hstack((a, np.zeros((4, 8 * (time.shape[0] - 1)))))
            p = np.vstack((points[0, :], np.zeros((3, 3))))
            # end point
            b = np.array([[pow(time[-1], 7), pow(time[-1], 6), pow(time[-1], 5), pow(time[-1], 4), pow(time[-1], 3),
                           pow(time[-1], 2), time[-1], 1],
                          [7 * pow(time[-1], 6), 6 * pow(time[-1], 5), 5 * pow(time[-1], 4), 4 * pow(time[-1], 3),
                           3 * pow(time[-1], 2), 2 * time[-1], 1, 0],
                          [42 * pow(time[-1], 5), 30 * pow(time[-1], 4), 20 * pow(time[-1], 3), 12 * pow(time[-1], 2),
                           6 * time[-1], 2, 0, 0],
                          [210 * pow(time[-1], 4), 120 * pow(time[-1], 3), 60 * pow(time[-1], 2), 24 * time[-1], 6, 0,
                           0, 0]])
            b = np.hstack((np.zeros((4, 8 * (time.shape[0] - 1))), b))
            end_boundary = np.vstack((a, b))
            p = np.vstack((p, points[-1, :], np.zeros((3, 3))))

            # intermediate position constraint 2 * (m - 1)
            intermediate_pos = np.array([])
            for i in range(0, time.shape[0] - 1):
                zeros_before = np.zeros((2, 8 * i))
                intermediate = \
                    np.array([[pow(time[i], 7), pow(time[i], 6), pow(time[i], 5), pow(time[i], 4), pow(time[i], 3),
                               pow(time[i], 2), time[i], 1, 0, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
                zeros_after = np.zeros((2, 8 * (time.shape[0] - i - 2)))
                intermediate = np.hstack((zeros_before, intermediate, zeros_after))
                intermediate_pos = np.vstack(
                    (intermediate_pos, intermediate)) if intermediate_pos.size else intermediate
                p = np.vstack((p, points[i + 1, :], points[i + 1, :]))

            # continuity constraints 2 * (n - 1) * (m - 1)
            continuity_constraint = np.array([])
            for i in range(0, time.shape[0] - 1):
                zeros_before = np.zeros((6, 8 * i))
                continuity_1 = \
                    np.array([[7 * pow(time[i], 6), 6 * pow(time[i], 5), 5 * pow(time[i], 4), 4 * pow(time[i], 3),
                               3 * pow(time[i], 2), 2 * time[i], 1, 0],
                              [42 * pow(time[i], 5), 30 * pow(time[i], 4), 20 * pow(time[i], 3), 12 * pow(time[i], 2),
                               6 * time[i], 2, 0, 0],
                              [210 * pow(time[i], 4), 120 * pow(time[i], 3), 60 * pow(time[i], 2), 24 * time[i], 6, 0,
                               0, 0],
                              [840 * pow(time[i], 3), 360 * pow(time[i], 2), 120 * time[i], 24, 0, 0, 0, 0],
                              [2520 * pow(time[i], 2), 720 * time[i], 120, 0, 0, 0, 0, 0],
                              [5040 * time[i], 720, 0, 0, 0, 0, 0, 0]])
                continuity_2 = \
                    np.array([[0, 0, 0, 0, 0, 0, -1, 0],
                              [0, 0, 0, 0, 0, -2, 0, 0],
                              [0, 0, 0, 0, -6, 0, 0, 0],
                              [0, 0, 0, -24, 0, 0, 0, 0],
                              [0, 0, -120, 0, 0, 0, 0, 0],
                              [0, -720, 0, 0, 0, 0, 0, 0]])
                zeros_after = np.zeros((6, 8 * (time.shape[0] - i - 2)))
                continuity = np.hstack((zeros_before, continuity_1, continuity_2, zeros_after))
                continuity_constraint = np.vstack(
                    (continuity_constraint, continuity)) if continuity_constraint.size else continuity
                p = np.vstack((p, np.zeros((6, 3))))

            # stack all the constraints together
            A = np.vstack((end_boundary, intermediate_pos, continuity_constraint))

            return A, p

    def _allocate_time(self, points, scale_factor):
        """
        Allocate time for each trajectory segments based on distance between trajectory points.

        Parameter:
            points, all the trajectory points of shape (m_pts, 3)
            scale_factor, to adjust the time, approx speed m/s
        Output:
            time, allocated time for all segments
        """
        dist = np.linalg.norm(points[1:, :] - points[:-1, :], axis=1)
        time = dist * 1 / scale_factor
        time[-1] = time[-1] * 2  # try to suppress saturation, allocate the last segment longer time
        time[0] = time[0] * 2
        return time

    def sparse_waypoints(self, points, increment):
        """
        Pick waypoints that not collinear to form sparse waypoints.

        Parameter:
            points, dense waypoints acquired by path searching algorithm
            increment, the increment value of points sequence for selection
        Output:
            sparse_waypoints, screened sparse waypoints from collineation
        """
        # all segments formed by dense waypoints
        segments = points[1:, :] - points[:-1, :]
        segments_len = np.linalg.norm(segments, axis=1)  # each segment length
        segments_dir = segments / segments_len[:, np.newaxis]  # each segment direction
        sparse_waypoints = [points[0, :]]
        slope = 0
        for i in range(int(increment / 2), points.shape[0] - 2, increment):
            if np.linalg.norm(segments_dir[i] - slope) > 0.18:  # approx 10 degree
                slope = segments_dir[i]
                sparse_waypoints.append(points[i + 1])
        sparse_waypoints.append(points[-1, :])  # include the end point anyway
        sparse_waypoints = np.array(sparse_waypoints)
        return sparse_waypoints

    def sparse_waypoints2(self):

        # connect from start points ahead as far as possible
        cur = 0
        keep_dist = 1
        nodes_num = self.nodes_num
        delete = []
        while cur < nodes_num - 1:
            search = nodes_num - 1
            while search > cur:
                if len(self.world.path_collisions(self.path[[cur, search]], self.margin)) == 0:
                    # without collision
                    checked_points = self.path[cur + 1:search]
                    _, closest_dist = self.world.closest_points(checked_points)
                    # removed index (keep those within keep_dist to avoid intersection of the traj with occupied grid)
                    rm_idx = np.where(closest_dist > keep_dist)[0] + cur + 1
                    delete = delete + rm_idx.tolist()
                    cur = search
                else:
                    search -= 1
        delete = list(set(delete))

        # sparse points again
        new_pts = np.delete(self.path, delete, axis=0)
        new_delete = set()
        last = 0
        cur = 1
        next = 2
        while next < len(new_pts):
            dist1 = np.linalg.norm(new_pts[cur] - new_pts[last])
            dist2 = np.linalg.norm(new_pts[cur] - new_pts[next])
            min_dist = np.min([dist1, dist2])
            if min_dist < 0.5:
                if len(self.world.path_collisions(new_pts[[last, next]], self.margin)) == 0:
                    # without collision
                    new_delete.add(cur)
                    cur = next
                    next += 1
                else:
                    last = cur
                    cur = next
                    next += 1
            else:
                last = cur
                cur = next
                next += 1
        new_delete = list(new_delete)
        new_pts = np.delete(new_pts, new_delete, axis=0)
        kept = []
        for i in range(len(new_pts)):
            kept.append(np.where(np.all(self.path == new_pts[i], axis=1))[0].item())
        delete = list(set(range(len(self.path))) - set(kept))

        # add necessary points back
        d = deepcopy(delete)
        new_pts = np.delete(self.path, delete, axis=0)
        for i in d:
            dist = np.linalg.norm(new_pts - self.path[i], axis=1)
            min_dist = np.min(dist)
            if min_dist > 0.9:
                delete.remove(i)
                new_pts = np.delete(self.path, delete, axis=0)
        return new_pts
