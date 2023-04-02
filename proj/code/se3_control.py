import numpy as np
from scipy.spatial.transform import Rotation


class SE3Control(object):
    """
    Nonlinear geometric controller
    """

    def __init__(self, quad_params):
        """
        This is the constructor for the SE3Control object. You may instead
        initialize any parameters, control gain values, or private state here.

        For grading purposes the controller is always initialized with one input
        argument: the quadrotor's physical parameters. If you add any additional
        input arguments for testing purposes, you must provide good default
        values!

        Parameters:
            quad_params, dict with keys specified by crazyflie_params.py

        """

        # Quadrotor physical parameters.
        self.mass = quad_params['mass']  # kg
        self.Ixx = quad_params['Ixx']  # kg*m^2
        self.Iyy = quad_params['Iyy']  # kg*m^2
        self.Izz = quad_params['Izz']  # kg*m^2
        self.arm_length = quad_params['arm_length']  # meters
        self.rotor_speed_min = quad_params['rotor_speed_min']  # rad/s
        self.rotor_speed_max = quad_params['rotor_speed_max']  # rad/s
        self.k_thrust = quad_params['k_thrust']  # N/(rad/s)**2
        self.k_drag = quad_params['k_drag']  # Nm/(rad/s)**2

        # You may define any additional constants you like including control gains.
        self.inertia = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))  # kg*m^2
        self.g = 9.81  # m/s^2

        # STUDENT CODE HERE
        self.Kp_x = np.diag(np.array([49, 49, 25]))
        self.Kd_x = np.diag(np.array([14, 14, 10]))
        self.Kr = np.diag(np.array([4900, 4900, 20]))
        self.Kw = np.diag(np.array([140, 140, 6]))

        # self.Kp_x = np.diag(np.array([64, 64, 25]))
        # self.Kd_x = np.diag(np.array([16, 16, 10]))
        # self.Kr = np.diag(np.array([6400, 6400, 20]))
        # self.Kw = np.diag(np.array([160, 160, 6]))

        # self.Kp_x = np.diag(np.array([25, 25, 25]))
        # self.Kd_x = np.diag(np.array([10, 10, 10]))
        # self.Kr = np.diag(np.array([2500, 2500, 20]))
        # self.Kw = np.diag(np.array([100, 100, 6]))

    def update(self, t, state, flat_output):
        """
        This function receives the current time, true state, and desired flat
        outputs. It returns the command inputs.

        Inputs:
            t, present time in seconds
            state, a dict describing the present state with keys
                x, position, m
                v, linear velocity, m/s
                q, quaternion [i,j,k,w]
                w, angular velocity, rad/s
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s

        Outputs:
            control_input, a dict describing the present computed control inputs with keys
                cmd_motor_speeds, rad/s
                cmd_thrust, N (for debugging and laboratory; not used by simulator)
                cmd_moment, N*m (for debugging; not used by simulator)
                cmd_q, quaternion [i,j,k,w] (for laboratory; not used by simulator)
        """
        cmd_motor_speeds = np.zeros((4,))
        cmd_thrust = 0
        cmd_moment = np.zeros((3,))
        cmd_q = np.zeros((4,))

        # STUDENT CODE HERE
        x_target = flat_output.get('x')
        x_dot_target = flat_output.get('x_dot')
        x_ddot_target = flat_output.get('x_ddot')
        x_dddot_target = flat_output.get('x_dddot')
        x_ddddot_target = flat_output.get('x_ddddot')
        yaw_target = flat_output.get('yaw')
        yaw_dot_target = flat_output.get('yaw_dot')

        x = state.get('x')
        vel = state.get('v')
        quaternion = state.get('q')
        ang_vel = state.get('w')

        rot_matrix = self.quaternion_to_rotation_mat(quaternion)  # rotation matrix of present state

        x_ddot_des = x_ddot_target - np.dot(self.Kd_x, (vel - x_dot_target)) \
                     - np.dot(self.Kp_x, (x - x_target))
        F_des = self.mass * x_ddot_des + np.array([0, 0, self.mass * self.g])
        cmd_thrust = np.dot(rot_matrix[:, -1], F_des)

        # calculate desired rotation matrix
        b3_des = F_des / np.linalg.norm(F_des)
        a_psi = np.array([np.cos(yaw_target), np.sin(yaw_target), 0])
        b2_des = np.cross(b3_des, a_psi) / np.linalg.norm(np.cross(b3_des, a_psi))
        b1_des = np.cross(b2_des, b3_des)
        rot_matrix_des = np.hstack((b1_des[:, np.newaxis], b2_des[:, np.newaxis], b3_des[:, np.newaxis]))

        Error_r_mat = 0.5 * (np.dot(rot_matrix_des.T, rot_matrix) - np.dot(rot_matrix.T, rot_matrix_des))
        Error_r = np.array([Error_r_mat[2, 1], -Error_r_mat[2, 0], Error_r_mat[1, 0]])

        # cmd_q = self.SO3_to_R3_quaternion(rot_matrix_des)[1]
        cmd_q = Rotation.from_matrix(rot_matrix_des).as_quat()

        ang_vel_des = 0
        Error_w = ang_vel - ang_vel_des

        # inertia tensor
        I = np.diag(np.array([self.Ixx, self.Iyy, self.Izz]))
        cmd_moment = np.dot(I, -(np.dot(self.Kr, Error_r) + np.dot(self.Kw, Error_w)))

        # calculate the rotor speed through u1 and u2
        mixer = np.array([[self.k_thrust, self.k_thrust, self.k_thrust, self.k_thrust],
                          [0, self.arm_length * self.k_thrust, 0, -self.arm_length * self.k_thrust],
                          [-self.arm_length * self.k_thrust, 0, self.arm_length * self.k_thrust, 0],
                          [self.k_drag, -self.k_drag, self.k_drag, -self.k_drag]])
        tao = np.append(abs(cmd_thrust), cmd_moment)
        w_sq2 = np.dot(np.linalg.inv(mixer), tao)

        # print("w^2 before mask: \n", w_sq2)
        # solve the situation where w^2 is negative
        mask = np.array(w_sq2 > 0)
        w_sq2 = w_sq2 * mask

        # print("w^2 after mask: \n", w_sq2)
        # print("max rotor speed: ", self.rotor_speed_max, "min rotor speed: ", self.rotor_speed_min)
        cmd_motor_speeds = np.sqrt(w_sq2)
        # print("cmd_motor_speeds before trim: ", cmd_motor_speeds)

        # trim the motor speed to within the given range
        for i in range(cmd_motor_speeds.size):
            if cmd_motor_speeds[i] > self.rotor_speed_max:
                cmd_motor_speeds[i] = self.rotor_speed_max
            elif cmd_motor_speeds[i] < self.rotor_speed_min:
                cmd_motor_speeds[i] = self.rotor_speed_min

        # print("cmd_motor_speeds after trim: ", cmd_motor_speeds)

        control_input = {'cmd_motor_speeds': cmd_motor_speeds,
                         'cmd_thrust': cmd_thrust,
                         'cmd_moment': cmd_moment,
                         'cmd_q': cmd_q}
        return control_input

    def quaternion_to_rotation_mat(self, q):
        """
        Converts a quaternion to a rotation matrix

        Input:
            q, quaternion [i,j,k,w]

        Output:
            rotation_matrix, a [3,3] array
        """
        q0 = q[-1]
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]

        r00 = 2 * (q0 ** 2 + q1 ** 2) - 1
        r01 = 2 * (q1 * q2 - q0 * q3)
        r02 = 2 * (q1 * q3 + q0 * q2)
        r10 = 2 * (q1 * q2 + q0 * q3)
        r11 = 2 * (q0 ** 2 + q2 ** 2) - 1
        r12 = 2 * (q2 * q3 - q0 * q1)
        r20 = 2 * (q1 * q3 - q0 * q2)
        r21 = 2 * (q2 * q3 + q0 * q1)
        r22 = 2 * (q0 ** 2 + q3 ** 2) - 1

        rotation_matrix = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])

        return rotation_matrix

    def SO3_to_R3_quaternion(self, matrix):
        """
        Converts a matrix belong to SO(3) to its so(3) vector belong to R^3 and quaternion

        Input: a matrix (3,3)
        Output: the R^3 vector belongs to so(3) (3,),
                the quaternion [i,j,k,w] (4,)
        """
        theta = np.arccos(0.5 * (np.trace(matrix) - 1))  # rotation angle
        n_hat = 1 / (2 * np.sin(theta)) * (matrix - matrix.T)
        n = np.array([n_hat[2, 1], -n_hat[2, 0], n_hat[1, 0]])  # unit vector, the rotation matrix
        quaternion = np.append(np.sin(theta / 2) * n, np.cos(theta / 2))

        return theta * n, quaternion
