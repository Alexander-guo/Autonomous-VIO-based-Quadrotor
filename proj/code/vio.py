#%% Imports

import numpy as np
from numpy.linalg import inv
from scipy.spatial.transform import Rotation


#%% Functions

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    new_p = np.zeros((3, 1))
    new_v = np.zeros((3, 1))
    new_q = Rotation.identity()
    R = q.as_matrix()

    new_p = p + v * dt + 1 / 2 * (R @ (a_m - a_b) + g) * (dt ** 2)
    new_v = v + (R @ (a_m - a_b) + g) * dt
    new_q = q * Rotation.from_rotvec(np.squeeze(w_m - w_b) * dt)

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    # construct the covariance matrix of the perturbation impulses
    V_i = (accelerometer_noise_density * dt) ** 2 * np.ones(3)
    Theta_i = (gyroscope_noise_density * dt) ** 2 * np.ones(3)
    A_i = accelerometer_random_walk ** 2 * dt * np.ones(3)
    Omega_i = gyroscope_random_walk ** 2 * dt * np.ones(3)
    Q_i = np.hstack((V_i, Theta_i, A_i, Omega_i))
    Q_i = np.diag(Q_i)  # covariance matrix of the perturbation impulse (12 x 12)

    # Jacobian wrt perturbation impulses
    F_i = np.vstack((np.zeros((3, 12)), np.identity(12), np.zeros((3, 12))))

    # Jacobian wrt error states
    F_x = jacobian_wrt_error_states(dt, a_b, w_b, a_m, w_m, q)

    # prediction update of the error state covariance
    error_state_covariance = F_x @ error_state_covariance @ F_x.T + F_i @ Q_i @ F_i.T
    # return an 18x18 covariance matrix
    return error_state_covariance


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    innovation = np.zeros((2, 1))

    # predicted coordinates in camera frame
    R = q.as_matrix()

    # u_pix = uv[0, 0]
    # v_pix = uv[1, 0]
    Pc = R.T @ (Pw - p)
    Zc = Pc[-1, 0]  # predicted depth

    # get the innovation and the norm
    innovation = uv - Pc[:2, :] / Zc  # 2x1
    norm_innov = np.linalg.norm(innovation)

    # reject the outlier
    if norm_innov < error_threshold:
        diff_Pc_delta_theta = vec2skew_symmetric(Pc)  # partial differentiate of Pc to delta_theta
        diff_Pc_delta_p = - R.T  # partial differentiate of Pc to delta_p

        # diff_z_Pc = 1 / Zc * np.array([[1, 0, -u_pix],
        #                                [0, 1, -v_pix]], dtype=np.float64)

        # partial differentiate of measured uv to predicted coordinates in camera frame
        diff_z_Pc = 1 / Zc * np.array([[1, 0, - Pc[0, 0] / Pc[-1, 0]],
                                       [0, 1, - Pc[1, 0] / Pc[-1, 0]]],
                                      dtype=np.float64)  # TODO: use this one, pass all the test

        diff_z_delta_theta = diff_z_Pc @ diff_Pc_delta_theta  # 2x3
        diff_z_delta_p = diff_z_Pc @ diff_Pc_delta_p  # 2x3
        zero_block = np.zeros((2, 3))

        # measurement Jacobian wrt the error state
        Ht = np.hstack((diff_z_delta_p, zero_block, diff_z_delta_theta, zero_block, zero_block, zero_block))  # 2x18

        error_state_covariance = error_state_covariance.astype(np.float64)

        # kalman gain
        Kt = error_state_covariance @ Ht.T @ np.linalg.inv(Ht @ error_state_covariance @ Ht.T + Q)  # 18x2
        I = np.identity(18)

        # update covariance
        error_state_covariance = (I - Kt @ Ht) @ error_state_covariance @ (I - Kt @ Ht).T + Kt @ Q @ Kt.T

        # # updated error states
        error_state = Kt @ innovation  # 18x1

        # update nominal state with error state
        p, v, q, a_b, w_b, g = update_nominal_state_with_error_state(error_state, nominal_state)

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation


def jacobian_wrt_error_states(dt, a_b, w_b, a_m, w_m, q):
    """
    Construct the Jacobian matrix wrt all the error states in dynamics step.
    @param dt: duration of time interval since last update in seconds
    @param a_b: 3x1 vector - accelerometer bias
    @param w_b: 3x1 vector - gyroscope bias
    @param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    @param w_m: 3x1 vector - measured angular velocity in radians per second
    @param q: orientation of the robot, a Rotation object
    @return: jacobian matrix 18x18
    """
    R = q.as_matrix()
    F_x = np.identity(18)
    F_x[0:3, 3:6] = np.identity(3) * dt
    F_x[3:6, 15:] = np.identity(3) * dt
    F_x[6:9, 12:15] = - np.identity(3) * dt
    F_x[3:6, 9:12] = - R * dt

    a_t = a_m - a_b
    # a_t_skewsymmetric = np.array([[0, -a_t[-1], a_t[1]],
    #                               [a_t[-1], 0, -a_t[0]],
    #                               [-a_t[1], a_t[0], 0]], dtype=float)
    a_t_skewsymmetric = vec2skew_symmetric(a_t)
    F_x[3:6, 6:9] = - R @ a_t_skewsymmetric * dt
    F_x[6:9, 6:9] = ((Rotation.from_rotvec(np.squeeze(w_m - w_b) * dt)).as_matrix()).T

    return F_x


def vec2skew_symmetric(x):
    """
    Convert an (3,) array to its skew symmetric matrix form
    @param x: the array to convert
    @return: the skew symmetric array
    """
    if x.ndim == 2:
        x = np.squeeze(x)

    x_hat = np.array([[0, -x[-1], x[1]],
                      [x[-1], 0, -x[0]],
                      [-x[1], x[0], 0]], dtype=np.float64)
    return x_hat


def update_nominal_state_with_error_state(error_state, nominal_state):
    """
    update nominal states with updated error stated after incorporating measurement
    @param error_state: 18x1
    @param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    @return: new_nominal_state_tuple
    """
    delta_p = error_state[:3, :]
    delta_v = error_state[3:6, :]
    delta_theta = error_state[6:9, :]
    delta_a_b = error_state[9:12, :]
    delta_w_b = error_state[12:15, :]
    delta_g = error_state[15:, :]

    p, v, q, a_b, w_b, g = nominal_state

    new_p = p + delta_p
    new_v = v + delta_v
    new_q = q * Rotation.from_rotvec(np.squeeze(delta_theta))
    new_a_b = a_b + delta_a_b
    new_w_b = w_b + delta_w_b
    new_g = g + delta_g

    return (new_p, new_v, new_q, new_a_b, new_w_b, new_g)
