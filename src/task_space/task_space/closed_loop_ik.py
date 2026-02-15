#!/usr/bin/env python3

import rclpy
import numpy as np
import matplotlib.pyplot as plt
from task_space import helper_fns as hf
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class ClosedLoopIk(Node):
    """
    _summary_

    :param Node: _description_
    :type Node: _type_
    """

    def __init__(self):
        super().__init__('closed_loop_ik')

        ### ------------Initialize parameters --------------------------------------------------###

        self.dt = 0.001
        self.time = 0.0
        self.total_time = 10.0
        self.L = [0.1625, 0.425, 0.3922, 0.1333, 0.0997, 0.09959]
        self.s_list = self.screw_list()
        self.t_prev = None
        self.q_cmd = None

        #### --------------- Transformations matrix for trajectory generation ------------------####

        self.M = np.array([[-1.0, 0.0, 0.0, self.L[2]],
                           [0.0, 0.0, 1.0, self.L[3]+self.L[5]],
                           [0.0, 1.0, 0.0, self.L[0]+self.L[1]-self.L[4]],
                           [0.0, 0.0, 0.0, 1.0]])

        self.t_ini = np.array([[-1.0, 0.0, 0.0, self.L[2]],
                               [0.0, 0.0, 1.0, self.L[3]+self.L[5]],
                               [0.0, 1.0, 0.0, self.L[0]+self.L[1]-self.L[4]],
                               [0.0, 0.0, 0.0, 1.0]])

        self.t_fin = np.array([[0.0, -1.0, 0.0, self.L[2]+self.L[4]],
                               [0.0, 0.0, 1.0, self.L[3]+self.L[5]],
                               [-1.0, 0.0, 0.0, self.L[0]+self.L[1]],
                               [0.0, 0.0, 0.0, 1.0]])
        self.t_fin = np.array([[0.0, 0.0, 1.0, self.L[3]+self.L[5]],
                               [1.0, 0.0, 0.0, -self.L[2]],
                               [0.0, 1.0, 0.0, self.L[0]+self.L[1]-self.L[4]],
                               [0.0, 0.0, 0.0, 1.0]])

        # self.t_fin = self.t_ini

        # # proportional gains
        self.kp = 0.1*np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

        ### -------------  joint parameters initialization ----------------------------------------###

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                            'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_count = len(self.joint_names)
        self.current_joint_values = None
        self.joint_offset = np.array([0.0, -1.5701, 1.57, 0.0, 0.0, 0.0])

        ### ----------------------------- publisher and subscriber ---------------------------------###

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)
        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.subscriber_callback, 10)
        self.timer_ = self.create_timer(self.dt, self.publisher_callback)

        ### ----------------- for plotting --------------------------###

        self.error_history = []
        self.time_history = []

    def screw_list(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """
        omg_list = np.array([[0.0, 0.0, 1.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 1.0, 0.0],
                             [0.0, 0.0, -1.0],
                             [0.0, 1.0, 0.0]
                             ])

        len_list = np.array([[0.0, 0.0, self.L[0]],
                             [0.0, 0.0, self.L[0]],
                             [0.0, 0.0, self.L[0]+self.L[1]],
                             [self.L[2], self.L[3], self.L[0]+self.L[1]],
                             [self.L[2], self.L[3], self.L[0]+self.L[1]-self.L[4]],
                             [self.L[2], self.L[3] +
                              self.L[5], self.L[0]+self.L[1]-self.L[4]]
                             ])

        S = []
        for omg, length in zip(omg_list, len_list):
            S.append(np.hstack([omg, -np.cross(omg, length)]))

        s_list = np.array(S).T

        return s_list

    def subscriber_callback(self, msg):
        """
        subscribes the actual joint values from joint_states topic
        and assigns to corresponding joint names

        """
        if self.current_joint_values is None:
            self.get_logger().warning("Fetching the current joint angles")

        joint_angles = []
        for joint_name in self.joint_names:
            idx = msg.name.index(joint_name)
            joint_angles.append(msg.position[idx])

        self.current_joint_values = np.array(joint_angles)-self.joint_offset

    def trajectory_generator(self, t_start, t_finish, traj_time, sim_time, method):
        """
        Takes input t_ini , t_fin
        output t_instant(4 x 4) Twist(6 x 1)
        """

        #### --------- Trajectory-Generator-------- ####

        u = sim_time/traj_time

        if method == 3:
            s = 3 * (u) ** 2 - 2 * (u) ** 3
        elif method == 5:
            s = 10*(u)**3 - 15*(u)**4 + 6*(u)**5
        else:
            raise ValueError(f"Undefined time scaling method: {method}")

        t_instant = t_start @ hf.MatrixExp6(
            hf.MatrixLog6(np.linalg.inv(t_start) @ t_finish) * s)

        ### ------------------ Twist Generator ---------------###

        if self.t_prev is None:
            twist = np.zeros(6)
            self.t_prev = t_instant
        else:
            delta_t = t_instant @ np.linalg.inv(self.t_prev)
            twist = (1 / self.dt) * hf.se3ToVec(
                hf.MatrixLog6(delta_t)
            )
            self.t_prev = t_instant

        return t_instant, twist

    def clik(self):
        """
        _summary_

        """

        if self.current_joint_values is None:
            return
        else:
            if self.q_cmd is None:
                self.q_cmd = self.current_joint_values

            if self.time <= self.total_time:
                t_des, twist_des = self.trajectory_generator(
                    self.t_ini, self.t_fin, traj_time=self.total_time, sim_time=self.time, method=5)

                t_act = hf.FKinSpace(self.M, self.s_list,
                                     self.current_joint_values)

                x_err = hf.se3ToVec(hf.MatrixLog6(
                    t_des @ np.linalg.inv(t_act)))

                self.error_history.append(x_err)  # storing values.....

                v_cmd = twist_des + self.kp@x_err

                # T_calc = hf.FKinSpace(self.M,
                #                       self.s_list, np.array([0.0, -1.3927, 1.4605, 0.7740, 0, -0.841]))

                # print(f'T_calc = {T_calc}')

                # Jacobian
                jac_s = hf.JacobianSpace(
                    self.s_list, self.current_joint_values)
                # jac_s_pinv = np.linalg.inv(
                #     jac_s.T @ jac_s + 0.001 * np.eye(6)
                # ) @ jac_s.T
                jac_s_pinv = np.linalg.pinv(jac_s)
                q_dot = jac_s_pinv @ v_cmd

                # Integrate INTERNAL reference
                self.q_cmd = self.q_cmd + q_dot * self.dt

                self.time += self.dt
                self.time_history.append(self.time)
            else:
                self.get_logger().info("Traj time completed holding at last joint position")
                self.q_cmd = self.q_cmd
                self.plot_error()

        return self.q_cmd

    def plot_error(self):

        errors = np.array(self.error_history)

        pos_errors = errors[:, 3:6]
        ori_errors = errors[:, 0:3]

        fig, axs = plt.subplots(2, 1, figsize=(8, 8))

        # ----- Position -----
        axs[0].plot(self.time_history, pos_errors[:, 0], label='X error')
        axs[0].plot(self.time_history, pos_errors[:, 1], label='Y error')
        axs[0].plot(self.time_history, pos_errors[:, 2], label='Z error')
        axs[0].set_ylabel("Position Error (m)")
        axs[0].set_title("Task Space Tracking Error")
        axs[0].legend()
        axs[0].grid()

        # ----- Orientation -----
        axs[1].plot(self.time_history, ori_errors[:, 0], label='Rx error')
        axs[1].plot(self.time_history, ori_errors[:, 1], label='Ry error')
        axs[1].plot(self.time_history, ori_errors[:, 2], label='Rz error')
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Orientation Error (rad)")
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.show()

    def publisher_callback(self):
        """
        publishes the desired joint values to '/position_controller/commands' topic

        """

        msg = Float64MultiArray()

        q_pub = self.clik()

        if q_pub is None:
            return
        else:
            msg.data = (q_pub+self.joint_offset).tolist()

        self.publisher_.publish(msg=msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopIk()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
