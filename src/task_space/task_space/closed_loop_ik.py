#!usr/bin/env python3


import rclpy
import numpy as np
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

        ### ------------Initialize parameters ------------------###

        self.dt = 0.01
        self.time = 0.0
        self.total_time = 10.0

        self.t_ini = np.eye(4)
        self.t_fin = np.eye(4)
        self.t_prev = None
        self.kp = np.eye(6)
        self.s_list = self.screw_list()
        self.q_cmd = None

        ### -------------  joint parameters initialization -----###

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                            'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_count = len(self.joint_names)
        self.current_joint_values = None

        ### ----------- publisher and subscriber ---------------###

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)
        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.subscriber_callback, 10)
        self.timer_ = self.create_timer(self.dt, self.publisher_callback)

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

        self.current_joint_values = np.array(joint_angles)

    def screw_list(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """

        return s_list

    def trajectory_generator(self, t_start, t_finish, traj_time, method):
        """
        Takes input t_ini , t_fin 
        output t_instant(4 x 4) Twist(6 x 1)
        """

        #### --------- Trajectory-Generator-------- ####

        u = self.time/traj_time

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

        t_des, twist_des = self.trajectory_generator(
            self.t_ini, self.t_fin, 10.0, 5)

        t_act = hf.FKinSpace(self.t_ini, self.s_list,
                             self.current_joint_values)

        x_err = hf.se3ToVec(hf.MatrixLog6(t_des @ np.linalg.inv(t_act)))

        v_cmd = twist_des + self.kp@x_err

        # Jacobian
        jac_s = hf.JacobianSpace(self.s_list, self.current_joint_values)
        jac_s_pinv = np.linalg.inv(
            jac_s.T @ jac_s + 0.001 * np.eye(7)
        ) @ jac_s.T

        q_dot = jac_s_pinv @ v_cmd

        # Integrate INTERNAL reference

        if self.q_cmd is None:
            self.q_cmd = self.current_joint_values

        self.q_cmd = self.q_cmd + q_dot * self.dt

        self.dt += self.dt

        return self.q_cmd

    def publisher_callback(self):
        """
        publishes the desired joint values to '/position_controller/commands' topic
        """
        msg = Float64MultiArray()
        q_pub = self.clik()
        msg.data = q_pub.tolist()

        self.publisher_.publish(msg=msg)


def main(args=None):
    rclpy.init(args=args)
    node = ClosedLoopIk()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
