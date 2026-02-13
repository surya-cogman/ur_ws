#!usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
from rcl_interfaces.msg import SetParametersResult


class TrapezoidalMotionPlanner(Node):

    def __init__(self):
        super().__init__('trapezoidal_motion_planner')

        ### ------- initial_parameters --------------------####
        self.time = 0.0
        self.dt = 0.01
        self.total_time = None
        self.Vmax = 3.0
        self.Amax = 1.0
        self.q_cmd_store = None

        ### ------------- Taking joint positons from terminal ---------------####

        self.declare_parameter(
            'target_position', [0.0, -1.5701, -1.5701, 0.0, 0.0, 0.0])
        self.target_joint_values = np.array(
            self.get_parameter('target_position').value)

        self.add_on_set_parameters_callback(self.parameter_callback)

        ### -------  publisher and subscriber --------###

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)
        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.cb_joint_sub, 10)

        self.timer_ = self.create_timer(self.dt, self.cb_joint_pub)

        ### ------ subscribing joint angles ------------------###

        self.initialize = False
        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                            'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        self.joint_count = len(self.joint_names)
        self.current_joint_values = None
        self.current_joint_velocity = np.zeros(self.joint_count)

    def cb_joint_sub(self, msg):
        """
        subscribes the joint points once when node gets started

        :param msg: contains state command and inteface data from joint_states topics

        """

        if self.initialize:
            return

        try:

            positions = []

            for joint_name in self.joint_names:
                idx = msg.name.index(joint_name)
                positions.append(msg.position[idx])

            self.get_logger().info('Joint angles subscribed')
            self.current_joint_values = np.array(positions)
            self.initialize = True

        except ValueError:
            pass

    def parameter_callback(self, params):
        """
        Returns True once New Joint values are given from Terminal 
        """

        for param in params:
            if param.name == 'target_position':
                new_target = np.array(param.value)

                self.target_joint_values = new_target
                self.get_logger().info(f"New target received: {new_target}")
                self.time = 0.0
                self.total_time = None
                self.initialize = False

        return SetParametersResult(successful=True)

    def trajectory_generator(self):
        """
        Generates a trapezoidal or traingular velocity profile motion based on 

        current_joint_values and target_joint_values
        """

        if not self.initialize:
            return self.get_logger().warning("waiting for initial joint angles")

        joint_dist = self.target_joint_values - self.current_joint_values
        max_dist = np.max(np.abs(joint_dist))
        dist_ratio = joint_dist/max_dist

        ### ------- Time needed for each phase  ------####

        t_acc = self.Vmax/self.Amax
        t_dec = t_acc

        S_acc = 0.5*t_acc * self.Vmax  # area in trapezoidal velocity profile during ascent
        S_dec = S_acc

        if max_dist < 2*S_acc:
            t_acc = np.sqrt(max_dist / self.Amax)
            t_cruise = 0.0
            self.Vmax = self.Amax * t_acc
        else:
            t_cruise = (max_dist - 2*S_acc) / self.Vmax

        self.total_time = t_acc + t_cruise + t_dec

        ### ------------- Generating trajectory points ----------- ###

        if self.time <= t_acc:
            S = 0.5 * self.Amax * self.time**2
        elif t_acc < self.time <= (t_acc + t_cruise):
            S = (0.5 * self.Amax * t_acc**2 +
                 self.Vmax * (self.time - t_acc))
        elif (t_acc + t_cruise) < self.time <= (t_acc + t_cruise + t_dec):

            S = (0.5 * self.Amax * t_acc**2 +
                 self.Vmax * t_cruise +
                 self.Vmax * (self.time - (t_acc + t_cruise)) -
                 0.5 * self.Amax * (self.time - (t_acc + t_cruise))**2)
        else:
            self.get_logger().info("Total time reached......")
            return self.q_cmd_store

        q_cmd = self.current_joint_values + S * dist_ratio
        self.time += self.dt
        self.q_cmd_store = q_cmd
        return q_cmd

    def cb_joint_pub(self):
        """
        Function that publishes the joint values calculates to  '/position_controller/commands' topic

        """

        if not self.initialize or self.total_time is None:
            self.trajectory_generator()
            return
        else:
            q_cmd = self.trajectory_generator()

        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.publisher_.publish(msg=msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidalMotionPlanner()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
