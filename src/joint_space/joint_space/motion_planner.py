#!usr/bin/env python3

import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from rcl_interfaces.msg import SetParametersResult


class MotionPlanner(Node):

    def __init__(self):
        super().__init__('motion_planner')

        #### ------ declaring parameters -------#####

        self.declare_parameter('target_position', [
                               0.0, -1.57, -1.57, 0.0, 0.0, 0.0])
        self.target_list = np.array(
            self.get_parameter('target_position').value)

        self.add_on_set_parameters_callback(self.parameter_callback)

        #### ------- defining initial parameters ---------------######

        self.dt = 0.01
        self.total_time = 5.0
        self.max_vel = 3.0
        self.max_acc = 1.0

        #### --------- params for joint_angles ------------#####

        self.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint',
                            'elbow_joint', 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']

        self.joint_count = len(self.joint_names)
        self.initialize = False
        self.current_positions = None
        self.current_velocity = np.zeros(self.joint_count)

        ###### -----------------  publisher and subscriber --------------#####

        self.subscriber_ = self.create_subscription(
            JointState, '/joint_states', self.cb_joint_sub, 10)

        self.publisher_ = self.create_publisher(
            Float64MultiArray, '/position_controller/commands', 10)

        self.timer_ = self.create_timer(self.dt, self.cb_joint_pub)

    def cb_joint_sub(self, msg):

        if self.initialize:
            return

        try:
            positions = []
            for name in self.joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])

                self.get_logger().info("joints angles obtained")
                self.current_positions = positions
                self.destroy_subscription(self.subscriber_)
                self.initialize = True

        except ValueError:
            pass

    def cb_joint_pub(self):

        if not self.initialize:
            return

        new_pos = []

        for i in range(self.joint_count):
            p = self.current_positions[i]
            v = self.current_velocity[i]
            pf = self.target_list[i]

            dist = pf - p

            if abs(dist) < 1e-4 and abs(v) < 1e-3:
                p = pf
                v = 0.0
            else:
                decel_dist = (v * v) / (2 * self.max_acc)

                if abs(dist) > decel_dist:
                    # accelerate
                    v += np.sign(dist) * self.max_acc * self.dt
                    v = np.clip(v, -self.max_vel, self.max_vel)
                else:
                    # decelerate toward zero velocity
                    v -= np.sign(v) * self.max_acc * self.dt

                p += v * self.dt

                self.current_positions[i] = p
                self.current_velocity[i] = v
                new_pos.append(float(p))

        if np.allclose(self.current_positions, self.target_list, atol=0.01):
            self.get_logger().info("Target reached.")
            self.current_velocity[:] = 0.0
            return

        msg = Float64MultiArray()
        msg.data = new_pos
        self.publisher_.publish(msg)

    def parameter_callback(self, params):

        for param in params:
            if param.name == "target_position":
                new_target = np.array(param.value)

                self.target_list = new_target

        return SetParametersResult(successful=True)


def main(args=None):

    rclpy.init(args=args)
    node = MotionPlanner()
    rclpy.spin(node=node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
