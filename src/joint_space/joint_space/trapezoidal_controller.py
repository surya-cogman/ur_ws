import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class TrapezoidalGenerator(Node):

    def __init__(self):
        super().__init__('trapezoidal_generator')

        # Publisher: position commands
        self.joint_publisher_ = self.create_publisher(
            Float64MultiArray,
            '/forward_command_controller/commands',
            10
        )

        # Subscriber: joint states
        self.joint_subscriber_ = self.create_subscription(
            JointState,
            '/joint_states',
            self.cb_joint_sub,
            10
        )

        self.joint_index = {}
        self.joint_index_initialize = False
        self.initial_received = False

        # initializing parameters
        self.dt = 0.01
        self.t = 0.0
        self.initial_angles = None
        self.desired_angles = np.array([90.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.final_angles = np.deg2rad(self.desired_angles)
        self.Vmax = 5.0
        self.Amax = 2.5

        self.timer_ = self.create_timer(self.dt, self.cb_joint_pub)

    def cb_joint_sub(self, msg):
        """
        _summary_

        :param msg: _description_
        :type msg: _type_
        """
        if not self.joint_index_initialize:
            for i, name in enumerate(msg.name):
                self.joint_index[name] = i
            self.joint_index_initialize = True

        if not self.initial_received:
            self.initial_angles = np.array(
                [msg.position[self.joint_index[name]] for name in msg.name])
            self.setup_trajectory()
            self.initial_received = True

    def setup_trajectory(self):
        """
        _summary_
        """

        self.distances = self.final_angles - self.initial_angles
        self.max_distance = np.max(np.abs(self.distances))

        self.ratios = self.distances/self.max_distance

        self.t_acc = self.Vmax/self.Amax
        d_acc_dec = self.Vmax * self.t_acc

        if self.max_distance < d_acc_dec:
            self.t_acc = np.sqrt(self.max_distance / self.Amax)
            self.t_cruise = 0.0
            self.Vmax = self.Amax * self.t_acc
        else:
            self.t_cruise = (self.max_distance - d_acc_dec) / self.Vmax

        self.total_time = 2 * self.t_acc + self.t_cruise

    def compute_trajectory_step(self):
        """
        _summary_

        :return: _description_
        :rtype: _type_
        """

        if self.t > self.total_time:
            return None

        # Accel
        if self.t < self.t_acc:
            S = 0.5 * self.Amax * self.t**2

        # Cruise
        elif self.t < (self.t_acc + self.t_cruise):
            S = (0.5 * self.Amax * self.t_acc**2 +
                 self.Vmax * (self.t - self.t_acc))

        # Decel
        else:
            t_dec = (self.t - (self.t_acc + self.t_cruise))
            S = (0.5 * self.Amax * self.t_acc**2 +
                 self.Vmax * self.t_cruise +
                 self.Vmax * t_dec -
                 0.5 * self.Amax * t_dec**2)

        q_cmd = self.initial_angles + S * self.ratios
        self.t += self.dt
        return q_cmd

    def cb_joint_pub(self):

        if not self.initial_received:
            return

        q_cmd = self.compute_trajectory_step()

        if q_cmd is None:
            self.get_logger().info("Trajectory Completed")
            self.timer_.cancel()
            return

        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.joint_publisher_.publish(msg=msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidalGenerator()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
