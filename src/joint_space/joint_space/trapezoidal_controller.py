#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState


class TrapezoidalController(Node):
    def __init__(self):
        super().__init__('trapezoidal_controller')

        # ---- USER PARAMETERS ----
        self.dt = 0.01                 # 100 Hz
        self.max_vel = 1.0             # rad/s
        self.max_acc = 3.0             # rad/s^2
        self.target_pos = np.array(
            [0.0, -1.57, -1.57, 0.0, 0.0, 0.0]
        )

        self.joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint"
        ]

        self.joint_count = len(self.target_pos)

        # Internal state (initialized later)
        self.current_pos = None
        self.current_vel = np.zeros(self.joint_count)

        self.initialized = False

        # Publisher
        self.publisher = self.create_publisher(
            Float64MultiArray,
            '/position_controller/commands',
            10
        )

        # Subscriber (read once)
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Timer
        self.timer = self.create_timer(self.dt, self.update)

        self.get_logger().info("Waiting for initial joint state...")

    def joint_state_callback(self, msg):

        if self.initialized:
            return

        try:
            positions = []
            for name in self.joint_names:
                idx = msg.name.index(name)
                positions.append(msg.position[idx])

            self.current_pos = np.array(positions)
            self.initialized = True

            self.get_logger().info("Initial joint positions received.")
            self.get_logger().info(f"Start pose: {self.current_pos}")

            # Stop listening after first read
            self.destroy_subscription(self.subscription)

        except ValueError:
            # Joint names not found yet
            pass

    def update(self):

        if not self.initialized:
            return

        new_pos = []

        for i in range(self.joint_count):
            p = self.current_pos[i]
            v = self.current_vel[i]
            pf = self.target_pos[i]

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

            self.current_pos[i] = p
            self.current_vel[i] = v
            new_pos.append(float(p))

            if np.allclose(self.current_pos, self.target_pos, atol=0.01):
                self.get_logger().info("Target reached.")
                self.timer.cancel()

        msg = Float64MultiArray()
        msg.data = new_pos
        self.publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TrapezoidalController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
