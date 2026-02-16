# Closed-Loop Task Space Inverse Kinematics for UR5e (ROS2)

## 📌 Overview

This project implements a Closed-Loop Inverse Kinematics (CLIK) controller in task space for the UR5e manipulator using:

- ROS2
- Product of Exponentials (Modern Robotics formulation)
- Space Jacobian
- Damped Least Squares inverse
- Quintic time scaling trajectory generation

The controller tracks a desired end-effector trajectory in SE(3) and publishes joint position commands to a ROS2 position controller.

---

## 🚀 Features

- Space-frame screw axis formulation
- Forward Kinematics using Product of Exponentials
- Space Jacobian computation
- Damped Least Squares pseudo-inverse
- Quintic time scaling trajectory generation
- Real-time closed-loop task space error correction
- Position and orientation error plotting

---

## 🧠 Control Law

The controller is based on:

    $ \dot{q} = J^+ (V_d + K_p e) $.

Where:

- J⁺ → Damped pseudo-inverse of Space Jacobian  
- V_d → Desired twist from trajectory generator  
- e → SE(3) error  
- Kp → Proportional gain matrix  



---

## 🏗 System Architecture

    /joint_states   →  Subscriber
    CLIK Controller →  Computes q_dot
    Position Command → /position_controller/commands

---

## ⚙️ Requirements

- ROS2 (Humble or later recommended)
- numpy
- matplotlib
- Modern Robotics helper functions (MatrixExp6, MatrixLog6, JacobianSpace, etc.)

---

## 🛠 How to Run

### 1️⃣ Launch UR5e Simulation

Make sure your UR5e is running with a position controller:

    ros2 control list_controllers

Ensure `/position_controller` is active.

---

### 2️⃣ (Optional) Move Robot to Initial Configuration

Example:

    ros2 topic pub -r 1.0 /position_controller/commands \
    std_msgs/msg/Float64MultiArray "{data: [0.0,-1.5701,1.57,0.0,0.0,0.0]}"

---

### 3️⃣ Run Controller Node

    ros2 run <your_package_name> closed_loop_ik

---

## 📊 Output

After trajectory completion:

- Position error plot (X, Y, Z)
- Orientation error plot (Rx, Ry, Rz)

Errors are computed using:

    e = se3ToVec( log(T_d T⁻¹) )

---

## 🔍 Important Notes

- t_ini is computed from the actual robot pose at startup.
- Do not manually shift joint values in the publisher.
- Joint mapping is done by name to ensure correct ordering.
- Damped Least Squares is used to avoid singularities.

---

## 🧪 Known Limitations

- No joint limit avoidance
- No collision checking
- No nullspace optimization
- No velocity saturation

---

## 📈 Future Improvements

- Joint limit avoidance using nullspace projection
- Velocity clamping
- Singularity detection
- Orientation weighting
- Secondary posture task

---

## 📚 References

- Modern Robotics – Kevin M. Lynch & Frank C. Park
- Product of Exponentials formulation
- Space Jacobian method

---

## 👨‍💻 Author

Your Name  
Robotics | Manipulator Control | Task-Space Control
