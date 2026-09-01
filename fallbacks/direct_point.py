"""Legacy single-point JointTrajectory fallback."""

from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def build(node, experiment, latest_positions):
    joints = list(experiment["joints"])
    if not all(joint in latest_positions for joint in joints):
        return None
    message = JointTrajectory()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "pit_fallback:direct_point"
    message.joint_names = joints
    point = JointTrajectoryPoint()
    point.positions = [float(value) for value in experiment["target_positions"]]
    point.time_from_start = Duration(
        seconds=float(experiment["fallback_duration_s"])
    ).to_msg()
    message.points = [point]
    return message
