"""Legacy two-stage JointTrajectory fallback with an explicit midpoint."""

from rclpy.duration import Duration
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def build(node, experiment, latest_positions):
    joints = list(experiment["joints"])
    if not all(joint in latest_positions for joint in joints):
        return None
    duration = float(experiment["fallback_duration_s"])
    targets = [float(value) for value in experiment["target_positions"]]
    starts = [float(latest_positions[joint]) for joint in joints]
    message = JointTrajectory()
    message.header.stamp = node.get_clock().now().to_msg()
    message.header.frame_id = "pit_fallback:two_stage"
    message.joint_names = joints
    midpoint = JointTrajectoryPoint()
    midpoint.positions = [(start + target) / 2.0 for start, target in zip(starts, targets)]
    midpoint.time_from_start = Duration(seconds=duration / 2.0).to_msg()
    target = JointTrajectoryPoint()
    target.positions = targets
    target.time_from_start = Duration(seconds=duration).to_msg()
    message.points = [midpoint, target]
    return message
