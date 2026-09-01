class _Vector3:
    def __init__(self) -> None:
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0


class Twist:
    """Fields used by the public planner before nav2py sends numeric values."""

    def __init__(self) -> None:
        self.linear = _Vector3()
        self.angular = _Vector3()
