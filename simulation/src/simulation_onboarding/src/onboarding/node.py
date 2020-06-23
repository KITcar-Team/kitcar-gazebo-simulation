"""OnboardingNode."""

__copyright__ = "KITcar"

from simulation.utils.ros_base.node_base import NodeBase


class OnboardingNode(NodeBase):
    """ROS node to teach new members."""

    def __init__(self):
        """Initialize the node."""
        super().__init__(name="onboarding_node")

        # Start running node
        # This will do the following
        # 1) Call self.start()
        # 2) Call self.steer() 60 times a second!
        # 3) Call self.stop() when ROS is shutting down
        self.run(function=self.steer, rate=60)

    def start(self):
        """Start node."""
        # When overwriting a function, ensure that the original function (NodeBase.start()) is also called
        super().start()

    def stop(self):
        """Turn off node."""
        # When overwriting a function, ensure that the original function (NodeBase.stop()) is also called
        super().stop()

    def steer(self):
        """Control the car's pose to drive along the road."""
        pass
