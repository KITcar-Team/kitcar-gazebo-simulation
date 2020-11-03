from dataclasses import dataclass
from typing import Any, Dict

from simulation.utils.basics.ros_cmd import ROSCmd


@dataclass
class DriveTestCmd(ROSCmd):
    """Simple Wrapper of ROSCmd to run drive tests."""

    desc: str = None
    """Description of the drive."""
    must_succeed: bool = False
    """If the command has to succeed."""
    success: bool = False
    """If the command has ran successfully."""

    def __init__(
        self,
        *,
        desc: str,
        must_succeed: bool = False,
        environment: Dict[str, Any] = None,
        **ros_args,
    ):
        """Initialize with description, env variables and ros arguments."""

        # Check if there are all required paremeters.
        assert "road" in ros_args, "Argument road is missing."
        assert "time_limit" in ros_args, "Argument time_limit is missing."
        assert "mission_mode" in ros_args, "Argument mission_mode is missing."

        self.desc = desc
        self.must_succeed = must_succeed
        super().__init__(
            "rostest simulation_evaluation drive.test", env_vars=environment, **ros_args
        )
