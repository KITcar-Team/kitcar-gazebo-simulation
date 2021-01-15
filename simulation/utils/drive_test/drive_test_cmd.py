import os
import shutil
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
    rosbag_path: str = None
    """The directory where the rosbags are stored."""
    keep_rosbag: bool = False
    """Keep rosbag in every case"""

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

        if "rosbag_path" in ros_args:
            self.rosbag_path = ros_args["rosbag_path"]
            ros_args["rosbag_path"] = os.path.join(self.rosbag_path, "rosbag")
            self.keep_rosbag = ros_args.get("keep_rosbag", False)

        self.desc = desc
        self.must_succeed = must_succeed
        super().__init__(
            "rostest simulation_evaluation drive.test", env_vars=environment, **ros_args
        )

    def run(self) -> str:
        if self.rosbag_path is None:
            status, output = super().run()
            self.success = status == 0
            return output

        dirs_before = set(os.listdir(self.rosbag_path))
        status, output = super().run()
        dirs_after = set(os.listdir(self.rosbag_path))

        self.success = status == 0

        # if successful and we are allowed to delete, then delete rosbag
        if self.success and not self.keep_rosbag:
            (new_dir,) = dirs_after - dirs_before
            shutil.rmtree(os.path.join(self.rosbag_path, new_dir))

        return output
