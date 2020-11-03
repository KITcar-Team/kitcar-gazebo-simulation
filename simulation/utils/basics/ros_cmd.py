import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Tuple


@dataclass
class ROSCmd:
    """Holds data for a ROS command and makes it executable."""

    cmd_base: str
    """Command string without any arguments."""
    env_vars: Dict[str, Any] = None
    """Environment variables that should be set prior to running the command."""
    ros_args: Dict[str, Any] = None
    """ROS arguments passed when running the command."""

    def __init__(self, cmd_base: str, *, env_vars: Dict[str, Any], **ros_args):
        """Initialize ROSCmd.

        Args:
            cmd_base: Command string without arguments.
            env_vars: Environment variables set prior to running the command.
            ros_args: ROS arguments passed when running the command.
        """
        self.cmd_base = cmd_base
        # Ensure that dicts exist and copy them to ensure
        # that they are not accidentally used in multiple places.
        self.env_vars = env_vars.copy() if env_vars is not None else dict()
        self.ros_args = ros_args

    def get_cmd(self) -> str:
        """Create the command string."""
        cmd = self.cmd_base

        # Append arguments
        for name, val in self.ros_args.items():
            cmd += f" {name}:={val}"

        return cmd

    def run(self) -> Tuple[int, str]:
        """Prepare and run the command.

        Returns:
            Returncode and std output.
        """
        # Setup defined environment variables
        for var, val in self.env_vars.items():
            os.environ[var] = str(val)

        process = subprocess.run(self.get_cmd(), capture_output=True, shell=True, text=True)

        return process.returncode, process.stdout
