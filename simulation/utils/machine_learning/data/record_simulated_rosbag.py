import argparse
import os
import subprocess
import threading
import time
from pathlib import Path

import rosnode


def ros_cmd(**kwargs):
    cmd = "roslaunch gazebo_simulation record_random_drive.launch"
    cmd = cmd.split(" ")
    for key, val in kwargs.items():
        if val is None:
            continue
        cmd.append(f"{key}:={val}")
    return cmd


def is_node_running(node_name: str) -> bool:
    """Check if node is still running."""
    try:
        # Select only nodes that have automatic_drive in their name
        return len([n for n in rosnode.get_node_names() if node_name in n]) > 0
    except rosnode.ROSNodeIOException:
        # Happens when roscore is not up yet
        return False


def run(cmd, max_duration: float = 120, node_name="automatic_drive"):
    """Run ROS cmd in background and stop when automatic drive node shuts down."""

    # Ros cmd
    ros_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    is_running = True

    def read(out):
        while is_running:
            if out.readable():
                output = out.readline()
                if not output:
                    continue
                output = output.decode().strip()  # Turn into nice string
                print(output)

    out_thread = threading.Thread(target=read, args=[ros_process.stdout])
    err_thread = threading.Thread(target=read, args=[ros_process.stderr])

    out_thread.start()
    err_thread.start()

    start = time.time()
    node_started = False

    # Wait for node to run
    # Keep active while automatic drive node is still running / at most max_duration seconds
    while time.time() - start < max_duration:
        running = is_node_running(node_name)
        node_started = node_started or running
        if node_started and not running:
            break
        time.sleep(0.1)

    # Stop output threads
    is_running = False

    # Ensure ROS is killed
    # Kill all nodes
    os.system("rosnode kill -a")
    # Kill gazebo
    os.system("killall -9 gzserver")
    os.system("killall -9 gzclient")
    # And the roslaunch process
    if ros_process.returncode is None:
        os.kill(ros_process.pid, 15)


def main(**kwargs):
    max_duration = kwargs["max_duration"]
    del kwargs["max_duration"]

    # Make rosbag dir if it doesn't exist
    if kwargs["rosbag_dir"] is not None:
        # Make path absolute
        kwargs["rosbag_dir"] = Path(kwargs["rosbag_dir"]).absolute()
        Path(kwargs["rosbag_dir"]).mkdir(parents=True, exist_ok=True)

    # Unpack seeds
    seeds = kwargs["seed"]
    del kwargs["seed"]

    cmds = [ros_cmd(seed=seed, **kwargs) for seed in seeds]

    for cmd in cmds:
        run(cmd, max_duration)
        time.sleep(1)  # Give some time to shut down


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Let car drive randomly along road and record rosbag."
            "If multiple seeds are passed, the road is simulated for every seed."
            "This requires that kitcar-ros is installed."
        )
    )

    parser.add_argument(
        "--rosbag_dir",
        help="Directory with rosbag(s).",
        default=os.path.join(
            os.environ["KITCAR_REPO_PATH"],
            "kitcar-gazebo-simulation",
            "data",
            "simulated_rosbags",
        ),
    )
    parser.add_argument("--rosbag_name", help="Name of the rosbag.", required=True)
    parser.add_argument("--gui", help="Launch gui.", default=False)
    parser.add_argument("--road", help="Name of the road.", default="ci_roads/random_road")
    parser.add_argument(
        "--seed", help="Seed(s) passed when generating the road.", default=[None], nargs="+"
    )
    parser.add_argument("--max_duration", help="Maximum recording time.", default=120)
    parser.add_argument(
        "--label_camera",
        help="Start the label camera as well.",
        action="store_true",
    )

    parser.add_argument(
        "--control_sim_rate",
        help="Whether to control the sim rate.",
        action="store_true",
    )

    kwargs = {k: v for k, v in parser.parse_args()._get_kwargs()}

    main(**kwargs)
