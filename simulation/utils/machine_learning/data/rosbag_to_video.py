import argparse
import os
import shutil

from simulation.utils.machine_learning.data.images_to_video import images_to_video
from simulation.utils.machine_learning.data.rosbag_to_images import rosbag_to_images


def rosbag_to_video(rosbag_dir: str, output_dir: str, image_topic: str):
    os.makedirs(output_dir, exist_ok=True)
    for root, dirs, files in os.walk(rosbag_dir):
        for name in files:
            if not name.lower().endswith(".bag"):
                continue

            input_file_path = os.path.join(root, name)
            rosbag_to_images(input_file_path, os.path.join(rosbag_dir, "tmp"), image_topic)

            output_file_path = os.path.join(output_dir, name.replace(".bag", ".mp4"))
            images_to_video(
                os.path.abspath(os.path.join(rosbag_dir, "tmp/*.png")),
                output_file_path,
                use_glob=True,
            )

            shutil.rmtree(os.path.join(rosbag_dir, "tmp"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rosbags to Videos")
    parser.add_argument(
        "--rosbag_dir",
        type=str,
        required=True,
        help="directory of all rosbags",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="the output directory for all videos",
    )
    parser.add_argument("--image_topic", default="/camera/image_raw", help="Image topic.")
    args = parser.parse_args()

    rosbag_to_video(args.rosbag_dir, args.output_dir, args.image_topic)
