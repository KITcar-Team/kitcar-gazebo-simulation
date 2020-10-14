import argparse
import os

import cv2
import rosbag
from cv_bridge import CvBridge


def rosbag_to_images(
    bag_path: str, output_dir: str, image_topic: str, name_after_header: bool = False
) -> None:

    # Create output_dir if it doesnt exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Find all rosbags recursively
    bag_files = []
    if os.path.isdir(bag_path):
        for root, dirs, files in os.walk(bag_path):
            for file in files:
                if file.endswith(".bag"):
                    bag_files.append(os.path.join(root, file))
    else:
        bag_files = [bag_path]

    for bag_file in bag_files:
        print(f"Extract images from {bag_file}.")
        bag = rosbag.Bag(bag_file, "r")
        bridge = CvBridge()
        count = 0

        for topic, msg, t in bag.read_messages(topics=[image_topic]):
            cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

            # Create image name from bag name
            img_name = (
                f"{bag.filename.split('/')[-1].split('.')[0]}"
                f"_frame{msg.header.seq if name_after_header else count:06}.png"
            )

            cv2.imwrite(os.path.join(output_dir, img_name), cv_img)
            count += 1

        bag.close()


if __name__ == "__main__":
    """Extract a folder of images from a rosbag."""
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--bag", help="Directory with rosbag(s).")
    parser.add_argument("--output_dir", help="Output directory.")
    parser.add_argument("--image_topic", help="Image topic.")
    parser.add_argument(
        "--name_after_header",
        help="Whether to use the message's header as the image's name.",
        action="store_true",
    )

    args = parser.parse_args()
    rosbag_to_images(args.bag, args.output_dir, args.image_topic, args.name_after_header)
