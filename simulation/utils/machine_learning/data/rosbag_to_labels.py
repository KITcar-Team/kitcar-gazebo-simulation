import argparse
import os

import rosbag
from simulation_groundtruth.msg import ImageLabels as ImageLabelsMsg
from simulation_groundtruth.msg import LabeledBoundingBox as LabeledBoundingBoxMsg

from .labeled_dataset import LabeledDataset


def rosbag_to_labels(bag_path: str, output_file: str, label_topic: str) -> None:

    # Find all rosbags recursively
    bag_files = []
    if os.path.isdir(bag_path):
        for root, dirs, files in os.walk(bag_path):
            for file in files:
                if file.endswith(".bag"):
                    bag_files.append(os.path.join(root, file))
    else:
        bag_files = [bag_path]

    labeled_dataset = (
        LabeledDataset.from_yaml(output_file)
        if os.path.exists(output_file)
        else LabeledDataset()
    )
    bb_msg_fields = list(LabeledBoundingBoxMsg.__slots__)
    del bb_msg_fields[bb_msg_fields.index("class_description")]
    labeled_dataset.attributes = tuple(["img_name"] + bb_msg_fields)

    for bag_file in bag_files:
        print(f"Extract labels from {bag_file}.")
        bag = rosbag.Bag(bag_file, "r")

        for topic, msg, t in bag.read_messages(topics=[label_topic]):
            msg: ImageLabelsMsg = msg

            # Create image name from bag name
            img_name = (
                f"{bag.filename.split('/')[-1].split('.')[0]}"
                f"_frame{msg.img_header.seq:06}.png"
            )
            labeled_dataset.labels[img_name] = []  # Add empty entry to labels
            for bb in msg.bounding_boxes:
                labeled_dataset.append_label(
                    key=img_name,
                    label=[img_name] + [getattr(bb, field) for field in bb_msg_fields],
                )

                labeled_dataset.classes[bb.class_id] = bb.class_description

        bag.close()

    labeled_dataset.save_as_yaml(output_file)


if __name__ == "__main__":
    """Extract a folder of labels from a rosbag."""
    parser = argparse.ArgumentParser(description="Extract labels from a ROS bag.")
    parser.add_argument("--bag", help="Directory with rosbag(s).")
    parser.add_argument("--output_file", help="Output file.")
    parser.add_argument("--label_topic", help="Label topic.")

    args = parser.parse_args()
    rosbag_to_labels(args.bag, args.output_file, args.label_topic)
