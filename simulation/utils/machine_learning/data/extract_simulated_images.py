import argparse
import shutil

from . import record_simulated_rosbag, rosbag_to_images, rosbag_to_labels


def main(**kwargs):

    # Unpack seeds
    seeds = kwargs["seed"]
    del kwargs["seed"]

    rosbag_dir = "tmp_rosbags"
    rosbag_name = kwargs["road"].split("/")[-1]

    for seed in seeds:
        # Record one rosbag at a time
        record_simulated_rosbag.main(
            rosbag_dir=rosbag_dir, rosbag_name=rosbag_name, seed=[seed], **kwargs
        )
        # Extract images from that rosbag.
        rosbag_to_images.rosbag_to_images(
            rosbag_dir, kwargs["output_dir"], kwargs["image_topic"], name_after_header=True
        )
        if kwargs["label_image_topic"] is True:
            # Extract images from that rosbag.
            rosbag_to_images.rosbag_to_images(
                rosbag_dir,
                kwargs["output_dir"] + "/debug",
                kwargs["label_image_topic"],
                name_after_header=True,
            )

        if kwargs["label_camera"] is True:
            assert "label_file" in kwargs, "Required argument label_file missing."
            assert "label_topic" in kwargs, "Required argument label_topic missing."

            # Extract images from that rosbag.
            rosbag_to_images.rosbag_to_images(
                rosbag_dir,
                kwargs["output_dir"] + "/debug",
                kwargs["label_image_topic"],
                name_after_header=True,
            )

            rosbag_to_labels.rosbag_to_labels(
                rosbag_dir,
                output_file=kwargs["label_file"],
                label_topic=kwargs["label_topic"],
            )

        # Delete the rosbag
        shutil.rmtree(rosbag_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Record rosbags, extract images, and delete rosbags in one pass."
            "If multiple seeds are passed, the road is simulated for every seed."
            "This requires that kitcar-ros is installed."
        )
    )

    parser.add_argument("--gui", help="Launch gui.", default=False)
    parser.add_argument("--road", help="Name of the road.", default="ci_roads/random_road")
    parser.add_argument("--output_dir", help="Directory for resulting images.")
    parser.add_argument("--image_topic", help="Topic of the image to be extracted.")
    parser.add_argument(
        "--randomize_path", help="Drive randomly on the road.", default=True
    )
    parser.add_argument(
        "--control_sim_rate",
        help="Automatically adapt simulation speed.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--apply_gan",
        help="Apply gan when running the simulation.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--factor_keep_pixels",
        help="Factor of original image that is kept when applying GAN.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--factor_keep_colored_pixels",
        help="Factor of colored pixels in original image that is kept when applying GAN.",
        type=float,
        default=0,
    )
    parser.add_argument(
        "--seed", help="Seed(s) passed when generating the road.", default=[None], nargs="+"
    )
    parser.add_argument("--max_duration", help="Maximum recording time.", default=120)

    # Labels
    parser.add_argument(
        "--label_image_topic", help="Topic of the labeled debug image to be extracted."
    )

    parser.add_argument(
        "--label_camera", help="Launch label camera.", action="store_true", default=False
    )
    parser.add_argument(
        "--label_topic", help="Topic of the labels to be extracted.", required=False
    )
    parser.add_argument(
        "--label_file", help="File path where to store the labels.", required=False
    )
    parser.add_argument(
        "--show_stdout",
        help="Whether or not to print stdout.",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--show_stderr",
        help="Whether or not to print stderr.",
        default=False,
        action="store_true",
    )

    kwargs = {k: v for k, v in parser.parse_args()._get_kwargs()}

    main(**kwargs)
