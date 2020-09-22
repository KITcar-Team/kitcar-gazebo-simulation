import argparse

from . import record_simulated_rosbag, rosbag_to_images


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
            rosbag_dir, kwargs["output_dir"], kwargs["image_topic"]
        )

        # Delete the rosbag
        import shutil

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
        "--seed", help="Seed(s) passed when generating the road.", default=[None], nargs="+"
    )
    parser.add_argument("--max_duration", help="Maximum recording time.", default=120)

    kwargs = {k: v for k, v in parser.parse_args()._get_kwargs()}

    main(**kwargs)
