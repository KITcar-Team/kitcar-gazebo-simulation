import argparse
import os


def video_to_gif(
    input_file: str, output_file: str, width: int = None, length: int = None
) -> None:
    """This function converts a video (input_file) to gif(output_file)
    Args:
        input_file: the path of the input video
        output_file: the path of the output gif
        width: the resulting width of the gif
        length: defines the length of the video which gets transformed to a gif
    """
    cmd = "ffmpeg -y "
    cmd += f"-i {input_file} "
    if width is not None:
        cmd += f"-vf scale={width}:-1 "
    if length is not None:
        cmd += f"-t {length} "
    cmd += f"-an {output_file} "
    os.system(cmd)


if __name__ == "__main__":
    """Extract a folder of images from a rosbag."""
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("--input_file", help="Path to video file")
    parser.add_argument("--output_file", help="Path to output file")
    parser.add_argument(
        "--width", type=int, default=None, help="The width of the resulting gif"
    )
    parser.add_argument("--length", type=int, default=None, help="The length of the gif")

    args = parser.parse_args()
    video_to_gif(args.input_file, args.output_file, args.width, args.length)
