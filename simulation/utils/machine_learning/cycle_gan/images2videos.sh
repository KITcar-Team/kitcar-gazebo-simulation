#!/bin/bash
# Convert test images into video
# Has to be run after running test.py!
#
# Run with
#
# ./images2videos.sh NAME_OF_MODEL PREFIX_IMAGE_NAMES
#
# E.g. if the images in the test results folder have names like long_name_frame090990_fake_B.png the second argument
# would be long_name_frame.

IMAGE_FOLDER=results/$1/test_latest/images
FAKE_VIDEO=results/$1/test_latest/fake_B.mp4
SIM_VIDEO=results/$1/test_latest/real_A.mp4
STACKED_VIDEO=results/$1/test_latest/combined.mp4

ffmpeg -r:v 30 -i "$IMAGE_FOLDER/$2%06d_fake_B.png" -codec:v libx264 -preset veryslow -pix_fmt yuv420p -vf scale=1280:650 -crf 28 -an $FAKE_VIDEO
ffmpeg -r:v 30 -i "$IMAGE_FOLDER/$2%06d_real_A.png" -codec:v libx264 -preset veryslow -pix_fmt yuv420p -vf scale=1280:650 -crf 28 -an $SIM_VIDEO
ffmpeg -i $SIM_VIDEO -i $FAKE_VIDEO -filter_complex vstack=inputs=2 $STACKED_VIDEO
