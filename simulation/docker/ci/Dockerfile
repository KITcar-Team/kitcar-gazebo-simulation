FROM ros:noetic-robot-focal
ARG KITCAR_REPO_PATH=/home/kitcar/

COPY init/* /

# Ensure that ROS packages are up-to-date
RUN apt-get update && apt-get upgrade -y

# Install packages that are necessary for Gazebo to properly work in a Docker container
RUN apt-get update && apt-get install -y \
dbus dbus-x11 libasound2 libasound2-plugins alsa-utils \
alsa-oss pulseaudio pulseaudio-utils \
xvfb

# Install packages from packages file.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive xargs --arg-file=/packages_focal.txt apt-get install -y

# Install pip packages
RUN pip3 install --upgrade --upgrade-strategy eager --no-warn-script-location \
    -r /requirements.txt

# Reduce image size by removing apt cache
RUN rm -rf /var/lib/apt/lists/*

# Source ROS within the bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Also add and build kitcar-rosbag
# Needed to record rosbags in CI
COPY kitcar-rosbag $KITCAR_REPO_PATH/kitcar-rosbag
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd ${KITCAR_REPO_PATH}/kitcar-rosbag && catkin_make"
RUN echo "source ${KITCAR_REPO_PATH}/kitcar-rosbag/devel/setup.bash --extend" >> ~/.bashrc
