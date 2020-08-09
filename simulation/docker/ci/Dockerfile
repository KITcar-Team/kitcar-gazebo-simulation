FROM ros:noetic-robot-focal

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
RUN pip3 install --no-warn-script-location -r /requirements.txt

# Reduce image size by removing apt cache
RUN rm -rf /var/lib/apt/lists/*

# Source ROS within the bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
