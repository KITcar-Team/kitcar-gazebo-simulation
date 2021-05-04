FROM ros:noetic-robot-focal
ARG KITCAR_REPO_PATH=/home/kitcar
ENV KITCAR_REPO_PATH=${KITCAR_REPO_PATH}
RUN mkdir -p ${KITCAR_REPO_PATH}

COPY init/* /

# Install apt packages
# 1. Update and upgrade the system
# 2. Install packages necessary for running the simulation in a docker container
# 3. Install packages from init/packages_focal.txt
# 4. Remove apt cache
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
    dbus dbus-x11 libasound2 libasound2-plugins \
    alsa-utils alsa-oss pulseaudio pulseaudio-utils xvfb && \
    DEBIAN_FRONTEND=noninteractive xargs --arg-file=/packages_focal.txt apt install -y && \
    rm -rf /var/lib/apt/lists/*

# Install pip packages
RUN pip3 install \
    --no-cache-dir \
    --upgrade \
    --upgrade-strategy eager \
    --no-warn-script-location \
    -r /requirements.txt \
    -r /requirements_pytorch_cpu.txt

# Also add and build kitcar-rosbag
# Needed to record rosbags in CI
COPY kitcar-rosbag $KITCAR_REPO_PATH/kitcar-rosbag
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd ${KITCAR_REPO_PATH}/kitcar-rosbag && pip3 install -r requirements.txt && catkin_make"
