FROM ros:noetic-robot-focal
# Actual KITCAR_REPO_PATH for which packages are built!
ARG KITCAR_REPO_PATH=/builds/kitcar
# Path where the KITCAR_REPO_PATH is stored at the end of the Dockerfile
# Necessary to prevent gitlab from overwritting it in the ci!
ARG TMP_KITCAR_REPO_PATH=/home/kitcar

COPY init/* /

# Ensure that ROS packages are up-to-date
RUN apt update && apt upgrade -y

# Install packages that are necessary for Gazebo to properly work in a Docker container
RUN apt update && apt install -y \
dbus dbus-x11 libasound2 libasound2-plugins alsa-utils \
alsa-oss pulseaudio pulseaudio-utils \
xvfb

# Install packages from packages file.
RUN apt update && DEBIAN_FRONTEND=noninteractive xargs --arg-file=/packages_focal.txt apt install -y

# Install pip packages
RUN pip3 install \
    --no-cache-dir \
    --upgrade \
    --upgrade-strategy eager \
    --no-warn-script-location \
    -r /requirements.txt \
    -r /requirements_pytorch_cpu.txt

# Reduce image size by removing apt cache
RUN rm -rf /var/lib/apt/lists/*

# Source ROS within the bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Also add and build kitcar-rosbag
# Needed to record rosbags in CI
COPY kitcar-rosbag $KITCAR_REPO_PATH/kitcar-rosbag
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd ${KITCAR_REPO_PATH}/kitcar-rosbag && pip3 install -r requirements.txt && catkin_make"
RUN echo "source ${KITCAR_REPO_PATH}/kitcar-rosbag/devel/setup.bash --extend" >> ~/.bashrc

# Copy the simulation into a different directory because /builds is removed by gitlab!
# The files are copied back into /builds within the CI!
RUN mkdir -p ${TMP_KITCAR_REPO_PATH} && mv ${KITCAR_REPO_PATH}/* ${TMP_KITCAR_REPO_PATH}
