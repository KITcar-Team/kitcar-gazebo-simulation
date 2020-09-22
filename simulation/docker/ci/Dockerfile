FROM ros:noetic-robot-focal

ARG INSTALL_DOC_PACKAGES=false
ARG INSTALL_ML_PACKAGES=false
ARG INSTALL_NODE_JS=false

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
# Install documentation and machine learning packages as well
RUN if [ "$INSTALL_DOC_PACKAGES" = "true" ]; then pip3 install --no-warn-script-location -r /requirements_documentation.txt; fi
RUN if [ "$INSTALL_ML_PACKAGES" = "true" ]; then pip3 install --no-warn-script-location -r /requirements_machine_learning.txt; fi

RUN if [ "$INSTALL_NODE_JS" = "true" ]; then curl -sL https://deb.nodesource.com/setup_12.x | bash && apt-get update && apt-get install nodejs -y && npm i -g @dvcorg/cml; fi


# Reduce image size by removing apt cache
RUN rm -rf /var/lib/apt/lists/*

# Source ROS within the bashrc
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
