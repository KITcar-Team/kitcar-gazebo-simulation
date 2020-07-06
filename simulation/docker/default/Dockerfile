ARG PARENT_TAG=focal
FROM git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation/ci:${PARENT_TAG}
ARG KITCAR_REPO_PATH=/home/kitcar/

# Update bashrc
RUN echo "export KITCAR_REPO_PATH=${KITCAR_REPO_PATH}" >> ~/.bashrc
RUN echo "export CAR_NAME=dr_drift" >> ~/.bashrc

# Copy kitcar-gazebo-simulation
COPY kitcar-gazebo-simulation $KITCAR_REPO_PATH/kitcar-gazebo-simulation
# Remove files from possible previous build
RUN rm -rf ${KITCAR_REPO_PATH}/kitcar-gazebo-simulation/simulation/devel ${KITCAR_REPO_PATH}/kitcar-gazebo-simulation/simulation/build
# Run kitcar gazebo's init script
# This also source gazebo within the bashrc!
RUN yes 1 | bash ${KITCAR_REPO_PATH}kitcar-gazebo-simulation/init/init.sh || true

# Build the simulation
RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && cd ${KITCAR_REPO_PATH}/kitcar-gazebo-simulation/simulation && catkin_make"
