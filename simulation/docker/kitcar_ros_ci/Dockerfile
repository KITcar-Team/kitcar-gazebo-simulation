ARG PARENT_TAG=ci
FROM git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation:${PARENT_TAG}
ARG KITCAR_REPO_PATH=/builds/kitcar
ARG TMP_KITCAR_REPO_PATH=/home/kitcar

COPY packages.txt /packages.txt

# Install fonts
RUN mkdir -p /usr/local/share/fonts/kitcar/
RUN cp -a "${KITCAR_REPO_PATH}/kitcar-gazebo-simulation/simulation/models/fonts/." /usr/local/share/fonts/kitcar/

# Install packages from packages file.
RUN apt update && DEBIAN_FRONTEND=noninteractive xargs --arg-file=/packages.txt apt install -y

# Reduce image size by removing apt cache
RUN rm -rf /var/lib/apt/lists/*

# Copy the simulation into a different directory because /builds is removed by gitlab!
# The files are copied back into /builds within the CI!
RUN rm -rf ${TMP_KITCAR_REPO_PATH} &&  mv ${KITCAR_REPO_PATH} ${TMP_KITCAR_REPO_PATH}
