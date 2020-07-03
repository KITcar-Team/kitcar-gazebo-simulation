ARG PARENT_TAG=ci
FROM git.kitcar-team.de:4567/kitcar/kitcar-gazebo-simulation:${PARENT_TAG}

COPY packages.txt /packages.txt

# Copy the simulation into a different directory because /builds is removed by gitlab!
# The files are copied back into /builds within the CI!
RUN mkdir -p /home/kitcar && mv /builds/kitcar/kitcar-gazebo-simulation /home/kitcar/kitcar-gazebo-simulation

# Install packages from packages file.
RUN apt-get update && DEBIAN_FRONTEND=noninteractive xargs --arg-file=/packages.txt apt-get install -y

# Reduce image size by removing apt cache
RUN rm -rf /var/lib/apt/lists/*
