#!/bin/bash

# Install ssh-agent if not already installed, it is required by Docker.
# (change apt-get to yum if you use a CentOS-based image)
command -v ssh-agent || ( apt-get update -y && apt-get install openssh-client -y )

# For Docker builds disable host key checking. Be aware that by adding that
# you are suspectible to man-in-the-middle attacks.
# WARNING: Use this only with the Docker executor, if you use it with shell
# you will overwrite your user's SSH config.
mkdir -p ~/.ssh
[[ -f /.dockerenv ]] && echo -e "Host *\n\tStrictHostKeyChecking no\n\n" > ~/.ssh/config

git config --global user.email "cml.bot@kitcar-team.de"
git config --global user.name "CML Bot"
git remote set-url origin git@git.kitcar-team.de:kitcar/kitcar-gazebo-simulation.git
