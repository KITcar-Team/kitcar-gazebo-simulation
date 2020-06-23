#!/bin/bash

# Gazebo does not seem to like docker containers.
# This script can be used to start an xserver and pulseaudio.
# Thats usually necessary for Gazebo to work.

# Install necessary packages
sudo apt-get update && apt-get install -y \
dbus dbus-x11 libasound2 libasound2-plugins alsa-utils \
alsa-oss pulseaudio pulseaudio-utils \
xvfb

export DISPLAY=:1.0

# So that dbus-daemon can create /var/run/dbus/system_bus_socket
sudo mkdir -p /var/run/dbus

sleep 1

Xvfb :1 -screen 0 1600x1200x16 &

dbus-daemon --system --fork &

sleep 1

pulseaudio -D &>/dev/null & disown
