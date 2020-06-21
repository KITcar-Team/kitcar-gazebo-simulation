#!/bin/bash

# Gazebo does not seem to like docker containers.
# This script can be used to start an xserver and pulseaudio.
# Thats usually necessary for Gazebo to work.

sleep 1

Xvfb :1 -screen 0 1600x1200x16 &

dbus-daemon --system --fork &

sleep 1

pulseaudio -D &>/dev/null & disown
