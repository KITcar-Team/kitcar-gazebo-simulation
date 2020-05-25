#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Definition of the SpeedSpeaker class."""
import rospy

from simulation.src.simulation_evaluation.src.speaker.speakers.speaker import Speaker

from simulation_evaluation.msg import Speaker as SpeakerMsg

from typing import List

from . import export


@export
class SpeedSpeaker(Speaker):
    """Publish information about the cars speed."""

    def __init__(
        self,
        time: float = 0,
        speed_factor: float = 10 * 3.6,
        stop_threshold: float = 1,
        halt_time: float = 1,
        stop_time: float = 3,
    ):
        """Initialize speed speaker.

        Args:
            time: current time in seconds
            speed_factor: conversion factor for speed from m/s into wanted unit
            stop_threshold: if vehicle drives slower, it's considered as stopping
            halt_time: time that vehicle needs to not move for a halt
            stop_time: time that vehicle needs to not move for a stop
        """
        self.speed_list = [
            SpeakerMsg.SPEED_1_10,
            SpeakerMsg.SPEED_11_20,
            SpeakerMsg.SPEED_21_30,
            SpeakerMsg.SPEED_31_40,
            SpeakerMsg.SPEED_41_50,
            SpeakerMsg.SPEED_51_60,
            SpeakerMsg.SPEED_61_70,
            SpeakerMsg.SPEED_71_80,
            SpeakerMsg.SPEED_81_90,
            SpeakerMsg.SPEED_91_,
        ]

        # Last time the car had more speed than threshold
        self.speed_time = time

        self.speed_factor = speed_factor
        self.stop_threshold = stop_threshold
        self.halt_time = halt_time
        self.stop_time = stop_time

    def speak(self, current_time: float = None) -> List[SpeakerMsg]:
        """List of SpeakerMsgs with the current speed as the only entry."""
        # Determine current_time
        if current_time is None:
            current_time = rospy.Time.now().to_sec()

        msg = SpeakerMsg()

        speed = self.car_speed * self.speed_factor

        if speed < self.stop_threshold:  # Car is not moving
            stop_time = current_time - self.speed_time

            if stop_time < self.halt_time:
                msg.type = SpeakerMsg.SPEED_0
            elif stop_time < self.stop_time:
                msg.type = SpeakerMsg.SPEED_HALTED
            else:
                msg.type = SpeakerMsg.SPEED_STOPPED

        elif speed > 90:
            msg.type = SpeakerMsg.SPEED_91_
            self.speed_time = current_time
        else:
            idx = int(speed / 10)
            msg.type = self.speed_list[idx]
            self.speed_time = current_time

        return [msg]
