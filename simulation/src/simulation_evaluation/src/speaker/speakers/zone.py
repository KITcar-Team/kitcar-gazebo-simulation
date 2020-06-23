#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Class definition of the ZoneSpeaker."""

from simulation.utils.geometry import Point

# Messages
from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_groundtruth.msg import (
    Section as SectionMsg,
    Lane as LaneMsg,
    LabeledPolygon as LabeledPolygonMsg,
)
from typing import Callable, List, Any, Tuple

import simulation_groundtruth.srv as groundtruth_srv
import bisect

import functools

import simulation.utils.road.sections.type as road_section_type

from simulation.src.simulation_evaluation.src.speaker.speakers.speaker import Speaker

from . import export


@export
class ZoneSpeaker(Speaker):
    """Information about the zone of the road the car is in."""

    def __init__(
        self,
        *,
        section_proxy: Callable[[], List[SectionMsg]],
        lane_proxy: Callable[[int], LaneMsg],
        obstacle_proxy: Callable[[int], List[LabeledPolygonMsg]],
        parking_proxy: Callable[[int], Any],
        intersection_proxy: Callable[[int], Any],
        overtaking_buffer: float = 2,
        start_zone_buffer: float = 1,
        end_zone_buffer: float = 1.5,
        yield_distance: Tuple[float, float] = (-0.6, -0.2),
    ):
        """Initialize zone speaker.

        Args:
            section_proxy: Returns all sections when called.
            lane_proxy: Returns a LaneMsg for each section.
            obstacle_proxy: function which returns obstacles in a section.
            parking_proxy: function which returns parking msg in a section.
            intersection_proxy: function which returns intersection msg in a section.
            parking_spot_buffer: buffer around parking spots in which a parking attempt \
                    is also accepted
            overtaking_buffer: buffer around obstacles that the car is allowed to overtake
            start_zone_buffer: beginning of the road that is considered as a start zone
            end_zone_buffer: end of the road that is considered as the end
            yield_distance: interval before intersections that the vehicle must yield in
        """
        super().__init__(
            section_proxy=section_proxy,
            lane_proxy=lane_proxy,
            obstacle_proxy=obstacle_proxy,
            intersection_proxy=intersection_proxy,
        )
        self.get_parking_msgs = parking_proxy

        self.overtaking_buffer = overtaking_buffer
        self.start_zone_buffer = start_zone_buffer
        self.end_zone_buffer = end_zone_buffer
        self.yield_distance = yield_distance

        # Get total length.
        self.total_length = self.middle_line.length

    @property
    @functools.lru_cache()
    def overtaking_zones(self) -> List[Tuple[float, float]]:
        """Intervals in which the car is allowed to overtake \
                along the :py:attr:`Speaker.middle_line`.
        """
        # Get all obstacle polygons
        obstacles = list(
            obstacle
            for sec in self.sections
            if sec.type != road_section_type.PARKING_AREA
            for obstacle in self.get_obstacles_in_section(sec.id)
        )

        # Intervals where polygons are along the middle line
        intervals = list(self.get_interval_for_polygon(obs) for obs in obstacles)

        if len(intervals) == 0:
            return []

        zone_intervals = [
            (
                intervals[0][0] - self.overtaking_buffer,
                intervals[0][1] + self.overtaking_buffer,
            )
        ]
        for start, end in intervals[1:]:
            last = zone_intervals[-1]
            # If the start of this section and end of the last overtaking zone
            # overlap the last interval is extended
            if start - self.overtaking_buffer < last[1]:
                zone_intervals[-1] = (last[0], end + self.overtaking_buffer)
            # Else a new interval is added
            else:
                zone_intervals.append(
                    (start - self.overtaking_buffer, end + self.overtaking_buffer)
                )
        # import rospy
        # rospy.loginfo(f"Obstacle zones: {zone_intervals}")

        return zone_intervals

    def _intersection_yield_zones(self, rule: int) -> List[Tuple[float, float]]:
        """Intervals in which the car is supposed to halt/stop (in front of intersections).

        Args:
            rule: only intersections with this rule are considered
        """
        intervals = []

        for sec in self.sections:
            if sec.type != road_section_type.INTERSECTION:
                continue
            # Get arclength of the last point of the middle line
            # at the intersection south opening
            intersection_msg = self.get_intersection(sec.id)
            arc_length = self.middle_line.project(
                Point(intersection_msg.south.middle_line[-1])
            )
            if intersection_msg.rule == rule:
                intervals.append(
                    (
                        arc_length + self.yield_distance[0],
                        arc_length + self.yield_distance[1],
                    )
                )

        return intervals

    @property
    @functools.lru_cache()
    def stop_zones(self) -> List[Tuple[float, float]]:
        """Intervals in which the car is supposed to stop (in front of intersections)."""
        return self._intersection_yield_zones(groundtruth_srv.IntersectionSrvResponse.STOP)

    @property
    @functools.lru_cache()
    def halt_zones(self) -> List[Tuple[float, float]]:
        """Intervals in which the car is supposed to halt (in front of intersections)."""
        return self._intersection_yield_zones(groundtruth_srv.IntersectionSrvResponse.YIELD)

    def _inside_any_interval(self, intervals: List[Tuple[float, float]]) -> bool:
        """Determine if the car is currently in any of the given intervals."""
        beginnings = list(interval[0] for interval in intervals)
        endings = list(interval[1] for interval in intervals)

        b_idx = bisect.bisect_left(beginnings, self.arc_length) - 1
        e_idx = bisect.bisect_left(endings, self.arc_length) - 1

        # If the vehicle is in interval x then the beginning is before x
        # and ending is behind x
        return b_idx - e_idx == 1

    def speak(self) -> List[SpeakerMsg]:
        """List of speaker msgs.

        Contents:
            * beginning of road -> :ref:`Speaker <speaker_msg>`.START_ZONE,
              end of road -> :ref:`Speaker <speaker_msg>`.END_ZONE,
              and in between -> :ref:`Speaker <speaker_msg>`.DRIVING_ZONE,
            * close to an obstacle -> :ref:`Speaker <speaker_msg>`.OVERTAKING_ZONE
            * before yield/stop lines \
                    -> :ref:`Speaker <speaker_msg>`.HALT_ZONE/SpeakerMsg.STOP_ZONE,
            * parking area -> :ref:`Speaker <speaker_msg>`.PARKING_ZONE
        """
        msgs = super().speak()

        def append_msg(t: int):
            msg = SpeakerMsg()
            msg.type = t
            msgs.append(msg)

        # Determine if car is in parking zone
        append_msg(
            SpeakerMsg.PARKING_ZONE
            if self.current_section.type == road_section_type.PARKING_AREA
            else SpeakerMsg.NO_PARKING_ZONE
        )

        # Overtaking
        append_msg(
            SpeakerMsg.OVERTAKING_ZONE
            if self._inside_any_interval(self.overtaking_zones)
            else SpeakerMsg.NO_OVERTAKING_ZONE
        )

        # Start/End zone
        if self.arc_length < self.start_zone_buffer:
            append_msg(SpeakerMsg.START_ZONE)
        elif self.arc_length + self.end_zone_buffer < self.total_length:
            append_msg(SpeakerMsg.DRIVING_ZONE)
        else:
            append_msg(SpeakerMsg.END_ZONE)

        # Stop / halt zone
        if self._inside_any_interval(self.halt_zones):
            append_msg(SpeakerMsg.HALT_ZONE)
        elif self._inside_any_interval(self.stop_zones):
            append_msg(SpeakerMsg.STOP_ZONE)
        else:
            append_msg(SpeakerMsg.NO_STOP_ZONE)

        return msgs
