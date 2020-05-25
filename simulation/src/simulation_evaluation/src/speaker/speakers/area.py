import functools
from simulation.utils.geometry import Polygon

from typing import Callable, List, Any

from simulation.src.simulation_evaluation.src.speaker.speakers.speaker import Speaker

from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_groundtruth.msg import Section as SectionMsg, Lane as LaneMsg

from . import export

import simulation.utils.road.sections.type as road_section_type


@export
class AreaSpeaker(Speaker):
    """Check in which area of the road the vehicle is (e.g. right corridor, parking lot)."""

    def __init__(
        self,
        *,
        section_proxy: Callable[[], List[SectionMsg]],
        lane_proxy: Callable[[int], LaneMsg],
        parking_proxy: Callable[[int], Any],
        min_wheel_count: int,
        area_buffer: int
    ):
        """Initialize area speaker.

        Args:
            section_proxy: Returns all sections when called.
            lane_proxy: Returns a LaneMsg for each section.
            parking_proxy: function which returns parking msg in a section.
            min_wheel_count: minimum number of wheels inside an area,
            that the car is considered to be in that area
            area_buffer: all areas are buffered that the car can be on the edge
        """
        super().__init__(section_proxy=section_proxy, lane_proxy=lane_proxy)

        self.get_parking_msgs = parking_proxy
        self.min_wheel_count = min_wheel_count
        self.area_buffer = area_buffer

    @property
    @functools.lru_cache()
    def left_corridor(self) -> Polygon:
        """Concatenated left corridor from all sections."""
        return Polygon(
            Polygon(self.left_line, self.middle_line)
            .buffer(self.area_buffer)
            .exterior.coords
        )

    @property
    @functools.lru_cache()
    def right_corridor(self) -> Polygon:
        """Concatenated right corridor from all sections."""
        return Polygon(
            Polygon(self.middle_line, self.right_line)
            .buffer(self.area_buffer)
            .exterior.coords
        )

    @property
    @functools.lru_cache()
    def parking_lots(self) -> List[Polygon]:
        """Return all parking bays as a list of polygons."""
        parking_polygons = []
        for sec in self.sections:
            if not sec.type == road_section_type.PARKING_AREA:
                continue
            srv = self.get_parking_msgs(id=sec.id)

            def polygons_from_msg(msg):
                if msg is not None:
                    return (
                        Polygon(
                            Polygon(border.points).buffer(self.area_buffer).exterior.coords
                        )
                        for border in msg.borders
                    )
                else:
                    return []

            parking_polygons += polygons_from_msg(srv.right_msg)
            parking_polygons += polygons_from_msg(srv.left_msg)

        return parking_polygons

    def speak(self):
        """Return a list of SpeakerMsgs.

        With **one** of the following messages.

        Contents:
            * Right lane -> :ref:`Speaker <speaker_msg>`.RIGHT_LANE,
            * right or left lange -> :ref:`Speaker <speaker_msg>`.LEFT_LANE,
            * right, left lane or parking lot -> :ref:`Speaker <speaker_msg>`.PARKING_LOT,
            * None of the above -> :ref:`Speaker <speaker_msg>`.OFF_ROAD
        """
        msgs = super().speak()

        current_type = None  # Keep track of the message type
        area: List[Polygon] = []  # Iteratively add areas (right lane, then left lane, ...)

        priority_list = [
            (self.right_corridor, SpeakerMsg.RIGHT_LANE),
            (self.left_corridor, SpeakerMsg.LEFT_LANE),
        ] + [(lot, SpeakerMsg.PARKING_LOT) for lot in self.parking_lots]
        # Areas and corresponding msg

        for new_area, area_msg in priority_list:
            area.append(new_area)

            # Check how many wheels of the vehicle are in already checked areas.
            if self.wheel_count_inside(*area, in_total=True) >= self.min_wheel_count:
                msg = SpeakerMsg()
                current_type = area_msg
                break
        else:
            # If car is outside of road with any part return off road
            current_type = SpeakerMsg.OFF_ROAD

        msg = SpeakerMsg()
        msg.type = current_type
        return msgs + [msg]
