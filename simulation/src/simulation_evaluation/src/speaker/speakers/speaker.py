"""Base class for all other speakers."""
import bisect
import functools
import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Tuple

from gazebo_simulation.msg import CarState as CarStateMsg
from simulation_evaluation.msg import Speaker as SpeakerMsg
from simulation_groundtruth.msg import LabeledPolygon as LabeledPolygonMsg
from simulation_groundtruth.msg import Lane as LaneMsg
from simulation_groundtruth.msg import Section as SectionMsg

from simulation.utils.geometry import Line, Polygon, Pose, Vector
from simulation.utils.road.sections.line_tuple import LineTuple


@dataclass
class LabeledSpeakerPolygon:
    id_: int
    desc: str
    frame: Polygon
    height: float


class Speaker:
    """Base class for all speakers.

    This class is part of the evaluation pipeline used to automatically evaluate
    the cars behavior in simulation.
    It converts information about the cars state and the predefined groundtruth
    into SpeakerMsg's which serve as inputs for the state machines.

    Information is passed to the speaker by calling the :py:func:`Speaker.listen` function
            with a new CarState msg.
    Output can be retrieved in form of Speaker msgs by calling the .speak function.

    Attributes:
        sections (Callable[[], List[SectionMsg]]): List of all sections as SectionMsgs
        get_lanes (Callable[[int], LaneMsg]): Get LaneMsg for a given section
        get_obstacles (Callable[[int], List[LabeledPolygonMsg]]): Get ObstacleMsg
            for a given section
        get_intersection (Callable[[int], Any]): Get intersections for a given section
    """

    def __init__(
        self,
        *,
        section_proxy: Callable[[], List[SectionMsg]],
        lane_proxy: Callable[[int], LaneMsg],
        obstacle_proxy: Callable[[int], List[LabeledPolygonMsg]] = None,
        surface_marking_proxy: Callable[[int], List[LabeledPolygonMsg]] = None,
        intersection_proxy: Callable[[int], Any] = None,
        sign_proxy: Callable[[int], List[LabeledPolygonMsg]] = None,
    ):
        """Initialize speaker with funtions that can be queried for groundtruth information.

        Args:
            section_proxy: Returns all sections when called.
            lane_proxy: Returns a LaneMsg for each section.
            obstacle_proxy: Optional function which returns obstacles in a section.
            surface_marking_proxy: Optional function which returns surface_markings
                in a section.
            intersection_proxy: Optional function which returns an IntersectionMsg
                for a section. (Only makes sense if the section is an intersection.)
        """
        self.sections = section_proxy().sections
        self.get_lanes = lane_proxy
        self.get_obstacles = obstacle_proxy
        self.get_surface_markings = surface_marking_proxy
        self.get_intersection = intersection_proxy
        self.get_traffic_signs = sign_proxy

    def listen(self, msg: CarStateMsg):
        """Receive information about current observations and update internal values."""
        # Store car frame
        self.car_frame = Polygon(msg.frame)
        self.car_pose = Pose(msg.pose)
        self.car_speed = abs(Vector(msg.twist.linear))

    def speak(self) -> List[SpeakerMsg]:
        """Speak about the current observations."""
        return []

    @functools.cached_property
    def middle_line(self) -> Line:
        """Complete middle line."""
        # Get the middle line of each seaction
        middle_line_pieces = (self.get_road_lines(sec.id).middle for sec in self.sections)
        # Sum it up (and start with empty line)
        return sum(middle_line_pieces, Line())

    @functools.cached_property
    def left_line(self) -> Line:
        """Complete left line."""
        # Get the left line of each seaction
        left_line_pieces = (self.get_road_lines(sec.id).left for sec in self.sections)
        # Sum it up (and start with empty line)
        return sum(left_line_pieces, Line())

    @functools.cached_property
    def right_line(self) -> Line:
        """Complete right line."""
        # Get the right line of each seaction
        right_line_pieces = (self.get_road_lines(sec.id).right for sec in self.sections)
        # Sum it up (and start with empty line)
        return sum(right_line_pieces, Line())

    @property
    def arc_length(self):
        """Position of the car projected on the middle line (== Length driven so far)."""
        return self.middle_line.project(self.car_pose.position)

    @functools.cached_property
    def section_intervals(self) -> List[Tuple[float, float]]:
        """Get (start,end) intervals of all sections."""
        # First extract the individual lengths of each section
        lengths = (self.get_road_lines(sec.id).middle.length for sec in self.sections)

        # The accumulate function computes an inclusive prefix sum
        prefix = list(itertools.accumulate(lengths))

        beginnings = [0] + prefix[:-1]
        endings = prefix

        return list(zip(beginnings, endings))

    @property
    def current_section(self):
        """Get the current section.

        Returns:
            SectionMsg of the current.
        """
        section_beginnings = list(interval[0] for interval in self.section_intervals)

        # Find the current section: the first section which starts before self.arc_length
        # (This will also return a value if arc_length is outside of the sections)
        idx = max(bisect.bisect_left(section_beginnings, self.arc_length) - 1, 0)

        return self.sections[idx]

    @functools.lru_cache()
    def get_road_lines(self, section_id: int) -> LineTuple:
        """Request and return the road lines of a single section.

        Args:
            section_id (int): id of the section

        Returns:
            LineTuple of the section as a named tuple.
        """
        msg = self.get_lanes(section_id).lane_msg

        return LineTuple(Line(msg.left_line), Line(msg.middle_line), Line(msg.right_line))

    def _get_labeled_polygons(
        self, proxy: Callable[[int], List[LabeledPolygonMsg]], section_id: int
    ) -> List[LabeledSpeakerPolygon]:
        """Get all objects inside a section returned by a service.

        Args:
            proxy: Service receiver function
            section_id: id of the section

        Returns:
            List of tuples containing id, description, polygon, and height.
        """
        assert isinstance(section_id, int)

        response = proxy(section_id)

        return [
            LabeledSpeakerPolygon(r.type, r.desc, Polygon(r.frame), r.height)
            for r in response.polygons
        ]

    def get_obstacles_in_section(self, section_id: int) -> List[LabeledSpeakerPolygon]:
        """Get all obstacles in a section."""
        return self._get_labeled_polygons(self.get_obstacles, section_id)

    def get_traffic_signs_in_section(self, section_id: int) -> List[LabeledSpeakerPolygon]:
        """Get all traffic_signs inside section."""
        return self._get_labeled_polygons(self.get_traffic_signs, section_id)

    def get_surface_markings_in_section(
        self, section_id: int
    ) -> List[LabeledSpeakerPolygon]:
        """Get all surface_markings inside a section."""
        return self._get_labeled_polygons(self.get_surface_markings, section_id)

    def get_interval_for_polygon(self, *polygons: Iterable[Polygon]) -> Tuple[float, float]:
        """Get start and end points of the polygons.

        Args:
            polygon: The polygon.

        Returns:
            Start and end point for each polygon as a tuple
        """
        projections = list(
            self.middle_line.project(p)
            for polygon in polygons
            for p in polygon.get_points()
        )

        return (min(projections), max(projections))

    def car_is_inside(
        self, *polygons: Iterable[Polygon], min_wheel_count: int = None
    ) -> bool:
        """Check if the car is currently inside one of the polygons.

        The car can also be in the union of the provided polygons.

        Args:
            polygons: The polygons.
            min_wheel_count: If provided it is enough for a given number of wheels
                to be inside the polygons (e.g. 3 wheels inside parking spot)
        """
        if min_wheel_count:
            return self.wheel_count_inside(*polygons, in_total=True) >= min_wheel_count

        return (
            sum(p.intersection(self.car_frame).area for p in polygons) / self.car_frame.area
            > 0.99
        )

    def wheel_count_inside(
        self, *polygons: Iterable[Polygon], in_total: bool = False
    ) -> int:
        """Get the maximum number of wheels inside one of the polygons.

        Args:
            polygons: The polygons.
            in_total: If true, the number of wheels are summed up
        """
        if len(polygons) == 0:
            return False

        frame_points = set(self.car_frame.get_points())

        # Loop over each point of the car frame and add it 1 to inside
        # if the point is in any of the polygons
        if in_total:
            inside = (
                any(polygon.contains(point) for polygon in polygons)
                for point in frame_points
            )
            return sum(inside)
        else:
            inside = (
                sum(polygon.contains(point) for point in frame_points)
                for polygon in polygons
            )
            return max(inside)

    def car_overlaps_with(self, *polygons: Iterable[Polygon]) -> bool:
        """Decide if the car overlaps with any of the polygons.

        Args:
            polygons: The polygons.
        """
        # True if any of the intersections are positive
        return any(p.intersection(self.car_frame).area > 0 for p in polygons)
