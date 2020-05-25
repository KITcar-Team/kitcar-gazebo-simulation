"""Functions creating Mock objects that behave like road sections."""
from unittest.mock import Mock
from typing import List, Tuple

from simulation.utils.geometry import Line, Polygon
import inspect

from simulation.utils.road.sections.line_tuple import LineTuple

"""A word on what's going on here.

The mock_... functions all contain these lines:

(1) frame = inspect.currentframe()
(2) args, _, _, values = inspect.getargvalues(frame)
(3) parameter_dict = {i: values[i] for i in args}

and are mostly the same otherwise.

Line (1) reads the current frame, line (2) then retrieves the function arguments \
        and their values.
Line (3) creates a dictionary containing the function arguments as keys \
        and their values as values.

Through these three lines, the function arguments are used to specify what attributes \
        (and values) the resulting road section mocks will have.
"""


def _generic_mock(**kwargs):
    """Create a mock object with the provided keyword arguments as attributes."""
    mock = Mock()

    # Treat type_ differently!!
    if "type_" in kwargs:
        # Type is a class variable of road sections,
        # it must be set in a different way
        mock.__class__.TYPE = kwargs["type_"]
        del kwargs["type_"]

    # Obstacles are not polygons but rather classes that have a frame attribute
    if "obstacles" in kwargs and kwargs["obstacles"] is not None:
        for i, obs in enumerate(kwargs["obstacles"]):
            m = Mock()
            m.frame = obs
            kwargs["obstacles"][i] = m

    for key, val in kwargs.items():
        setattr(mock, key, val)
    return mock


def mock_generic_section(
    *,
    id: int,
    type_: int,
    left_line: Line,
    middle_line: Line,
    right_line: Line,
    obstacles: List[Polygon] = None,
) -> Mock:
    """Mock a road section.

    Args:
        id
        type_
        left_line
        middle_line
        right_line
        obstacles
    """

    # The following 3 lines extract all function arguments
    # and values into a dictionary!
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    parameter_dict = {i: values[i] for i in args}

    return _generic_mock(**parameter_dict)


def _mock_spots(spots: List[Tuple[int, Polygon]]) -> List[Mock]:
    mocks = []
    for spot in spots:
        m = Mock()
        m.kind = spot[0]
        m.frame = spot[1]
        mocks.append(m)
    return mocks


def _mock_lots(lots: List[Tuple[Line, List[Tuple[int, Polygon]]]]) -> List[Mock]:
    mocks = []
    for lot in lots:
        m = Mock()
        m.border = lot[0]
        m.spots = _mock_spots(lot[1])
        mocks.append(m)
    return mocks


def mock_parking_section(
    *,
    id: int,
    type_: int,
    left_line: Line,
    middle_line: Line,
    right_line: Line,
    obstacles: List[Polygon] = None,
    left_lots: List[Tuple[Line, List[Tuple[int, Polygon]]]] = None,
    right_lots: List[Tuple[Line, List[Tuple[int, Polygon]]]] = None,
    start_line: Polygon = None,
) -> Mock:
    """Mock a parking section.

    Args:
        id
        type_
        left_line
        middle_line
        right_line
        obstacles
        left_lots: parking lots on the left side;
            each with a border line and a list of spots.
        right_lots: parking lots on the right side;
            each with a border line and a list of spots.
        start_line: frame of the start line.
    """

    # The following 3 lines extract all function arguments
    # and values into a dictionary!
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    parameter_dict = {i: values[i] for i in args}

    # Create spots and borders instead of lots.
    del parameter_dict["left_lots"]
    if left_lots is not None:
        parameter_dict["left_lots"] = _mock_lots(left_lots)

    del parameter_dict["right_lots"]
    if right_lots is not None:
        parameter_dict["right_lots"] = _mock_lots(right_lots)

    return _generic_mock(**parameter_dict)


def mock_intersection(
    *,
    id: int,
    type_: int,
    left_line: Line,
    middle_line: Line,
    right_line: Line,
    obstacles: List[Polygon] = None,
    turn: int = 0,
    rule: int = 0,
    south: LineTuple = None,
    west: LineTuple = None,
    east: LineTuple = None,
    north: LineTuple = None,
) -> Mock:
    """Mock an intersection.

    Args:
        id
        type_
        left_line
        middle_line
        right_line
        obstacles
        turn
        rule
        south
        west
        east
        north
        """

    # The following 3 lines extract all function arguments
    # and values into a dictionary!
    frame = inspect.currentframe()
    args, _, _, values = inspect.getargvalues(frame)
    parameter_dict = {i: values[i] for i in args}

    def extract_line_tuple(lt: LineTuple, ns: str):
        if lt is not None:
            parameter_dict["left_line_" + ns] = lt.left
            parameter_dict["middle_line_" + ns] = lt.middle
            parameter_dict["right_line_" + ns] = lt.right

    extract_line_tuple(south, "south")
    extract_line_tuple(west, "west")
    extract_line_tuple(east, "east")
    extract_line_tuple(north, "north")

    return _generic_mock(**parameter_dict)
