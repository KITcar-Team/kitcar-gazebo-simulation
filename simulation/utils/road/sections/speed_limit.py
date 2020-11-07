from dataclasses import dataclass

from simulation.utils.geometry import Point
from simulation.utils.road.config import Config
from simulation.utils.road.sections import SurfaceMarkingRect, TrafficSign


@dataclass
class SpeedLimit:
    x_position: float
    """The position on the road"""
    limit: int
    """The speed limit"""

    @property
    def traffic_sign(self) -> TrafficSign:
        """TrafficSign: The traffic sign for this speed limit."""
        kind = getattr(
            TrafficSign, f"ZONE_{abs(self.limit)}_{'START' if self.limit>0 else 'END'}"
        )

        return TrafficSign(
            kind=kind,
            angle=0,
            center=Point(self.x_position, -Config.road_width - 0.1),
        )

    @property
    def surface_marking(self) -> SurfaceMarkingRect:
        """SurfaceMarking: The surface marking for this speed limit."""
        return SurfaceMarkingRect(
            center=Point(self.x_position, -Config.road_width / 2),
            width=0.2,
            depth=0.4,
            kind=getattr(
                SurfaceMarkingRect,
                f"ZONE_{abs(self.limit)}_{'START' if self.limit>0 else 'END'}",
            ),
        )
