from simulation.utils.geometry import Polygon
from typing import Iterable


def polygon_list_almost_equal(list1: Iterable[Polygon], list2: Iterable[Polygon]):
    list1 = list1.copy()
    list2 = list2.copy()

    def approx_equal(poly1, poly2):
        return (poly1.intersection(poly2).area / poly1.union(poly2).area) > 0.99

    for poly in list1:
        idx = next((idx for idx, p in enumerate(list2) if approx_equal(poly, p)), -1)

        if idx >= 0:
            print(f"Polygon: found {poly}.")
            del list2[idx]
        else:
            print(f"Polygon: {poly} not found in {list2}")
            return False
    return len(list2) == 0


def assert_msgs_for_pos(speaker, frame, *msg_types):
    """Ensure that the speaker returns the correct msg when the vehicle is at a
    given position."""
    msg_types = list(msg_types)
    speaker.car_frame = frame
    for t in speaker.speak():
        print(f"received msg {t}")
        try:
            idx = msg_types.index(t.type)
            del msg_types[idx]
        except ValueError:
            return False
    else:
        return len(msg_types) == 0
