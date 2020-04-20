#!/usr/bin/env python3
"""Generate and optionally render a road."""

import sys
import argparse
import xml.dom.minidom
import random
import importlib
import os
import errno

from road import schema
from road.road import Road
from road.renderer import sdf


DEFAULT_ROAD_DIR = os.path.join(
    os.environ.get("KITCAR_REPO_PATH"),
    "kitcar-gazebo-simulation",
    "simulation",
    "models",
    "env_db",
)


def load_road(road_name: str, seed: str = "KITCAR") -> Road:
    """Load road object from file.

    Args:
        road_name: Name of the file containing the road definition.
        seed: Predetermine random values.
    """

    sys.path.append(DEFAULT_ROAD_DIR)

    random.seed(seed)

    road_module = importlib.import_module(road_name, DEFAULT_ROAD_DIR)

    return road_module.road


def create_commonroad(road: Road) -> schema.commonRoad:
    """Create commonroad schema of road."""
    doc = schema.commonRoad()
    doc.commonRoadVersion = "1.0"
    # doc.append(ego_vehicle())
    id = 0
    lanelet_pairs = []
    for p in road.sections:
        export = p.export()
        lanelet_pairs += export.lanelet_pairs
        for obj in export.objects:
            id -= 1
            obj.id = id
            doc.append(obj)

    # adjacents
    for pair in lanelet_pairs:
        pair[0].adjacentLeft = schema.laneletAdjacentRef(
            ref=pair[1].id, drivingDir="opposite"
        )
        pair[1].adjacentLeft = schema.laneletAdjacentRef(
            ref=pair[0].id, drivingDir="opposite"
        )
        pair[0].successor = schema.laneletRefList()
        pair[0].predecessor = schema.laneletRefList()
        pair[1].successor = schema.laneletRefList()
        pair[1].predecessor = schema.laneletRefList()

    # right lanes
    for i in range(len(lanelet_pairs) - 1):
        lanelet_pairs[i][0].successor.lanelet.append(
            schema.laneletRef(ref=lanelet_pairs[i + 1][0].id)
        )
        lanelet_pairs[i + 1][0].predecessor.lanelet.append(
            schema.laneletRef(ref=lanelet_pairs[i][0].id)
        )

    # left lanes
    for i in range(len(lanelet_pairs) - 1, 0, -1):
        lanelet_pairs[i][1].successor.lanelet.append(
            schema.laneletRef(ref=lanelet_pairs[i - 1][1].id)
        )
        lanelet_pairs[i - 1][1].predecessor.lanelet.append(
            schema.laneletRef(ref=lanelet_pairs[i][1].id)
        )

    return doc


def save_commonroad(doc: schema.commonRoad, road_name: str):
    """Save commonroad schema to file."""
    road_path = os.path.join(DEFAULT_ROAD_DIR, "." + road_name, "commonroad.xml")

    if not os.path.exists(os.path.dirname(road_path)):
        try:
            os.makedirs(os.path.dirname(road_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(road_path, "w+") as file:
        doc_parsed = xml.dom.minidom.parseString(doc.toxml())
        prettyfied_xml = doc_parsed.toprettyxml()
        file.write(prettyfied_xml)


def render_commonroad(xml: str, road_name: str):
    """Load commonroad xml from file and render it."""

    road_dir = os.path.join(DEFAULT_ROAD_DIR, "." + road_name)

    generate_args = {
        "xml_content": xml,  # Content of commonroad
        "target_dir": road_dir,  # Directory which will contain the world.sdf file
        "generator_dir": os.path.join(
            os.environ.get("KITCAR_REPO_PATH"),
            "kitcar-gazebo-simulation",
            "simulation",
            "utils",
        ),  # Directory which contains commonroad
        "road_name": road_name,  # Name of road
        "add_vehicle": True,  # Add dr_drift model to world
        "background": False,  # If background wall images should be included
        "segmentation": False,  # If segmentation colors should be used for rendering
        "fast_physics": False,  # If segmentation colors should be used for rendering
    }

    sdf.generate_sdf(**generate_args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a common road file from a roadfile."
    )
    parser.add_argument("road_name", nargs="?")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        road = load_road(road_name=args.road_name, seed=args.seed)
    else:
        road = load_road(road_name=args.road_name)

    doc = create_commonroad(road)
    save_commonroad(doc, args.road_name)
    render_commonroad(doc.toxml(), road_name=args.road_name)
