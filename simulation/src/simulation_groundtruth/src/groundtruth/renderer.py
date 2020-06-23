from dataclasses import dataclass, field
from simulation.utils.geometry import Vector, Transform
from typing import Tuple, Set, DefaultDict, List, Dict, Callable
import itertools
from collections import defaultdict
import os
from simulation.utils.road.road import Road

from simulation.utils.road.renderer.tile import Tile
from time import gmtime, strftime
import time as time_module
import yaml

import rospy

import shutil
from contextlib import suppress

from simulation_groundtruth.msg import GroundtruthStatus


def current_pretty_time():
    return strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())


@dataclass
class PreviousRendering:
    DEFAULT_FILE_NAME = "previous_rendering.yaml"
    seed: str
    tile_size: Tuple[float, float]
    tile_resolution: Tuple[float, float]
    tiles: Dict[str, str] = field(default_factory=dict)
    time: float = field(default_factory=time_module.time)
    pretty_time: str = field(default_factory=current_pretty_time)

    @classmethod
    def load(cls, dir: str) -> "PreviousRendering":
        """Load information about previous rendering from file.

        Args:
            dir: Path to directory that contains previous rendering.
        """
        file_path = os.path.join(dir, PreviousRendering.DEFAULT_FILE_NAME)
        try:
            with open(file_path, "r") as file:
                prev = yaml.load(file.read(), Loader=yaml.FullLoader)
            assert (
                type(prev) == PreviousRendering
            ), "Information of previous rendering is corrupt. Will be ignored in the following!"
            return prev
        except Exception as e:
            print(e)
            pass

    def save(self, dir: str):
        """Store information about the current rendering.

        Args:
            dir: Path to directory that contains current rendering.
        """
        file_path = os.path.join(dir, PreviousRendering.DEFAULT_FILE_NAME)
        with open(file_path, "w+") as file:
            file.write(yaml.dump(self))

    @classmethod
    def delete(cls, dir: str):
        """Delete directory containing all materials of previous renderings.

        Args:
            dir: Path to directory that contains previous rendering.
        """
        with suppress(FileNotFoundError):
            shutil.rmtree(dir)


@dataclass
class Renderer:
    road: Road
    remove_model: Callable[[str], None]
    spawn_model: Callable[[str], None]
    pause_gazebo: Callable[[], None]
    unpause_gazebo: Callable[[], None]
    info_callback: Callable[[int, int, int], None]
    """Function that is called, when the renderer's state changes."""
    tile_size: Vector = Vector(2, 2)
    tile_resolution: Vector = Vector(512, 512)

    @property
    def roads_path(self) -> str:
        return os.path.join(
            os.environ.get("KITCAR_REPO_PATH"),
            "kitcar-gazebo-simulation",
            "simulation",
            "models",
            "env_db",
        )

    @property
    def materials_path(self) -> str:
        return os.path.join(self.roads_path, f".{self.road._name}",)

    @property
    def road_file_path(self) -> str:
        return os.path.join(self.roads_path, f"{self.road._name}.py",)

    # Previous
    @property
    def prev_rendering(self) -> PreviousRendering:
        return PreviousRendering.load(self.materials_path)

    def _prev_rendering_available(self) -> bool:
        return (
            self.prev_rendering is not None
            and self.road.use_seed
            and self.prev_rendering.seed == self.road._seed
            and self.prev_rendering.time > os.path.getmtime(self.road_file_path)
        )

    def _load_prev_tiles(self) -> List[Tile]:
        prev_rend = self.prev_rendering

        tiles = [
            Tile(
                key,
                size=self.tile_size,
                resolution=self.tile_resolution,
                road_folder_name=f".{self.road._name}",
                id=id,
                already_rendered=True,
            )
            for key, id in prev_rend.tiles.items()
        ]
        return tiles

    def save_state(self, tiles: List[Tile]):
        current = PreviousRendering(
            seed=self.road._seed if self.road.use_seed else None,
            tile_resolution=(self.tile_resolution.x, self.tile_resolution.y),
            tile_size=(self.tile_size.x, self.tile_size.y),
        )

        for tile in tiles:
            current.tiles[tile.index] = tile.id

        current.save(self.materials_path)

    # New
    def _create_new_tiles(self) -> List[Tile]:
        sections = self.road.sections

        active_tiles: DefaultDict[Tuple[int, int], Set[int]] = defaultdict(set)

        tile_poly = Tile(
            index=(0, 0), size=self.tile_size, resolution=self.tile_resolution
        ).frame

        for sec in sections:
            box = sec.get_bounding_box()
            minx, miny, maxx, maxy = box.bounds

            tiles_x = range(
                int(minx / self.tile_size.x) - 2, int(maxx / self.tile_size.x) + 2,
            )
            tiles_y = range(
                int(miny / self.tile_size.y) - 2, int(maxy / self.tile_size.y) + 2,
            )

            tile_keys = itertools.product(tiles_x, tiles_y)

            for key in tile_keys:
                if (
                    Transform([key[0] * self.tile_size.x, key[1] * self.tile_size.y], 0)
                    * tile_poly
                ).intersects(box):
                    active_tiles[key].add(sec.id)

        tiles = [
            Tile(
                key,
                sections={sec_id: sections[sec_id] for sec_id in secs},
                size=self.tile_size,
                resolution=self.tile_resolution,
                road_folder_name=f".{self.road._name}",
            )
            for key, secs in active_tiles.items()
        ]
        return tiles

    def _remove_displayed_tiles(self, model_names: List[str]):
        all_tile_names = [model_name for model_name in model_names if "tile" in model_name]

        if len(all_tile_names) > 0:
            rospy.loginfo(
                f"Removing currently displayed road. {len(all_tile_names)} tiles."
            )
            for tile_name in all_tile_names:
                self.remove_model(tile_name)
            rospy.logdebug("Removed all displayed tiles.")

    # Connections to the outside
    def interrupt(self):
        self.stop_requested = True

    def load(self, model_names: List[str]):
        self.stop_requested = False

        self.info_callback(
            status=GroundtruthStatus.REMOVE_OLD_TILES, processed_tiles=0, number_of_tiles=0
        )
        rospy.loginfo(f"Starting to render {self.road._name}.")
        self._remove_displayed_tiles(model_names)

        if self.stop_requested:
            self.info_callback(status=GroundtruthStatus.READY)
            return

        rospy.loginfo(f"Loading ground tiles for {self.road._name}.")

        self.pause_gazebo()

        try:
            self.info_callback(GroundtruthStatus.RENDER_NEW_TILES)
            if self._prev_rendering_available():
                tiles = self._load_prev_tiles()
            else:
                # Delete old tiles
                PreviousRendering.delete(self.materials_path)
                tiles = self._create_new_tiles()

            self.info_callback(number_of_tiles=len(tiles))
            rospy.loginfo(f"Successfully loaded {len(tiles)} tiles for {self.road._name}.")
            rospy.loginfo(f"Start to display {len(tiles)} tiles for {self.road._name}.")

            for i, tile in enumerate(tiles):
                if self.stop_requested:
                    return
                rospy.loginfo(f"Render tile {i + 1}/{len(tiles)}.")
                if not tile.already_rendered:
                    tile.render_to_file(roads_path=self.roads_path)

                if self.stop_requested:
                    return

                self.spawn_model(tile.get_model_string())
                self.info_callback(processed_tiles=i + 1, number_of_tiles=len(tiles))

            rospy.loginfo(f"Successfully rendered {self.road._name}.")
            self.save_state(tiles)
        finally:
            self.info_callback(GroundtruthStatus.READY)
            self.unpause_gazebo()
