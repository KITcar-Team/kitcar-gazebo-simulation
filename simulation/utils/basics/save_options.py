from typing import Any, Dict

import yaml


class SaveOptions:
    def save_as_yaml(
        self, file_path: str, custom_dict: Dict[str, Any] = None, dumper=yaml.SafeDumper
    ):
        """Save to file as yaml.

        This dumps the complete class dict to a yaml file.

        Args:
            file_path: Path to file.
        """
        if custom_dict is None:
            custom_dict = dict(self.__dict__)
        with open(file_path, "w+") as file:
            yaml.dump(custom_dict, file, Dumper=dumper, default_flow_style=False)
