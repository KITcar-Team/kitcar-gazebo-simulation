import yaml


class SaveOptions:
    def save_as_yaml(self, file_path: str):
        """Save to file as yaml.

        This dumps the complete class dict to a yaml file.

        Args:
            file_path: Path to file.
        """
        with open(file_path, "w+") as file:
            yaml.dump(
                dict(self.__dict__), file, Dumper=yaml.SafeDumper, default_flow_style=False
            )
