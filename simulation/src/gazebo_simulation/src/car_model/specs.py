import yaml


class Specs:
    """Class that allows to save/load instances from yaml files."""

    @classmethod
    def from_file(cls, file_path: str):
        """Load instance from a yaml file.

        Only fields that are defined within the class __annotations__ are loaded.

        Args:
            file_path: Location of the yaml file.
        """
        # Extract information from yaml file as a dict
        with open(file_path) as file:
            specs_dict = yaml.load(file, Loader=yaml.SafeLoader)
        # Select only the information that the class wants
        keys = cls.__annotations__.keys()
        specs_dict = {k: v for k, v in specs_dict.items() if k in keys}

        return cls(**specs_dict)

    def save(self, file_path: str):
        """Save to file.

        Args:
            file_path: Path to file.
        """
        with open(file_path, "w+") as file:
            yaml.dump(
                dict(self.__dict__), file, Dumper=yaml.SafeDumper, default_flow_style=False
            )
