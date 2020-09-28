from typing import Any, Dict

import yaml


class InitOptions:
    @classmethod
    def from_dict(cls, **kwargs: Dict[str, Any]):
        """Create instance from relevant keywords in dictionary.

        Instead of passing exactly the arguments defined by the class' __init__,
        this allows to pass a dictionary of many more values and the function will
        then only pass the correct arguments to the class' __init__.

        Example:
            >>> from simulation.utils.basics.init_options import InitOptions
            >>> class SomeClass(InitOptions):
            ...
            ...     def __init__(self, arg1, arg2):
            ...         self.arg1 = arg1
            ...         self.arg2 = arg2
            ...
            ...
            >>> many_args = {"arg1": 1, "arg2": 2, "arg3": 3}
            ... # This will raise an exception because __init__ does not expect arg3!
            >>> try:
            ...     SomeClass(**many_args)
            ... except TypeError:
            ...     pass
            ... # However this will work:
            >>> something = SomeClass.from_dict(**many_args)

        Args:
            **kwargs: Keyword arguments matching the constructor's variables.
        """
        init_keys = cls.__init__.__code__.co_varnames  # Access the init functions arguments
        kwargs = {
            key: kwargs[key] for key in init_keys if key in kwargs
        }  # Select all elements in kwargs, that are also arguments of the init function
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, file_path: str):
        """Load instance from a yaml file.

        Only fields that are defined within the __init__ are loaded.

        Args:
            file_path: Location of the yaml file.
        """
        # Extract information from yaml file as a dict
        with open(file_path) as file:
            specs_dict = yaml.load(file, Loader=yaml.SafeLoader)
        # Select only the information that the class wants
        return cls.from_dict(**specs_dict)
