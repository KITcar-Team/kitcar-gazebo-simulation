class InitFromOptions:
    @classmethod
    def from_options(cls, **kwargs: dict):
        """Create instance from relevant keywords in dictionary.

        Args:
            **kwargs (dict): the dict with keys matching the constructor
                variables
        """
        init_keys = cls.__init__.__code__.co_varnames  # Access the init functions arguments
        kwargs = {
            key: kwargs[key] for key in init_keys if key in kwargs
        }  # Select all elements in kwargs, that are also arguments of the init function
        return cls(**kwargs)
