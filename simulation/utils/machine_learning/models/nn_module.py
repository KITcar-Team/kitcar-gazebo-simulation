from unittest.mock import Mock

from torch import nn as nn

# If the documentation is built, the class is substituted with object to prevent
# issues with better-apidoc.
BaseClass = nn.Module
if isinstance(BaseClass, Mock):
    BaseClass = object


class NNModule(BaseClass):
    pass
