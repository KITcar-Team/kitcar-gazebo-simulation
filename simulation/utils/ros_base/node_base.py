#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ROS node base class with a pythonic parameter interface."""

import rospy
from cachetools import cached, TTLCache
import yaml

from typing import Any, Callable, Dict


class NodeBase:
    """Abstract ROS Node class with additional functionality

    Args:
        name (str): Name of the node
        parameter_cache_time (int) = 1: Duration for which parameters will be cached, for performance
        log_level (int) = rospy.INFO: Loglevel with which the node works.

    A basic node with a subscriber and publisher can be created in the following way:

    >>> from simulation.utils.ros_base.node_base import NodeBase
    >>> class NodeSubClass(NodeBase):
    ...     def __init__(self):
    ...         super(NodeSubClass,self).__init__("node_name") # Important!
    ...         # ...
    ...         self.run() # Calls .start() if self.param.active is True (default: True)
    ...     def start(self):
    ...         # self.subscriber = ...
    ...         # self.publisher = ...
    ...         super().start() # Make sure to call this!
    ...     # Called when deactivating the node by setting self.param.active to false
    ...     # E.g. through command line with: rosparam set .../node/active false
    ...     # or when ROS is shutting down
    ...     def stop(self):
    ...         # self.subscriber.unregister()
    ...         # self.publisher.unregister()
    ...         super().stop() # Make sure to call this!

    Attributes:
        param (ParameterObject): Attribute of type :class:`ParameterObject`,
            which provides an abstraction layer to access ROS parameters.
            The following line shows how to access a ROS parameter in any subclass of called *param_1*:

            >>> self.param.param_1  # doctest: +SKIP
            \'value_1\'

            This is equivalent to:

            >>> rospy.get_param(\'~param_1\')  # doctest: +SKIP
            'value_1'

            Setting a parameter is equally easy:

            >>> self.param.param_2 = \'value_2\'  # doctest: +SKIP

            This is equivalent to:

            >>> rospy.set_param(\'~param_2\', \'value_2\')  # doctest: +SKIP

            The magic begins when parameters are defined in a hierarchical structure.
            After starting a node with the following YAML parameter file:

            .. highlight:: yaml
            .. code-block:: yaml

                car:
                    name: 'dr_drift'
                    size:
                        length: 0.6
                        width: 0.4
                ...

            the cars dimensions can be retrieved just like any other python attribute:

            >>> self.param.car.size.length  # doctest: +SKIP
            0.6

            and changes are also synchronized with ROS:

            >>> rospy.get_param(\"~car/name\")  # doctest: +SKIP
            \'dr_drift\'
            >>> self.param.car.name = \'captain_rapid\'  # doctest: +SKIP
            >>> rospy.get_param(\"~car/name\")  # doctest: +SKIP
            \'captain_rapid\'
    """

    def __init__(
        self, *, name: str, parameter_cache_time: float = 1, log_level: int = rospy.INFO
    ):

        rospy.init_node(name, log_level=log_level)

        # Parameters
        self._parameter_cache = TTLCache(maxsize=128, ttl=parameter_cache_time)
        self.param = ParameterObject(
            ns="~", set_param_func=self._set_param, get_param_func=self._get_param
        )

        # Node is not yet active
        self.__active = False

        # Always call stop on shutdown!
        rospy.on_shutdown(self.__shutdown)

        # Node is by default active
        try:
            self.param.active
        except KeyError:
            self.param.active = True

    def __shutdown(self):
        """Called when ROS is shutting down.

        If the node was active before, self.stop is called.
        """
        if self.__active:
            self.__active = False
            self.stop()

    def _get_param(self, key: str) -> Any:
        """Get (possibly) cached ROS parameter.

        Arguments:
            key (str): Name of the ROS parameter

        Returns:
            If the parameter is in the parameter cache, the cached value is returned.
            Otherwise rospy.get_param(key) is returned.
        """
        # Cached version of rospy.get_param:
        get_cached_param = cached(cache=self._parameter_cache)(rospy.get_param)

        # request param
        return get_cached_param(key)

    def _set_param(self, key: str, value: Any):
        """Set ROS parameter.
        Also the parameter cache is cleared, to prevent incoherence.

        Arguments:
            key (str): Name of the ROS parameter
            value (Any): New value
        """

        # To ensure that there are no cache conflicts
        self._parameter_cache.clear()

        # Set the parameter
        rospy.set_param(key, value)

    def run(self, *, function: Callable = None, rate: float = 1):
        """Helper function, starting the node and shutting it down once ROS signals to.
        Can only be called if the subclass implements start and stop functions.

        Args:
            rate (float): Rate with which to update active/ not active status of the node
            function: Called with a frequency of ``rate`` when node is active
        """
        rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            # Node should be active, but is not.
            if self.param.active and not self.__active:
                self.__active = True
                self.start()
                rospy.loginfo(f"Activating {rospy.get_name()}")
            elif not self.param.active and self.__active:
                self.__active = False
                self.stop()
                rospy.loginfo(f"Deactivating {rospy.get_name()}")

            if self.__active and function:
                function()
            rate.sleep()

    def start(self):
        """Called when activating the node."""
        pass

    def stop(self):
        """Called when deactivating or shutting down the node."""
        pass


class ParameterObject:
    """ROS parameter wrapper to recursively get and set parameters.

    This class enables to access nested parameters within nodes.
    For example in any subclass of NodeBase one can call nested parameters (if they are defined!) in the following way:

    >>> self.param.dict_of_parameters.key  # doctest: +SKIP

    Which is the same as calling:

    >>> rospy.get_param(\"~dict_of_parameters/key\")  # doctest: +SKIP

    This is achieved by overriding the __getattr__ and __setattr__ functions and passing calls through to the


    Arguments:
        ns (str): Namespace this parameter dictionary operates in
        set_param_func (Callable[[str,Any], None]): Callable object which gets called when a parameter is set
        get_param_func(Callable[[str],Any]): Callable object which gets called when a parameter is accessed

    Attributes:
        _ns (str): Namespace of this object
        _set_param (Callable[[str, Any], None]): Called when a new parameter should be set
        _get_param (Callable[[str], Any]): Called when a parameter is requested
    """

    def __init__(
        self,
        *,
        ns: str,
        set_param_func: Callable[[str, Any], None],
        get_param_func: Callable[[str], Any],
    ):
        self._ns = ns
        self._set_param = set_param_func
        self._get_param = get_param_func

    def __getattr__(self, key: str) -> Any:
        """Retrieving a parameter

        Returns:
            Value of the parameter in this namespace with key ``key`` or a ParameterObject in the subnamespace
            of ``key``.

        Raises:
            KeyError if parameter is not found.
        """
        # Load the parameter
        item = self._get_param(f"{self._ns}{key}")
        # If the parameter is a dictionary, a new ParameterObject is returned
        # with the current key appended to the namespace!
        if type(item) == dict:
            return ParameterObject(
                ns=f"{self._ns}{key}/",
                set_param_func=self._set_param,
                get_param_func=self._get_param,
            )
        return item

    def __setattr__(self, key: str, value: Any):
        """Setting the value of an attribute."""

        # No parameter
        if key.startswith("_"):
            object.__setattr__(self, key, value)
            return
        # Parameter
        self._set_param(f"{self._ns}{key}", value)

    def as_dict(self) -> Dict[str, Any]:
        """Return all parameters in current namespace as dicts."""
        return self._get_param(self._ns)

    def __repr__(self) -> str:
        return yaml.dump(self._get_param(self._ns), default_flow_style=False)
