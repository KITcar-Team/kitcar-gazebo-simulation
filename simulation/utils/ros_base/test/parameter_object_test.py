#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test if the Parameter object used in NodeBase behaves as expected.
"""
import unittest

from simulation.utils.ros_base.node_base import ParameterObject


class TestParameterObject(unittest.TestCase):
    def setUp(self):
        # Construct a complete parameter object for testing

        dummy_ns = "ns/"
        self.param_dict = {
            "dummy_param": "value",
            "outer_dict": {"inner_param": "inner_value"},
            "very_outer_dict": {"inner_dict": {"very_inner_param": "verry_inner_value"}},
        }

        def get_param(key: str):
            if key.startswith(dummy_ns):
                keys = key.split("/")[1:]
                val = self.param_dict
                for k in keys:
                    val = val[k]
                return val

            return None

        def set_param(key, val):
            self.last_key = key
            self.last_val = val

        self.param_object = ParameterObject(
            ns=dummy_ns, set_param_func=set_param, get_param_func=get_param
        )

    def test_getting_params(self):

        self.assertEqual(self.param_object.dummy_param, self.param_dict["dummy_param"])
        self.assertEqual(
            self.param_object.outer_dict.inner_param,
            self.param_dict["outer_dict"]["inner_param"],
        )
        self.assertEqual(
            self.param_object.very_outer_dict.inner_dict.very_inner_param,
            self.param_dict["very_outer_dict"]["inner_dict"]["very_inner_param"],
        )

    def test_setting_params(self):
        dummy_value = "value"

        # Test if setting a simple parameter works
        self.param_object.parameter1 = dummy_value
        self.assertEqual(self.last_key, "ns/parameter1")
        self.assertEqual(self.last_val, dummy_value)

        self.param_object.outer_dict.inner_param = dummy_value
        self.assertEqual(self.last_key, "ns/outer_dict/inner_param")
        self.assertEqual(self.last_val, dummy_value)


if __name__ == "__main__":
    unittest.main()
