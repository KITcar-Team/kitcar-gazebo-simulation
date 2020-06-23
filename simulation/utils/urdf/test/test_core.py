"""Tests of the urdf.core module.

The tests written here consist of simple classes that are filled
with values using the hypothesis package.
"""
import functools
import itertools
from dataclasses import dataclass
from typing import Tuple, Iterable

import xml.etree.cElementTree as cET
import xml.etree.ElementTree as ET

from hypothesis import given
import hypothesis.strategies as st

from simulation.utils.geometry import Vector

from simulation.utils.urdf.core import Attribute, XmlObject


def _elements_equal(e1: cET.Element, e2: cET.Element) -> bool:
    """Test if two xml elements are equal."""
    if e1.tag != e2.tag:
        return False
    if e1.text != e2.text:
        return False
    if e1.tail != e2.tail:
        return False
    if e1.attrib != e2.attrib:
        return False
    if len(e1) != len(e2):
        return False
    return all(_elements_equal(c1, c2) for c1, c2 in zip(e1, e2))


def log_parameters(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(
            f"Running {func.__name__} with arguments: {args} and keyword arguments: {kwargs}"
        )
        func(*args, **kwargs)

    return wrapper


@given(
    tag=st.text(),
    attr1_val=st.text(),
    attr2_name=st.text(),
    attr2_val=st.text(),
    float_val=st.floats(),
    str_val=st.text(),
)
@log_parameters
def test_basic_xml_object(tag, attr1_val, attr2_name, attr2_val, float_val, str_val):
    @dataclass
    class TestXml(XmlObject):
        TAG = tag

        attr1: Attribute
        attr2: Attribute

        float_el: float
        str_el: str

    test_xml = TestXml(attr1_val, Attribute(attr2_val, name=attr2_name), float_val, str_val)
    generated_xml = test_xml.create_xml()

    expected_xml = cET.Element(tag, attrib={"attr1": attr1_val, attr2_name: attr2_val})
    cET.SubElement(expected_xml, "float_el").text = str(float_val)
    cET.SubElement(expected_xml, "str_el").text = str_val

    assert _elements_equal(
        generated_xml, expected_xml
    ), "Generated xml does not match the expected_xml."


@given(
    tag=st.text(),
    inner_tag=st.text(),
    attr1_val=st.text(),
    attr2_name=st.text(),
    attr2_val=st.text(),
    float_val=st.floats(),
    str_val=st.text(),
)
@log_parameters
def test_nested_xml_object(
    tag, inner_tag, attr1_val, attr2_name, attr2_val, float_val, str_val
):
    @dataclass
    class InnerTestXml(XmlObject):
        TAG = inner_tag

        attr1: Attribute
        attr2: Attribute

        float_el: float
        str_el: str

    @dataclass
    class TestXml(XmlObject):
        TAG = tag

        sub_el: XmlObject

    test_xml = TestXml(
        sub_el=InnerTestXml(
            attr1_val, Attribute(attr2_val, name=attr2_name), float_val, str_val
        )
    )
    generated_xml = test_xml.create_xml()

    expected_xml = cET.Element(tag)
    inner_expected_xml = cET.SubElement(
        expected_xml, inner_tag, attrib={"attr1": attr1_val, attr2_name: attr2_val}
    )
    cET.SubElement(inner_expected_xml, "float_el").text = str(float_val)
    cET.SubElement(inner_expected_xml, "str_el").text = str_val

    assert _elements_equal(
        generated_xml, expected_xml
    ), "Generated xml does not match the expected_xml."


@given(
    vec_xyz=st.tuples(st.floats(), st.floats(), st.floats()),
    iterable=st.iterables(st.one_of(st.floats(), st.text())),
)
@log_parameters
def test_custom_values(vec_xyz: Tuple[float, float, float], iterable: Iterable):

    iterable, itr_cp = itertools.tee(iterable)

    @dataclass
    class TestXml(XmlObject):
        TAG = "tag"

        vec: Vector = Vector(*vec_xyz)
        itr: Iterable = iterable

    test_xml = TestXml()
    generated_xml = test_xml.create_xml()

    expected_xml = cET.Element("tag")
    cET.SubElement(expected_xml, "vec").text = f"{vec_xyz[0]} {vec_xyz[1]} {vec_xyz[2]}"
    cET.SubElement(expected_xml, "itr").text = " ".join(str(val) for val in itr_cp)

    assert _elements_equal(
        generated_xml, expected_xml
    ), f"Generated {ET.tostring(generated_xml)} not match the expected {ET.tostring(expected_xml)}."


if __name__ == "__main__":
    test_basic_xml_object()
    test_nested_xml_object()
    test_custom_values()
