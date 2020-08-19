import unittest
import math

from simulation.utils.geometry.frame import Frame, validate_and_maintain_frames
from simulation.utils.geometry.transform import Transform
from simulation.utils.geometry.vector import Vector


class ModuleTest(unittest.TestCase):
    def test_connecting_frames(self):
        """Test if frames can be connected."""

        frame_1 = Frame("frame_1")
        frame_2 = Frame("frame_2")
        frame_3 = Frame("frame_3")

        frame_1.connect_to(
            frame_2, transformation_to_frame=Transform([0, 0], math.radians(90))
        )
        frame_2.connect_to(
            frame_3, transformation_to_frame=Transform([0, 0], math.radians(-90))
        )
        frame_1.connect_to(
            frame_3, transformation_to_frame=Transform([0, 0], math.radians(-90))
        )

    def test_transformations(self):
        """Test if frame transformations work as expected.

        Note: This is only tested for vectors because the behavior must be the same
        for all classes that are transformable.
        """
        frame_1 = Frame("frame_1")
        frame_2 = Frame("frame_2")

        frame_1.connect_to(
            frame_2, transformation_to_frame=Transform([0, 0], math.radians(90))
        )

        vec_frame_1 = Vector(1, 0, 0, frame=frame_1)
        vec_frame_2 = frame_2(vec_frame_1)

        self.assertEqual(vec_frame_2, Vector(0, 1, 0, frame=frame_2))

        vec_frame_2 = frame_2(vec_frame_2)
        self.assertEqual(vec_frame_2, Vector(0, 1, 0, frame=frame_2))

    def test_frame_decorator(self):
        """Test if frames are correctly checked and propagated by decorator."""

        frame_1 = Frame("frame_1")
        frame_2 = Frame("frame_2")

        class Framed:
            def __init__(self, frame):
                self._frame = frame

        @validate_and_maintain_frames
        def test_func(*args, **kwargs):
            return kwargs["result"]

        # Test case 1:
        # Two objects, same frame
        args = (Framed(frame_1), Framed(frame_1))
        result = test_func(*args, result=Framed(None))
        self.assertEqual(result._frame, frame_1)

        # Test case 2:
        # Two objects, only one has a frame
        # Result should still have a frame
        args = (Framed(frame_1), Framed(None))
        result = test_func(*args, result=Framed(None))
        self.assertEqual(result._frame, frame_1)

        args = (Framed(None), Framed(frame_2))
        result = test_func(*args, result=Framed(None))
        self.assertEqual(result._frame, frame_2)

        # Test case 3:
        # Objects have different frames
        args = (Framed(frame_1), Framed(frame_2))
        with self.assertRaises(ValueError):
            result = test_func(*args, result=Framed(None))


if __name__ == "__main__":
    unittest.main()
