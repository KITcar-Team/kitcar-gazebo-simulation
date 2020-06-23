import collections

# Used to pass all three lines of a section
LineTuple = collections.namedtuple("LineTuple", ("left", "middle", "right"))
"""A line tuple can be created from the lines of a section."""
LineTuple.left.__doc__ = """left (Line): Left line of the section."""
LineTuple.middle.__doc__ = """middle (Line): Middle line of the section."""
LineTuple.right.__doc__ = """right (Line): Right line of the section."""
