Basics
==========================

Let's define a simple class:

>>> class BaseClass:
...     def __init__(self, name: str):
...         self.name = name
...     def say_hi(self):
...         print(f"I'm {self.name} of type {type(self)}")
...     def short_desc(self) -> str:
...         return f"BaseClass({self.name})"

and inspect an instance of :py:class:`BaseClass`:

>>> base = BaseClass("base_foo")
>>> base.say_hi()
I'm base_foo of type <class 'BaseClass'>
>>> base.short_desc()
'BaseClass(base_foo)'

Nothing special so far. How about a child class?

>>> class ChildClass(BaseClass):
...     pass

Instances of :py:class:`ChildClass` inherit functions and attributes of their base classes.

>>> child = ChildClass("child_foo")
>>> child.say_hi()
I'm child_foo of type <class 'ChildClass'>



super()
-------

The :py:func:`short_desc` function doesn't make sense anymore:

>>> child.short_desc()
'BaseClass(child_foo)'

We can fix it by overwriting :py:func:`short_desc`:

>>> class ChildClass(BaseClass):
...    def short_desc(self) -> str:
...        return "Child of " + super().short_desc()

The :py:func:`super`-function can be used to access the attributes and functions of the base class.

>>> child = ChildClass("child_foo_fixed")
>>> child.short_desc()
'Child of BaseClass(child_foo_fixed)'
