References > Names
==================

Let's consider:

>>> nums = [1, 2, 3]
>>> x = nums[1]

Now the name *x* and *nums[1]* both **reference** the same value.

**What we have just found out about names also applies to references!**

What are references?
--------------------

* Object attributes
* List elements
* Dict values
* Anything on the left side of an assignment...

.. topic:: Assignments

   .. doctest::
      :options: +SKIP

      >>> x = ...
      >>> def ...
      >>> for x in ...:
      >>> import x
      >>> ...

Ok. Interesting. So what?
-------------------------

Every assignment assigns a value to a **name**. Understanding this is crucial to grasping Python's behavior!

Let's see why:

>>> class DummyBase:
...     def __init__(self, val):
...         self.value = val
...
...     def __repr__(self):
...         return f"{self.__class__.__name__}({self.value})"
...
>>> class Dummy1(DummyBase):
...     def __iadd__(self, other):
...         self.value += other.value
...         return self
...
>>> class Dummy2(DummyBase):
...     def __iadd__(self, other):
...         return Dummy2(self.value + other.value)

Let's consider:

>>> dummies = [Dummy1(1), Dummy2(1)]
>>> for d in dummies:
...     d += DummyBase(10)
...
>>> dummies
[Dummy1(11), Dummy2(1)]

**What's going on here?**

At each iteration the name **d** is assigned to the value the **item in the list** is assigned to.
So the for loop is equivalent to:

>>> d = dummies[0]  # Dummy1(1)
>>> d += DummyBase(10)  # Modified in place
>>> d = dummies[1]  # Dummy2(1)
>>> d += DummyBase(10)  # New instance
