Multiple Inheritance
====================

Python also allows for a class to inherit from multiple base classes.
Adding to our existing :py:class:`BaseClass` we can now define another class:

>>> class BaseClass:
...      def __init__(self, name: str):
...          self.name = name
...      def say_hi(self):
...          print(f"I'm {self.name} of type {type(self)}")
...      def short_desc(self) -> str:
...          return f"BaseClass({self.name})"
>>> class SecondBaseClass:
...     def say_ho(self) -> str:
...         return f"I don't have a name, but the type {type(self)}."
...     def short_desc(self) -> str:
...         return f"SecondBaseClass()"

Let's create a class that inherits from both classes:

>>> class DoubleChildClass(BaseClass, SecondBaseClass):
...     pass

As to expect, we can access the functions of **both** base classes:

>>> double_child = DoubleChildClass("double_child_foo")
>>> double_child.say_hi()
I'm double_child_foo of type <class 'DoubleChildClass'>
>>> double_child.say_ho()
"I don't have a name, but the type <class 'DoubleChildClass'>."

However, :py:func:`short_desc` is defined in both base classes.
What happens if we attempt to call :py:func:`short_desc`?

>>> double_child.short_desc()
'BaseClass(double_child_foo)'

If there are two base classes, the first base class is prioritized.

Method Resolution Order
-----------------------

What if the inheritance gets more complicated?

>>> class A: pass
>>> class B: pass
>>> class C(A, B): pass
>>> class D(A, B): pass
>>> class E(C, D): pass
>>> class F(A, B): pass
>>> class G(A, B): pass
>>> class H(F, G): pass
>>> class I(E, H): pass

Whose functions will be prioritized? The dependency graph is a puzzle:

.. graphviz::

  digraph first {
    rankdir=UD;
    C -> A, B;
    D -> A, B;
    E -> C, D;
    F -> A, B;
    G -> A, B;
    H -> F, G;
    I -> E, H;
  }

But help is on the way:

>>> help(I)  # doctest: +NORMALIZE_WHITESPACE
Help on class I in module builtins:
<BLANKLINE>
class I(E, H)
 |  Method resolution order:
 |      I
 |      E
 |      C
 |      D
 |      H
 |      F
 |      G
 |      A
 |      B
 |      object
 |
 |  Data descriptors inherited from A:
 |
 |  __dict__
 |      dictionary for instance variables (if defined)
 |
 |  __weakref__
 |      list of weak references to the object (if defined)
<BLANKLINE>

 The **method resolution order** defines which base class comes next when looking for attributes and functions.
Python uses the `C3 <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.19.3910&rep=rep1&type=pdf>`_ algorithm to linearize the dependency graph.


