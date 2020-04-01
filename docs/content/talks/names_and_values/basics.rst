Warm up
==========================

**1)** What happens when

>>> x = 1

is executed?
The **name** "x" is assigned the **value** 1.

.. graphviz::

  digraph first {
    rankdir=LR;
    one [label = "1", style=dotted, shape=polygon];
    x [label="x", shape=circle];
    x -> one;
  }

|

**2)** Let's take a look at

>>> x = 1
>>> y = x
>>> print(y)
1

The name "y" is assigned the value of the name "x".

.. graphviz::

  digraph first {
    rankdir=LR;
    one [label = "1", style=dotted, shape=polygon];
    x [label="x", shape=circle];
    y [label="y", shape=circle];
    x -> one [weight=100];
    y -> one;
  }

Nothing special so far, just terminology.

Lessons
-------------

Fundamental facts behind values and names in Python:

* **Names cannot be assigned to another name.**
* **Assignment never copies data.**
