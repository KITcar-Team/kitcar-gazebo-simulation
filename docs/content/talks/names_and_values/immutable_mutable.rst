Mutable vs. Immutable
=====================

Let's look at two more examples:

>>> str1 = "hello"
>>> str2 = str1
>>> str2 += " world"

What values are **str1** and **str2** assigned to?

>>> str1
'hello'
>>> str2
'hello world'

That's not too surprising. Let's do the same with lists:

>>> list1 = ["hello"]
>>> list2 = list1
>>> list2 += [" world"]
>>> list1
['hello', ' world']
>>> list2
['hello', ' world']

**Wait. What happened??**

After initializing the strings and lists the situation is:

.. graphviz::

  digraph first {
    rankdir=LR;
    hello [label = "hello", style=dotted, shape=polygon];
    "str1" -> hello [weight=100];
    "str2" -> hello;
  }

|

.. graphviz::

  digraph first {
    rankdir=LR;
    hello [label = "[\"hello\"]", style=dotted, shape=polygon];
    "list1" -> hello [weight=100];
    "list2" -> hello;
  }

|

Where's the difference then??

**1)** Strings are **immutable** objects.

Executing

>>> str2 += " world"

.. graphviz::

  digraph first {
    rankdir=LR;
    hello [label = "hello", style=dotted, shape=polygon];
    helloworld [label = "hello world", style=dotted, shape=polygon];
    "str2" -> helloworld;
    "str1" -> hello;
  }

|

creates a new string!

**2)** Lists are **mutable** objects. This means that a list can be modified:

>>> list2 += [" world"]

does **not** create a new list! The list is modified in place. Therefore both the names **list1** and **list2** still have the same value.

.. graphviz::

  digraph first {
    rankdir=LR;
    hello [label = "[\"hello\", \" world\"]", style=dotted, shape=polygon];
    "list1" -> hello [weight=100];
    "list2" -> hello;
  }

However:

>>> list1 = ["hello"]
>>> list2 = list1
>>> list2 = list2 + [" world"]
>>> list1
['hello']
>>> list2
['hello', ' world']

does create a new list. 

.. graphviz::

  digraph first {
    rankdir=LR;
    hello [label = "[\"hello\"]", style=dotted, shape=polygon];
    helloworld [label = "[\"hello\", \" world\"]", style=dotted, shape=polygon];
    "list1" -> hello [weight=100];
    "list2" -> helloworld;
  }

*There's something shady going on...*



Immutable types
---------------

* Cannot change their value
* **ints, floats, strings, tuples**

Mutable types
-------------

* Can change their value
* All names that have the **same** value see the change! *(Same value does not mean equal value!!)*
* Behavior when performing operations depends on the object's implementation!
  E.g. in the earlier example

>>> list2 += [" world"]

is actually

>>> list2.__iadd__(["world"])  # doctest: +SKIP

. So the behavior depends on the implementation of **__iadd__**. However

>>> list2 = list2 + [" world"]

is actually

>>> list2.__add__(["world"])  # doctest: +SKIP

. So the behavior depends on the implementation of **__add__**.
