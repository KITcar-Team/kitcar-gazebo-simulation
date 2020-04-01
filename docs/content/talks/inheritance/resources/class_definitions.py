
class BaseClass:
    """Simple BaseClass with a name."""
    def __init__(self, name: str):
        self.name = name
    def say_hi(self):
        print(f"I'm {self.name} of type {type(self)}")
    def short_desc(self) -> str:
        return f"BaseClass({self.name})"

class ChildClass(BaseClass):
    """Simple child class inheriting from BaseClass."""
    def short_desc(self) -> str:
        return "Child of " + super().short_desc()

class SecondBaseClass:
    """Simple BaseClass with a name."""
    def say_ho(self) -> str:
        return f"I don't have a name, but the type {type(self)}."
    def short_desc(self) -> str:
        return f"SecondBaseClass()"

class DoubleChildClass(BaseClass, SecondBaseClass):
    """Simple child class inheriting from BaseClass and SecondBaseClass."""
    def short_desc(self) -> str:
        return "Child of " + super(DoubleChildClass, self).short_desc()



class A: pass

class B: pass

class C(A, B): pass

class D(A, B): pass

class E(C, D): pass

class F(A, B): pass

class G(A, B): pass

class H(F, G): pass

class I(E, H): pass

