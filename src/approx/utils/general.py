import os
import warnings
import os


def check_file(file, ext=None):
    if file is None:
        return False
    if not os.path.exists(file):
        warnings.warn(f"{file} is not exists")
        return False
    if not os.path.isfile(file):
        warnings.warn(f"{file} must be a file")
        return False
    if ext:
        if not os.path.splitext(file)[1] in ext:
            # warnings.warn(f"the type of {file} must be in {ext}")
            return False
    return True


def parse_filename(path):
    paths = os.path.split(path)
    return os.path.splitext(paths[-1])[0]


def to_2tuple(x) -> tuple:
    if isinstance(x, tuple):
        return x[:2]
    return x, x


def is_method_overridden(method, base_class, derived_class):
    """Check if a method of base class is overridden in derived class.

    Args:
        method (str): the method name to check.
        base_class (type): the class of the base class.
        derived_class (type | Any): the class or instance of the derived class.
    """
    assert isinstance(base_class, type), \
        "base_class doesn't accept instance, Please pass class instead."

    if not isinstance(derived_class, type):
        derived_class = derived_class.__class__

    base_method = getattr(base_class, method)
    derived_method = getattr(derived_class, method)
    return derived_method != base_method
