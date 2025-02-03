import argparse
import os

def check_greater_than_int(value, number):
    """Checks to see if the given value is an integer greater than number.

    Args:
        value (string): The value to parse and compare.
        number (integer): The number for comparison.

    Raises:
        argparse.ArgumentTypeError: The parsed value <= number.

    Returns:
        integer: The parsed integer.
    """
    parsed_value = int(value)
    if parsed_value <= number:
        raise argparse.ArgumentTypeError("{value} <= {number}")
    return parsed_value

def check_greater_than_float(value, number):
    """Checks to see if the given value is a float greater than number.

    Args:
        value (string): The value to parse and compare.
        number (float): The number for comparison.

    Raises:
        argparse.ArgumentTypeError: The parsed value <= number.

    Returns:
        float: The parsed float.
    """
    parsed_value = float(value)
    if parsed_value <= number:
        raise argparse.ArgumentTypeError("{value} <= {number}")
    return parsed_value

def is_valid_file(value):
    """Checks to see if the given value is a valid file.

    Args:
        value (string): The string to do the file check on.

    Raises:
        FileNotFoundError: value is not a file.

    Returns:
        string: The file string.
    """
    if not value:
        return value
    if os.path.exists(value):
        return value
    raise FileNotFoundError(value)