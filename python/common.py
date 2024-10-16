from collections import namedtuple

class Result(namedtuple('Result', ['value', 'gradient'])):
    """The result of a C^1 function. Contains function value and gradient if desired.
    The tuple is (value, gradient).
    """
    pass

class OracleResult(namedtuple('OracleResult', ['answer', 'value', 'gradient'])):
    """The result of a separation oracle.
    The tuple is (answer, value, gradient).
    """
    pass

class EllipsoidResult(namedtuple('EllipsoidResult', ['answer', 'value', 'point', 'gradient'])):
    """The result of the ellipsoid algorithm.
    the tuple is (answer, value, point, gradient).
    """
    pass
