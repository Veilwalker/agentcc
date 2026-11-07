"""Select the numeric backend once, without changing scheduling decisions.

The submission image requires C++; development may use the Python reference.
This module does not compile code, read credentials, or make model calls.
"""
import importlib
import os


def load_native(mode):
    if mode not in {'auto', 'python', 'cpp'}:
        raise ValueError('AGENTCC_CAPACITY_BACKEND must be auto, python, or cpp')
    if mode == 'python':
        return None
    try:
        module = importlib.import_module('._capacity_native', __package__)
    except ImportError as error:
        if mode == 'cpp':
            raise RuntimeError(
                'C++ capacity extension unavailable. Build with native/build.py '
                'using the same Python interpreter as the service.'
            ) from error
        return None
    if module.API_VERSION != 1:
        raise RuntimeError('unsupported C++ capacity extension API')
    return module


native = load_native(os.environ.get('AGENTCC_CAPACITY_BACKEND', 'auto'))
BACKEND = 'cpp' if native is not None else 'python'


class NativeCapacityAssignment:
    """Per-round native loads; input paths stay owned by Python."""
    def __init__(self, costs):
        self._state = native.make_assignment(len(costs.indices))

    @property
    def time_us(self):
        return native.assignment_time(self._state)

    def choose(self, options, count, packet_bytes=None):
        return native.choose(self._state, tuple(options), count,
                             count if packet_bytes is None else packet_bytes)
