import argparse
import sys

class RecordAction(argparse.Action):
    """
    A custom action to handle the --record flag, which can accept
    a string (prefix) and an integer (interval) in any order.
    
    - If --record is not present: value is None.
    - If --record is present alone: uses defaults ('test', 1).
    - If --record is present with one arg: it determines if it's an int or str.
    - If --record is present with two args: it parses both.
    """
    def __call__(self, parser, namespace, values, option_string=None):
        # default values
        prefix = 'eval'
        interval = 1

        # goes through vars on cmd line
        for value in values:
            # try to update interval value
            try:
                interval = int(value)
            # if it fails then it is the prefix value
            except ValueError:
                prefix = value
        setattr(namespace, self.dest, {'prefix': prefix, 'interval': interval})
