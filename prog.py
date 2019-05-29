import argparse
import math

def perfect_square(string):
     value = int(string)
     sqrt = math.sqrt(value)
     if sqrt != int(sqrt):
         msg = "%r is not a perfect square" % string
         raise argparse.ArgumentTypeError(msg)
     return math.sqrt(value)

parser = argparse.ArgumentParser(prog='PROG')
parser.add_argument('foo', type=perfect_square)
args = parser.parse_args()
print(args.foo)