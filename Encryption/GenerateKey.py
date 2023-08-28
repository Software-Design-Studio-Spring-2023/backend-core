import sys
import random

def generateKey():
    key = '{:09}'.format(random.randrange(1, 10**9))
    # print(key)
    return key

if __name__ == '__main__':
    print(generateKey())