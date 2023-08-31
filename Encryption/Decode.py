import sys
import random

def decode():
    key = input("Enter decrypt key: ")
    filename = input("Filename: ")
    cipher = open(filename, 'rb').read()
    
    repetitions = (len(cipher)-1)//len(key)+1
    
    key = (key*repetitions)[:len(cipher)].encode('utf-8')
    decoded = bytes([i1 ^ i2 for (i1, i2) in zip(cipher, key)])
    return decoded.decode('utf-8')

if __name__ == '__main__':
    print(decode())