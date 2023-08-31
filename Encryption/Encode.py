import sys
import socket

def encode():
    hostname = socket.gethostname()
    # temp = socket.gethostbyname(socket.getfqdn())
    ipAddr = socket.gethostbyname(hostname)
    print("Your host name is: " + hostname)
    print("Your IP is: " + ipAddr)
    # print(temp)
    # toEncode = input("Enter String: ")
    key = input("Enter encrypt key: ")
    filename = input("File name: ")
    
    # how many repetitions the key needs to be repeated to match the length of the encode
    repetitions = (len(ipAddr)-1)//len(key)+1
    
    utfEncode = ipAddr.encode('utf-8')
    
    # match the length of the key to the length of the encode
    key = (key * repetitions)[:len(ipAddr)].encode('utf-8')

    cipher = bytes([i1 ^ i2 for (i1, i2) in zip(utfEncode, key)])
    
    open(filename, 'wb').write(cipher)
    
if __name__ == '__main__':
    encode()