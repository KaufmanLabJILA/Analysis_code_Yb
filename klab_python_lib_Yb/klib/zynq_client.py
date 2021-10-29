import socket
import sys

class zynq_tcp_client:
    def __init__(self):
        self.server_address = ("10.10.0.2", 8080)
        # Create a TCP/IP socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connectByteLen = 64
        self.dioByteLen = 28
        self.dacByteLen = 42
        self.ddsByteLen = 46

    def connect(self):
        self.sock.connect(self.server_address)

    def sendMessage(self, message, length):
        messagePad = message.ljust(length, b'\0')
        self.sock.sendall(messagePad)

    def triggerGigamoogWithTweezersOn(self):
        try:
            self.sendMessage(b'DIOseq_2', self.connectByteLen)
            self.sendMessage(b't000186A0_b0000000000300180', self.dioByteLen)
            self.sendMessage(b't000196A0_b0000000000300100', self.dioByteLen)
            self.sendMessage(b'end_0', self.connectByteLen)
            self.sendMessage(b'trigger', self.connectByteLen)
        except:
            print('write failed')

    def turnOffTTLs(self):
        try:
            self.sendMessage(b'DIOseq_1', self.connectByteLen)
            self.sendMessage(b't000186A0_b0000000000000000', self.dioByteLen)
            self.sendMessage(b'end_0', self.connectByteLen)
            self.sendMessage(b'trigger', self.connectByteLen)
        except:
            print('write failed')

    def triggerBlueMot(self):
        try:
            self.sendMessage(b'DIOseq_1', self.connectByteLen)
            self.sendMessage(b't000186A0_b000000000007000C', self.dioByteLen)
            self.sendMessage(b'end_0', self.connectByteLen)
            self.sendMessage(b'trigger', self.connectByteLen)
        except:
            print('write failed')

    def disconnect(self):

        # print('closing socket')
        self.sock.close()

if __name__ == "__main__":
    client = zynq_tcp_client()
    client.connect()
    client.triggerGigamoogWithTweezersOn()
    client.disconnect()
