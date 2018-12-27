# stdlib
import pickle
from queue import Full, Queue
import socket
import sys
from threading import Thread
import time


def _serve_forever(port, queue):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(('', port))
        while True:
            data, return_address = sock.recvfrom(2**3)
            response = pickle.dumps(queue.get())
            sock.sendto(response, return_address)


class Server:
    def __init__(self, port):
        self.queue = Queue(maxsize=20)
        Thread(target=_serve_forever, args=[port, self.queue]).start()

    def serve(self, values):
        try:
            self.queue.put(values)
        except Full:
            time.sleep(1e-100)  # Why does this work? Don't know.


if __name__ == '__main__':
    port = int(sys.argv[1])
    server = Server(port)
    while True:
        server.serve(input('Enter message:'))
