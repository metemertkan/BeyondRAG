#!/usr/bin/env python3
"""Lightweight permanent bridge forwarding Docker bridge interfaces to host Ollama."""

import socket
import threading
import time

def handle(src, dst):
    try:
        while True:
            data = src.recv(8192)
            if not data:
                break
            dst.sendall(data)
    except Exception:
        pass
    finally:
        for s in (src, dst):
            try:
                s.close()
            except Exception:
                pass

def forward(client):
    try:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.connect(('127.0.0.1', 11434))
        threading.Thread(target=handle, args=(client, server), daemon=True).start()
        threading.Thread(target=handle, args=(server, client), daemon=True).start()
    except Exception:
        client.close()

def listen_on(ip):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((ip, 11434))
        s.listen(32)
        while True:
            client, _ = s.accept()
            threading.Thread(target=forward, args=(client,), daemon=True).start()
    except Exception as e:
        pass

for ip in ('172.17.0.1', '172.24.0.1'):
    threading.Thread(target=listen_on, args=(ip,), daemon=True).start()

while True:
    time.sleep(3600)
