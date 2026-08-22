import socket
import threading
import time

from server import run_server


def reserve_local_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def connect_with_retry(port, timeout=3.0):
    deadline = time.monotonic() + timeout
    while True:
        try:
            return socket.create_connection(("127.0.0.1", port), timeout=0.5)
        except ConnectionRefusedError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.02)


def test_server_accepts_a_new_client_after_disconnect():
    port = reserve_local_port()
    stop_event = threading.Event()
    server_thread = threading.Thread(
        target=run_server,
        kwargs={
            "host": "127.0.0.1",
            "port": port,
            "stop_event": stop_event,
        },
        daemon=True,
    )
    server_thread.start()

    first_client = connect_with_retry(port)
    first_client.close()

    second_client = connect_with_retry(port)
    stop_event.set()
    second_client.close()

    server_thread.join(timeout=3.0)
    assert not server_thread.is_alive()
