"""Open the existing Streamlit UI in a native desktop window."""

from __future__ import annotations

import atexit
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8510
URL = f"http://127.0.0.1:{PORT}"


def _port_open() -> bool:
    sock = socket.socket()
    sock.settimeout(0.3)
    try:
        sock.connect(("127.0.0.1", PORT))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def _ready() -> bool:
    try:
        with urllib.request.urlopen(URL, timeout=0.8) as response:
            return response.status < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _start_server() -> subprocess.Popen[bytes] | None:
    if _ready():
        return None
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(ROOT / "app.py"),
            "--server.port",
            str(PORT),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--browser.gatherUsageStats",
            "false",
            "--client.toolbarMode",
            "viewer",
        ],
        cwd=str(ROOT),
    )
    for _ in range(100):
        if _ready():
            return proc
        if proc.poll() is not None:
            raise SystemExit("앱 서버를 시작하지 못했습니다.")
        time.sleep(0.2)
    raise SystemExit("앱이 준비되는 데 시간이 너무 오래 걸렸습니다.")


def main() -> None:
    proc = _start_server()
    if proc is not None:
        atexit.register(proc.terminate)

    try:
        import webview
    except ImportError:
        raise SystemExit("pywebview가 필요합니다. .venv/bin/pip install pywebview")

    webview.create_window(
        "음성 분석",
        URL,
        width=1280,
        height=860,
        min_size=(960, 700),
    )
    webview.start(private_mode=False)


if __name__ == "__main__":
    main()
