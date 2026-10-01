"""
serve.py

Starts the ExamVision web application.

    python serve.py                     http://127.0.0.1:8000 (this computer only), opens the browser
    python serve.py --port 9000
    python serve.py --host 0.0.0.0      reachable from other computers on the network

There is no login yet: anyone who can reach the address can upload
videos and record review decisions. Only use --host 0.0.0.0 on a
network you trust.
"""
from __future__ import annotations

import argparse
import importlib
import sys
import threading
import webbrowser
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
_REQUIRED = {"av": "av", "cv2": "opencv-python", "mediapipe": "mediapipe", "fastapi": "fastapi",
             "uvicorn": "uvicorn[standard]", "multipart": "python-multipart", "numpy": "numpy"}


def check_environment() -> None:
    """Refuse to start in a Python that lacks the project's packages --
    e.g. the system Python instead of the project's venv. Without PyAV the
    app would silently save videos browsers cannot play."""
    missing = []
    for module, package in _REQUIRED.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    h264 = False
    if "av" not in missing:
        import av
        h264 = any(c in av.codecs_available for c in ("libx264", "h264_mf", "h264_qsv", "h264_amf", "h264_nvenc"))
    if missing or not h264:
        venv_python = PROJECT_DIR / "venv" / "Scripts" / "python.exe"
        print("ERROR: ExamVision can't start with this Python:")
        print(f"  {sys.executable}")
        if missing:
            print(f"  Missing packages: {', '.join(missing)}")
        else:
            print("  PyAV has no H.264 encoder, so browsers could not play the videos.")
        if venv_python.exists() and Path(sys.executable).resolve() != venv_python.resolve():
            print("\nStart it with the project's own Python instead:")
            print(f'  & "{venv_python}" serve.py')
        else:
            print("\nInstall the requirements into this Python:")
            print(f'  "{sys.executable}" -m pip install -r requirements.txt')
        sys.exit(1)


def main() -> None:
    check_environment()
    import uvicorn
    from web.app import create_app

    parser = argparse.ArgumentParser(description="ExamVision web application")
    parser.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this computer only)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="don't open a browser window")
    args = parser.parse_args()

    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print("WARNING: ExamVision has no login. Anyone who can reach "
              f"{args.host}:{args.port} can upload videos and record review decisions.")

    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '::') else args.host}:{args.port}/"
    print(f"ExamVision is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(PROJECT_DIR), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
