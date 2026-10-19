"""Start the API: `uv run python -m brainct.serving [--host 127.0.0.1] [--port 8000]`."""

from __future__ import annotations

import argparse

import uvicorn


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    uvicorn.run("brainct.serving.app:create_app", factory=True, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
