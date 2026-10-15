"""Run the development mock: python -m hala.serving."""
import os
import sys

from hala.inference import MockBackend, RealBackend
from hala.serving import Config, Service, create_server


def main():
    try:
        mode = os.environ.get("HALA_BACKEND", "mock")
        if mode not in ("mock", "real"):
            raise ValueError("Invalid backend.")
        config = Config(token=os.environ.get("HALA_BEARER_TOKEN", ""),
                        tenant_id=os.environ.get("HALA_TENANT_ID", ""))
        port = int(os.environ.get("HALA_PORT", "8765"))
        server = create_server(Service(config, MockBackend() if mode == "mock" else RealBackend()), port=port)
    except (ValueError, OSError):
        print("Hala startup failed: validate environment and loopback port.", file=sys.stderr)
        return 2
    print("Hala development service listening on IPv4 loopback; backend=" + mode)
    try:
        server.serve_forever(poll_interval=.1)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
