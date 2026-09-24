from __future__ import annotations

import uvicorn

HOST = "127.0.0.1"
PORT = 8787
APP_FACTORY = "muse_glimmer_token_service.app:create_app"


def main() -> None:
    uvicorn.run(
        APP_FACTORY,
        factory=True,
        host=HOST,
        port=PORT,
        proxy_headers=False,
        server_header=False,
    )


if __name__ == "__main__":
    main()
