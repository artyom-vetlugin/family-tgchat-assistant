"""Entrypoint: python -m family_assistant"""

import asyncio
import logging

from .bot import BotApp
from .config import get_settings


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = BotApp(get_settings())
    asyncio.run(app.run())


if __name__ == "__main__":
    main()
