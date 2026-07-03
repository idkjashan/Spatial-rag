import asyncio
from core.stores.postgres import PostgresClient
from core.config import EngineConfig
import os

async def main():
    config = EngineConfig()
    pg = PostgresClient(config.db.postgres_dsn)
    await pg.connect()
    await pg.init_schema()
    await pg.close()
    print("Schema initialized.")

if __name__ == "__main__":
    asyncio.run(main())