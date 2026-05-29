import asyncio
from core.db import QdrantClientWrapper
from core.config import settings
import redis.asyncio as redis

async def main():
    qdrant_wrapper = QdrantClientWrapper()
    redis_client = redis.Redis.from_url(settings.REDIS_URL, decode_responses=True)
    
    print("Clearing old active and explore pools...")
    await redis_client.delete("gallery:pool:active")
    await redis_client.delete("gallery:pool:explore")
    
    offset = None
    total_added = 0
    while True:
        points, offset = await qdrant_wrapper.scroll(limit=100, offset=offset)
        if not points:
            break
            
        for p in points:
            await redis_client.zadd("gallery:pool:active", {str(p.id): 0})
            await redis_client.zadd("gallery:pool:explore", {str(p.id): 0})
            total_added += 1
            
        print(f"Added {len(points)} points. Total: {total_added}")
        
        if not offset:
            break
            
    print(f"Initialization complete. Total images in active pool: {total_added}")
    
    await redis_client.close()
    await qdrant_wrapper.client.close()

if __name__ == "__main__":
    asyncio.run(main())
