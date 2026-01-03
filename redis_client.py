import redis
import json
import hashlib
from functools import wraps # make the wrapper preserve the original function's metadata
import logging
import os

logging.basicConfig(level=logging.INFO)

# Support both local development and Docker environments
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
REDIS_DB = int(os.getenv('REDIS_DB', 0))

redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB, decode_responses=True)

def cache_result(ttl=86400):
    def decorator(fn):
        @wraps(fn) 
        def wrapper(*args, **kwargs):
            # Create cache key based on function name + args + kwargs
            key_parts = [fn.__name__] + [str(a) for a in args] + [f"{k}:{v}" for k,v in kwargs.items()]
            cache_key = hashlib.md5(" ".join(key_parts).encode()).hexdigest()  # MD5 accepts byte sequences, encode strings into bytes using the encode() method
            
            # Try cache
            cached = redis_client.get(cache_key)
            if cached:
                logging.info(f"Redis CACHE HIT: {cache_key[:16]}... for {fn.__name__}")
                return json.loads(cached)
            
            # If no cache hit compute and cache result
            logging.info(f"Redis CACHE MISS: {cache_key[:16]}... for {fn.__name__}")
            result = fn(*args, **kwargs)
            redis_client.setex(cache_key, ttl, json.dumps(result))
            return result
        return wrapper
    return decorator

            