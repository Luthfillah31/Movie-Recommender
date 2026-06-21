import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_API_TOKEN = os.getenv("TMDB_API_TOKEN")
MAX_CONCURRENT_REQUESTS = 40


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=10),
    stop=stop_after_attempt(5),
    reraise=True
)
async def _fetch_json(session: aiohttp.ClientSession, url: str, params: Optional[Dict] = None) -> Dict[str, Any]:
    """Helper to perform HTTP GET request with exponential backoff retry mechanism."""
    async with session.get(url, params=params) as response:
        if response.status == 429:
            logger.warning(f"Rate limited by TMDB on {url}. Retrying...")
        response.raise_for_status()
        return await response.json()


async def fetch_popular_movie_ids(session: aiohttp.ClientSession, pages: int = 20) -> List[int]:
    """Retrieve popular movie IDs from TMDB for historical seeding."""
    logger.info(f"Extracting movie IDs from top {pages} popular pages...")
    url = f"{TMDB_BASE_URL}/movie/popular"
    
    tasks = [
        _fetch_json(session, url, params={"language": "en-US", "page": str(page)})
        for page in range(1, pages + 1)
    ]
    
    pages_data = await asyncio.gather(*tasks, return_exceptions=True)
    
    movie_ids = []
    for data in pages_data:
        if isinstance(data, BaseException):
            logger.error(f"Failed to fetch a page of IDs: {data}")
            continue
            
        results = data.get("results", [])
        movie_ids.extend([movie["id"] for movie in results])
        
    unique_ids = list(set(movie_ids))
    logger.info(f"Extracted {len(unique_ids)} unique popular movie IDs.")
    return unique_ids


async def fetch_changed_movie_ids(session: aiohttp.ClientSession) -> List[int]:
    """Retrieve IDs of movies that changed in the last 24 hours."""
    logger.info("Fetching movies updated in the last 24 hours...")
    url = f"{TMDB_BASE_URL}/movie/changes"
    
    try:
        data = await _fetch_json(session, url)
        results = data.get("results", [])
        changed_ids = [movie["id"] for movie in results]
        
        logger.info(f"Found {len(changed_ids)} movies that changed today.")
        return changed_ids
    except Exception as e:
        logger.error(f"Failed to fetch changed movie IDs: {e}")
        return []


async def fetch_movie_details(session: aiohttp.ClientSession, movie_ids: List[int]) -> List[Dict[str, Any]]:
    """Fetch detailed metadata (including credits and keywords) for a list of movie IDs."""
    logger.info(f"Fetching detailed metadata for {len(movie_ids)} movies...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async def bounded_fetch(movie_id: int) -> Optional[Dict[str, Any]]:
        async with semaphore:
            url = f"{TMDB_BASE_URL}/movie/{movie_id}"
            params = {"append_to_response": "credits,keywords"}
            try:
                return await _fetch_json(session, url, params=params)
            except Exception as e:
                logger.error(f"Failed to extract details for movie {movie_id}: {e}")
                return None

    tasks = [bounded_fetch(movie_id) for movie_id in movie_ids]
    results = await asyncio.gather(*tasks)
    
    valid_results = [res for res in results if res is not None]
    logger.info(f"Successfully extracted detailed metadata for {len(valid_results)} movies.")
    return valid_results


def _get_session_headers() -> Dict[str, str]:
    if not TMDB_API_TOKEN:
        raise ValueError("TMDB_API_TOKEN environment variable is missing.")
    return {
        "Authorization": f"Bearer {TMDB_API_TOKEN}",
        "accept": "application/json"
    }


def run_historical_seed(pages: int = 20) -> List[Dict[str, Any]]:
    """Synchronous entry point to extract historical data for database seeding."""
    async def _execute():
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(headers=_get_session_headers(), timeout=timeout) as session:
            ids = await fetch_popular_movie_ids(session, pages=pages)
            if not ids:
                return []
            return await fetch_movie_details(session, ids)

    return asyncio.run(_execute())


def run_daily_update() -> List[Dict[str, Any]]:
    """Synchronous entry point to run incremental nightly updates."""
    async def _execute():
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(headers=_get_session_headers(), timeout=timeout) as session:
            changed_ids = await fetch_changed_movie_ids(session)
            if not changed_ids:
                return []
            return await fetch_movie_details(session, changed_ids)

    return asyncio.run(_execute())


if __name__ == "__main__":
    print("Testing Daily Delta Extraction...")
    daily_data = run_daily_update()
    if daily_data:
        print(f"Successfully extracted {len(daily_data)} changed movies today.")
        print(f"Sample: {daily_data[0].get('title')}")