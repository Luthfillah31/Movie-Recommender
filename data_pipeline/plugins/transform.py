import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
EMBEDDING_MODEL = "perplexity/pplx-embed-v1-0.6b"
MAX_CONCURRENT_EMBEDDINGS = 20


def clean_and_format_movie(raw_movie: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Extract relevant fields from the raw TMDB metadata and construct a text chunk for embedding."""
    try:
        movie_id = str(raw_movie.get("id"))
        title = raw_movie.get("title", "Unknown Title")
        overview = raw_movie.get("overview", "")
        release_date = raw_movie.get("release_date", "")
        year = release_date.split("-")[0] if release_date else "Unknown Year"
        poster_path = raw_movie.get("poster_path") or ""
        
        genres = [g.get("name") for g in raw_movie.get("genres", [])]
        genre_str = ", ".join(genres)
        
        credits = raw_movie.get("credits", {})
        crew = credits.get("crew", [])
        cast = credits.get("cast", [])
        
        directors = [member["name"] for member in crew if member.get("job") == "Director"]
        director_str = ", ".join(directors) if directors else "Unknown Director"
        
        top_cast = [member["name"] for member in cast[:10]]
        cast_str = ", ".join(top_cast) if top_cast else "Unknown Cast"

        text_to_embed = (
            f"Title: {title} ({year}). "
            f"Genres: {genre_str}. "
            f"Director: {director_str}. "
            f"Starring: {cast_str}. "
            f"Synopsis: {overview}"
        )

        metadata = {
            "title": title,
            "year": year,
            "genres": genre_str,
            "director": director_str,
            "cast": cast_str,
            "poster_path": poster_path,
            "text": text_to_embed  
        }

        return {
            "id": movie_id,
            "text_to_embed": text_to_embed,
            "metadata": metadata
        }
    except Exception as e:
        logger.warning(f"Skipping movie due to parsing error: {e}")
        return None


@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(5))
async def _fetch_dense_embedding(session: aiohttp.ClientSession, text: str) -> List[float]:
    """Generate a dense vector embedding using OpenRouter's API."""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": EMBEDDING_MODEL,
        "input": text
    }
    
    async with session.post(url, headers=headers, json=payload) as response:
        response.raise_for_status()
        data = await response.json()
        return data["data"][0]["embedding"]


async def generate_dense_vectors(texts: List[str]) -> List[List[float]]:
    """Batch generate semantic dense vector embeddings concurrently."""
    logger.info(f"Generating semantic (dense) vectors for {len(texts)} movies via OpenRouter...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)
    
    async def bounded_embed(text: str) -> List[float]:
        async with semaphore:
            return await _fetch_dense_embedding(session, text)

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded_embed(text) for text in texts]
        return await asyncio.gather(*tasks)


def generate_sparse_vectors(texts: List[str], is_seed: bool = False) -> List[Dict[str, Any]]:
    """Generate BM25 keyword vectors. Fits model if seeding; otherwise loads model."""
    bm25 = BM25Encoder()
    bm25_path = "bm25_model.json"
    
    if is_seed or not os.path.exists(bm25_path):
        logger.info("Fitting new BM25 model on corpus and saving state...")
        bm25.fit(texts)
        bm25.dump(bm25_path)
    else:
        logger.info("Loading pre-fit BM25 model for incremental update...")
        bm25.load(bm25_path)
    
    return bm25.encode_documents(texts)


def transform_data(raw_movies: List[Dict[str, Any]], is_seed: bool = False) -> List[Dict[str, Any]]:
    """Process raw TMDB movies and produce Pinecone-compatible Hybrid Vector payloads."""
    if not raw_movies:
        logger.warning("No raw movies provided to transform.")
        return []

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing from environment.")

    cleaned_data = []
    for movie in raw_movies:
        formatted = clean_and_format_movie(movie)
        if formatted:
            cleaned_data.append(formatted)
            
    texts = [item["text_to_embed"] for item in cleaned_data]

    sparse_vectors = generate_sparse_vectors(texts, is_seed=is_seed)
    dense_vectors = asyncio.run(generate_dense_vectors(texts))

    pinecone_payload = []
    for i in range(len(cleaned_data)):
        record = {
            "id": cleaned_data[i]["id"],
            "values": dense_vectors[i],
            "sparse_values": sparse_vectors[i],
            "metadata": cleaned_data[i]["metadata"]
        }
        pinecone_payload.append(record)

    logger.info(f"Successfully transformed {len(pinecone_payload)} movies into Hybrid Vectors.")
    return pinecone_payload


if __name__ == "__main__":
    sample_raw_movies = [
        {
            "id": 155, 
            "title": "The Dark Knight", 
            "overview": "Batman raises the stakes in his war on crime.", 
            "release_date": "2008-07-16",
            "genres": [{"name": "Action"}, {"name": "Crime"}],
            "credits": {
                "crew": [{"name": "Christopher Nolan", "job": "Director"}],
                "cast": [{"name": "Christian Bale"}, {"name": "Heath Ledger"}]
            }
        }
    ]
    
    print("Testing Transformation Pipeline (Dense + Sparse)...")
    vectors = transform_data(sample_raw_movies)
    if vectors:
        print(f"Success! ID: {vectors[0]['id']}")
        print(f"Dense Vector Length: {len(vectors[0]['values'])}")
        print(f"Sparse Indices Sample: {vectors[0]['sparse_values']['indices'][:3]}")