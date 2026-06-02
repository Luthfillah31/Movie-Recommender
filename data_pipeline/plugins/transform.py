import os
import logging
import asyncio
from typing import List, Dict, Any, Tuple

import aiohttp
from tenacity import retry, wait_exponential, stop_after_attempt
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
# Nomic-embed-text is highly optimized for RAG, cheap, and available on OpenRouter
EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"
MAX_CONCURRENT_EMBEDDINGS = 20


def clean_and_format_movie(raw_movie: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Extracts relevant fields from the TMDB JSON and constructs a rich 
    text chunk for embedding.
    """
    try:
        movie_id = str(raw_movie.get("id"))
        title = raw_movie.get("title", "Unknown Title")
        overview = raw_movie.get("overview", "")
        release_date = raw_movie.get("release_date", "")
        year = release_date.split("-")[0] if release_date else "Unknown Year"
        poster_path = raw_movie.get("poster_path", "")
        
        # Extract Genres
        genres = [g.get("name") for g in raw_movie.get("genres", [])]
        genre_str = ", ".join(genres)
        
        # Extract Director and Top 3 Cast members from the appended credits
        credits = raw_movie.get("credits", {})
        crew = credits.get("crew", [])
        cast = credits.get("cast", [])
        
        directors = [member["name"] for member in crew if member.get("job") == "Director"]
        director_str = ", ".join(directors) if directors else "Unknown Director"
        
        top_cast = [member["name"] for member in cast[:3]]
        cast_str = ", ".join(top_cast) if top_cast else "Unknown Cast"

        # The "Rich Chunk" - This is what the AI actually reads and embeds
        text_to_embed = (
            f"Title: {title} ({year}). "
            f"Genres: {genre_str}. "
            f"Director: {director_str}. "
            f"Starring: {cast_str}. "
            f"Synopsis: {overview}"
        )

        # We keep the metadata separated for Streamlit to use later
        metadata = {
            "title": title,
            "year": year,
            "genres": genre_str,
            "director": director_str,
            "cast": cast_str,
            "poster_path": poster_path,
            "text": text_to_embed  # Crucial for the LLM to read during generation
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
    """Hits OpenRouter's embedding API asynchronously."""
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
    """Manages the async batching for OpenRouter semantic embeddings."""
    logger.info(f"Generating semantic (dense) vectors for {len(texts)} movies via OpenRouter...")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBEDDINGS)
    
    async def bounded_embed(text: str) -> List[float]:
        async with semaphore:
            return await _fetch_dense_embedding(session, text)

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = [bounded_embed(text) for text in texts]
        return await asyncio.gather(*tasks)


def generate_sparse_vectors(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Generates BM25 keyword vectors for exact-match searching (Sparse).
    """
    logger.info("Generating keyword (sparse) vectors using BM25...")
    bm25 = BM25Encoder()
    
    # In a massive enterprise system, you would load a pre-fit bm25.json file here.
    # For a daily delta pipeline, fitting it on the daily batch + historical data works perfectly.
    bm25.fit(texts)
    
    # encode_documents returns a dict: {"indices": [1, 5, 20], "values": [0.5, 1.2, 0.8]}
    sparse_vectors = bm25.encode_documents(texts)
    return sparse_vectors


def transform_data(raw_movies: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    The main orchestrator. Takes raw TMDB JSON and outputs Pinecone-ready Hybrid Vectors.
    """
    if not raw_movies:
        logger.warning("No raw movies provided to transform.")
        return []

    if not OPENROUTER_API_KEY:
        raise ValueError("OPENROUTER_API_KEY is missing from environment.")

    # 1. Clean Data & Format Text
    cleaned_data = []
    for movie in raw_movies:
        formatted = clean_and_format_movie(movie)
        if formatted:
            cleaned_data.append(formatted)
            
    texts = [item["text_to_embed"] for item in cleaned_data]

    # 2. Generate Sparse (Keyword) Vectors synchronously
    sparse_vectors = generate_sparse_vectors(texts)

    # 3. Generate Dense (Semantic) Vectors asynchronously
    dense_vectors = asyncio.run(generate_dense_vectors(texts))

    # 4. Assemble the final payload for Pinecone Upsert
    pinecone_payload = []
    for i in range(len(cleaned_data)):
        record = {
            "id": cleaned_data[i]["id"],
            "values": dense_vectors[i],               # Dense semantic embedding
            "sparse_values": sparse_vectors[i],       # Sparse keyword embedding
            "metadata": cleaned_data[i]["metadata"]   # Metadata for Streamlit
        }
        pinecone_payload.append(record)

    logger.info(f"Successfully transformed {len(pinecone_payload)} movies into Hybrid Vectors.")
    return pinecone_payload


if __name__ == "__main__":
    # Local Testing Block
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