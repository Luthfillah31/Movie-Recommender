import os
import logging
from typing import List, Dict, Any

from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "movie-recommender")
UPSERT_BATCH_SIZE = 100


def _chunk_data(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """Yield successive n-sized chunks from data."""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]


def _get_pinecone_index():
    """Initialize Pinecone client and validate index existence."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not configured.")
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        raise ValueError(f"Target index '{PINECONE_INDEX_NAME}' does not exist.")
        
    return pc.Index(PINECONE_INDEX_NAME)


def load_to_pinecone(vectors: List[Dict[str, Any]]) -> bool:
    """
    Upserts hybrid vectors into Pinecone using optimal batch sizing.
    """
    if not vectors:
        logger.warning("Empty vector list provided. Aborting upsert operation.")
        return False

    try:
        index = _get_pinecone_index()
        batches = _chunk_data(vectors, UPSERT_BATCH_SIZE)
        
        logger.info(f"Initiating upsert of {len(vectors)} vectors across {len(batches)} batches.")

        for i, batch in enumerate(batches, start=1):
            index.upsert(vectors=batch)
            logger.debug(f"Successfully processed batch {i}/{len(batches)}.")

        logger.info("Pinecone vector upsert completed successfully.")
        return True

    except Exception as e:
        logger.error(f"Pinecone upsert failed: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    logger.info("Running Pinecone load module integration test.")
    
    test_vectors = [
        {
            "id": "test-1",
            "values": [0.01] * 768,
            "sparse_values": {"indices": [1, 2], "values": [0.5, 0.8]},
            "metadata": {
                "title": "Pipeline Integration Test",
                "year": "2026",
                "text": "Automated test payload for hybrid vector upsert."
            }
        }
    ]
    
    # load_to_pinecone(test_vectors)