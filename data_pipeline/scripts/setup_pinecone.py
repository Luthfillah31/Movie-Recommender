import os
import logging
import sys

from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "movie-recommender")
VECTOR_DIMENSION = 1024


def initialize_infrastructure():
    """Idempotently configure and provision the Pinecone vector index."""
    if not PINECONE_API_KEY:
        logger.critical("PINECONE_API_KEY is not configured.")
        sys.exit(1)
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    active_indexes = pc.list_indexes().names()
    
    if PINECONE_INDEX_NAME in active_indexes:
        logger.info(f"Index '{PINECONE_INDEX_NAME}' already exists.")
        return

    logger.info(f"Provisioning index '{PINECONE_INDEX_NAME}' (1024-dim, dotproduct, AWS us-east-1)...")
    try:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric="dotproduct",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        logger.info("Pinecone index provisioned successfully.")
    except Exception as e:
        logger.critical(f"Failed to provision Pinecone index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    initialize_infrastructure()