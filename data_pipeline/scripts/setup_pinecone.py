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

# perplexity/pplx-embed-v1-0.6b vector size
VECTOR_DIMENSION = 1024 


def initialize_infrastructure():
    """Idempotent function to provision the Pinecone vector index."""
    if not PINECONE_API_KEY:
        logger.error("PINECONE_API_KEY is not configured.")
        sys.exit(1)
        
    pc = Pinecone(api_key=PINECONE_API_KEY)
    active_indexes = pc.list_indexes().names()
    
    if PINECONE_INDEX_NAME in active_indexes:
        logger.info(f"Target index '{PINECONE_INDEX_NAME}' already exists. No action required.")
        return

    logger.info(f"Provisioning new Serverless index: '{PINECONE_INDEX_NAME}'...")
    
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
        logger.info(f"Infrastructure provisioned successfully: {PINECONE_INDEX_NAME} (1024-dim, dotproduct).")
        
    except Exception as e:
        logger.error(f"Failed to provision Pinecone index: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    initialize_infrastructure()