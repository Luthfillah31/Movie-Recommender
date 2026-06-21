import logging
import sys
from pathlib import Path

# Add parent directory to path to enable package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from plugins.extract import run_historical_seed
from plugins.transform import transform_data
from plugins.load import load_to_pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting historical database seeding...")
    try:
        raw_movies = run_historical_seed(pages=500)
        if not raw_movies:
            logger.warning("Extraction returned no movies. Aborting seed operation.")
            sys.exit(0)
            
        vectors = transform_data(raw_movies, is_seed=True)
        if not vectors:
            logger.warning("Transformation yielded empty vector payload. Aborting.")
            sys.exit(0)
            
        if load_to_pinecone(vectors):
            logger.info("Historical database seeding completed successfully.")
        else:
            logger.critical("Seed operation failed during Pinecone load stage.")
            sys.exit(1)
    except Exception as e:
        logger.critical(f"Seed operation failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()