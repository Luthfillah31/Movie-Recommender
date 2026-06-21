import logging
import sys
from pathlib import Path

# Add parent directory to path to enable package imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from plugins.extract import run_daily_update
from plugins.transform import transform_data
from plugins.load import load_to_pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting daily incremental update...")
    try:
        raw_movies = run_daily_update()
        if not raw_movies:
            logger.info("No upstream changes found. Exiting.")
            sys.exit(0)
            
        vectors = transform_data(raw_movies)
        if not vectors:
            logger.warning("Transformation yielded empty vector list. Aborting.")
            sys.exit(0)
            
        load_to_pinecone(vectors)
        logger.info("Daily update successfully completed.")
    except Exception as e:
        logger.critical(f"Daily update failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()