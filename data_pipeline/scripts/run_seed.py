import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from plugins.extract import run_historical_seed
from plugins.transform import transform_data
from plugins.load import load_to_pinecone

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    logger.info("Initiating historical database seed.")
    
    try:
        raw_movies = run_historical_seed(pages=500)
        if not raw_movies:
            logger.warning("Extraction yielded no data. Aborting seed operation.")
            sys.exit(0)
            

        vectors = transform_data(raw_movies, is_seed=True)
        if not vectors:
            logger.warning("Transformation yielded empty vector payload. Aborting.")
            sys.exit(0)
            
        success = load_to_pinecone(vectors)
        if success:
            logger.info("Historical seed operation completed successfully.")
        else:
            logger.error("Seed operation aborted due to Pinecone load failure.")
            sys.exit(1)
        
    except Exception as e:
        logger.error(f"Seed operation failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()