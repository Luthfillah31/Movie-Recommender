# dags/tmdb_to_pinecone_dag.py
from airflow.decorators import task
from plugins.extract import run_extraction_pipeline

@task
def extract_data():
    # This securely executes the entire async pipeline and returns the final dictionary
    return run_extraction_pipeline(pages=20)