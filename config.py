import os

PROJECT_ID = "genai-projects-429119"  # Replace with your project ID
LOCATION = "us-east4"
VECTOR_SEARCH_INDEX_NAME = f"{PROJECT_ID}-vector-search-index-ht"
VECTOR_SEARCH_EMBEDDING_DIR = f"{PROJECT_ID}-vector-search-bucket-ht"
VECTOR_SEARCH_DIMENSIONS = 768
BUCKET_NAME = f"{PROJECT_ID}-vector-search-bucket-ht"
BUCKET_URI = f"gs://{BUCKET_NAME}"
DATASET_ID = "dataset"  # Replace with your BigQuery dataset ID
TABLE_ID = "embeddings"  # Replace with your BigQuery table name

# Ensure data directories exist
if not os.path.exists("./data/Images"):
    os.makedirs("./data/Images")
