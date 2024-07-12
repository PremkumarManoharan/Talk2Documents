from config import PROJECT_ID, DATASET_ID, TABLE_ID, VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_DIMENSIONS, BUCKET_URI
from helpers.query_processing import get_answer
from helpers.bigquery_operations import load_embeddings_from_bigquery
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID)

# Load the embeddings from BigQuery
loaded_embeddings_df = load_embeddings_from_bigquery(PROJECT_ID, DATASET_ID, TABLE_ID)

# Initialize the index endpoint
index_endpoint = aiplatform.MatchingEngineIndexEndpoint(
    index_endpoint_name=f"projects/{PROJECT_ID}/locations/{LOCATION}/indexEndpoints/{VECTOR_SEARCH_INDEX_NAME}"
)

# Initialize text embedding model
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

def main():
    query = "What are the steps of Transformer Manufacturing Flow?"
    result, page_source = get_answer(query, loaded_embeddings_df, index_endpoint, text_embedding_model, VECTOR_SEARCH_INDEX_NAME)
    print(result)

if __name__ == "__main__":
    main()
