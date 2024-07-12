import os
import json
from config import PROJECT_ID, LOCATION, VECTOR_SEARCH_INDEX_NAME, VECTOR_SEARCH_DIMENSIONS, BUCKET_URI, DATASET_ID, TABLE_ID
from helpers.pdf_processing import split_pdf_to_images
from helpers.embedding_processing import extract_text_and_generate_embeddings
from helpers.bigquery_operations import save_embeddings_to_bigquery
from google.cloud import aiplatform

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=LOCATION)

def main():
    pdf_filename = "04a02.pdf"  # Replace with your PDF filename
    image_paths = split_pdf_to_images(pdf_filename)
    embeddings_df = extract_text_and_generate_embeddings(image_paths)

    save_embeddings_to_bigquery(embeddings_df, PROJECT_ID, DATASET_ID, TABLE_ID)
    print("PDF processed and embeddings saved to BigQuery.")

    # Create and deploy the index
    index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
        display_name=f"{VECTOR_SEARCH_INDEX_NAME}",
        contents_delta_uri=BUCKET_URI,
        dimensions=VECTOR_SEARCH_DIMENSIONS,
        approximate_neighbors_count=20,
        distance_measure_type="DOT_PRODUCT_DISTANCE",
    )
    index_id = index.resource_name.split("/")[-1]
    print(f"Index created with ID: {index_id}")
    
    # Create the index endpoint
    index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
        display_name=f"{VECTOR_SEARCH_INDEX_NAME}",
        public_endpoint_enabled=True,
    )
    index_endpoint_id = index_endpoint.resource_name.split("/")[-1]
    print(f"Index Endpoint created with ID: {index_endpoint_id}")

    # Deploy the index to the endpoint
    deployed_index_id = f"deployed-{index_id}"
    index_endpoint.deploy_index(index=index, deployed_index_id=deployed_index_id)
    print("Index deployed.")

    # Save the IDs to a config file
    config = {
        "INDEX_ID": index_id,
        "DEPLOYED_INDEX_ID": deployed_index_id,
        "INDEX_ENDPOINT_ID": index_endpoint_id
    }
    
    with open("config.json", "w") as config_file:
        json.dump(config, config_file)
    print("Index and endpoint IDs saved to config.json.")

if __name__ == "__main__":
    main()
