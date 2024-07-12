from google.cloud import aiplatform
import vertexai
from vertexai.generative_models import GenerativeModel
import pandas as pd

# Initialize Vertex AI models
model = GenerativeModel("gemini-1.0-pro")

def generate_text_embedding(text, text_embedding_model):
    embeddings = text_embedding_model.get_embeddings([text])
    return embeddings[0].values

def get_prompt_text(question, context):
    return f"Answer the question using the context below. Respond with only from the text provided\nQuestion: {question}\nContext : {context}"

def get_answer(query, embedding_df, index_endpoint, text_embedding_model, deployed_index_id):
    query_embeddings = generate_text_embedding(query, text_embedding_model)

    response = index_endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_embeddings],
        num_neighbors=5,
    )

    for neighbor in response[0]:
        context = embedding_df[embedding_df["id"] == neighbor.id]["page_content"].values[0]
        prompt = get_prompt_text(query, context)
        result = model.generate_content(prompt).text
        if "The provided context does not contain information" not in result:
            page_source = embedding_df[embedding_df["id"] == neighbor.id]["page_source"].values[0]
            return result, page_source

    return "No relevant answer found", "./data/Images/blank.jpg"
