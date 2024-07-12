from vertexai.generative_models import GenerativeModel, Image
from vertexai.language_models import TextEmbeddingModel
import pandas as pd
import uuid
import time

# Initialize Vertex AI models
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

def extract_text_and_generate_embeddings(image_paths):
    page_source = []
    page_content = []
    page_id = []
    text_embeddings_list = []
    id_list = []
    p_id = 0

    for image_path in image_paths:
        try:
            image = Image.load_from_file(image_path)
            prompt_text = "Extract all text content in the image"
            prompt_table = "Detect table in this image. Extract content maintaining the structure"

            contents = [image, prompt_text]
            response = multimodal_model.generate_content(contents)
            text_content = response.text

            contents = [image, prompt_table]
            response = multimodal_model.generate_content(contents)
            table_content = response.text

            page_source.append(image_path)
            page_content.append(text_content + "\n" + table_content)
            page_id.append(p_id)

            # Generate text embeddings
            embeddings = text_embedding_model.get_embeddings([text_content + "\n" + table_content])
            vector = embeddings[0].values
            text_embeddings_list.append(vector)
            id_list.append(str(uuid.uuid4()))

            p_id += 1
        except Exception as err:
            print(f"Error processing {image_path}: {err}")
            time.sleep(1)

    df = pd.DataFrame({"page_id": page_id, "page_source": page_source, "page_content": page_content, "id": id_list, "embedding": text_embeddings_list})
    return df
