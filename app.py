# Define project information
PROJECT_ID = "genai-projects-429119"  # @param {type:"string"}
LOCATION = "us-central1"  # @param {type:"string"}

# Initialize Vertex AI
import vertexai

# File system operations and displaying images
import os
os.environ["GOOGLE_CLOUD_PROJECT"] = "genai-projects-429119"

# Import utility functions for timing and file handling
import time

# Libraries for downloading files, data manipulation, and creating a user interface
import uuid
from datetime import datetime

import fitz
import gradio as gr
import pandas as pd

# Initialize Vertex AI libraries for working with generative models
from google.cloud import aiplatform
from PIL import Image as PIL_Image
from vertexai.generative_models import GenerativeModel, Image
from vertexai.language_models import TextEmbeddingModel


aiplatform.init(project="genai-projects-429119")
# Print Vertex AI SDK version
print(f"Vertex AI SDK version: {aiplatform.__version__}")

# Import LangChain components
import langchain

print(f"LangChain version: {langchain.__version__}")
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import DataFrameLoader

vertexai.init(project=PROJECT_ID, location=LOCATION)

#Initializing Gemini Vision Pro and Text Embedding models

# Loading Gemini Pro Vision Model
multimodal_model = GenerativeModel("gemini-1.0-pro-vision")

# Initializing embedding model
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")

# Loading Gemini Pro Model
model = GenerativeModel("gemini-1.0-pro")


# Create an "Images" directory if it doesn't exist
Image_Path = "./Images/"
if not os.path.exists(Image_Path):
    os.makedirs(Image_Path)

# Run the following code for each file
PDF_FILENAME = "04a02.pdf"  # Replace with your filename

# To get better resolution
zoom_x = 2.0  # horizontal zoom
zoom_y = 2.0  # vertical zoom
mat = fitz.Matrix(zoom_x, zoom_y)  # zoom factor 2 in each dimension

doc = fitz.open(PDF_FILENAME)  # open document
for page in doc:  # iterate through the pages
    pix = page.get_pixmap(matrix=mat)  # render page to an image
    outpath = f"./Images/{PDF_FILENAME}_{page.number}.jpg"
    pix.save(outpath)  # store image as a PNG

# Define the path where images are located
image_names = os.listdir(Image_Path)
Max_images = len(image_names)

# Create empty lists to store image information
page_source = []
page_content = []
page_id = []

p_id = 0  # Initialize image ID counter
rest_count = 0  # Initialize counter for error handling

while p_id < Max_images:
    try:
        # Construct the full path to the current image
        image_path = Image_Path + image_names[p_id]

        # Load the image
        image = Image.load_from_file(image_path)

        # Generate prompts for text and table extraction
        prompt_text = "Extract all text content in the image"
        prompt_table = (
            "Detect table in this image. Extract content maintaining the structure"
        )

        # Extract text using your multimodal model
        contents = [image, prompt_text]
        response = multimodal_model.generate_content(contents)
        text_content = response.text

        # Extract table using your multimodal model
        contents = [image, prompt_table]
        response = multimodal_model.generate_content(contents)
        table_content = response.text

        # Log progress and store results
        print(f"processed image no: {p_id}")
        page_source.append(image_path)
        page_content.append(text_content + "\n" + table_content)
        page_id.append(p_id)
        p_id += 1

    except Exception as err:
        # Handle errors during processing
        print(err)
        print("Taking Some Rest")
        time.sleep(1)  # Pause execution for 1 second
        rest_count += 1
        if rest_count == 5:  # Limit consecutive error handling
            rest_count = 0
            print(f"Cannot process image no: {image_path}")
            p_id += 1  # Move to the next image

# Create a DataFrame to store extracted information
df = pd.DataFrame(
    {"page_id": page_id, "page_source": page_source, "page_content": page_content}
)
del page_id, page_source, page_content  # Conserve memory
df.head()  # Preview the DataFrame

def generate_text_embedding(text) -> list:
    """Text embedding with a Large Language Model."""
    embeddings = text_embedding_model.get_embeddings([text])
    vector = embeddings[0].values
    return vector


# Create a DataFrameLoader to prepare data for LangChain
loader = DataFrameLoader(df, page_content_column="page_content")

# Load documents from the 'page_content' column of your DataFrame
documents = loader.load()

# Log the number of documents loaded
print(f"# of documents loaded (pre-chunking) = {len(documents)}")

# Create a text splitter to divide documents into smaller chunks
text_splitter = CharacterTextSplitter(
    chunk_size=10000,  # Target size of approximately 10000 characters per chunk
    chunk_overlap=200,  # overlap between chunks
)

# Split the loaded documents
doc_splits = text_splitter.split_documents(documents)

# Add a 'chunk' ID to each document split's metadata for tracking
for idx, split in enumerate(doc_splits):
    split.metadata["chunk"] = idx

# Log the number of documents after splitting
print(f"# of documents = {len(doc_splits)}")

texts = [doc.page_content for doc in doc_splits]
text_embeddings_list = []
id_list = []
page_source_list = []
for doc in doc_splits:
    id = uuid.uuid4()
    text_embeddings_list.append(generate_text_embedding(doc.page_content))
    id_list.append(str(id))
    page_source_list.append(doc.metadata["page_source"])
    time.sleep(1)  # So that we don't run into Quota Issue

# # Creating a dataframe of ID, embeddings, page_source and text
embedding_df = pd.DataFrame(
    {
        "id": id_list,
        "embedding": text_embeddings_list,
        "page_source": page_source_list,
        "text": texts,
    }
)
embedding_df.head()


VECTOR_SEARCH_REGION = "us-central1"
VECTOR_SEARCH_INDEX_NAME = f"{PROJECT_ID}-vector-search-index-ht"
VECTOR_SEARCH_EMBEDDING_DIR = f"{PROJECT_ID}-vector-search-bucket-ht"
VECTOR_SEARCH_DIMENSIONS = 768

# # save id and embedding as a json file
jsonl_string = embedding_df[["id", "embedding"]].to_json(orient="records", lines=True)
with open("data.json", "w") as f:
    f.write(jsonl_string)


# Generates a unique ID for session
UID = "07111951"

# # Creates a GCS bucket
# BUCKET_URI = f"gs://{VECTOR_SEARCH_EMBEDDING_DIR}-{UID}"
# ! gsutil mb -l $LOCATION -p {PROJECT_ID} {BUCKET_URI}
# ! gsutil cp data.json {BUCKET_URI}

BUCKET_URI = f"gs://{VECTOR_SEARCH_EMBEDDING_DIR}-{UID}"

# create index
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=f"{VECTOR_SEARCH_INDEX_NAME}",
    contents_delta_uri=BUCKET_URI,
    dimensions=768,
    approximate_neighbors_count=20,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)

# # # create IndexEndpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"{VECTOR_SEARCH_INDEX_NAME}",
    public_endpoint_enabled=True,
)

DEPLOYED_INDEX_NAME = VECTOR_SEARCH_INDEX_NAME.replace(
    "-", "_"
)  # Can't have - in deployment name, only alphanumeric and _ allowed
DEPLOYED_INDEX_ID = "f940139616353124352"
# deploy the Index to the Index Endpoint
# my_index_endpoint.deploy_index(index=my_index, deployed_index_id=DEPLOYED_INDEX_ID)

def Test_LLM_Response(txt):
    """
    Determines whether a given text response generated by an LLM indicates a lack of information.

    Args:
        txt (str): The text response generated by the LLM.

    Returns:
        bool: True if the LLM's response suggests it was able to generate a meaningful answer,
              False if the response indicates it could not find relevant information.

    This function works by presenting a formatted classification prompt to the LLM (`gemini_pro_model`).
    The prompt includes the original text and specific categories indicating whether sufficient information was available.
    The function analyzes the LLM's classification output to make the determination.
    """

    classification_prompt = f""" Classify the text as one of the following categories:
        -Information Present
        -Information Not Present
        Text=The provided context does not contain information.
        Category:Information Not Present
        Text=I cannot answer this question from the provided context.
        Category:Information Not Present
        Text:{txt}
        Category:"""
    classification_response = model.generate_content(classification_prompt).text

    if "Not Present" in classification_response:
        return False  # Indicates that the LLM couldn't provide an answer
    else:
        return True  # Suggests the LLM generated a meaningful response


def get_prompt_text(question, context):
    """
    Generates a formatted prompt string suitable for a language model, combining the provided question and context.

    Args:
        question (str): The user's original question.
        context (str): The relevant text to be used as context for the answer.

    Returns:
        str: A formatted prompt string with placeholders for the question and context, designed to guide the language model's answer generation.
    """
    prompt = """
      Answer the question using the context below. Respond with only from the text provided
      Question: {question}
      Context : {context}
      """.format(
        question=question, context=context
    )
    return prompt


def get_answer(query):
    """
    Retrieves an answer to a provided query using multimodal retrieval augmented generation (RAG).

    This function leverages a vector search system to find relevant text documents from a
    pre-indexed store of multimodal data. Then, it uses a large language model (LLM) to generate
    an answer, using the retrieved documents as context.

    Args:
        query (str): The user's original query.

    Returns:
        dict: A dictionary containing the following keys:
            * 'result' (str): The LLM-generated answer.
            * 'neighbor_index' (int): The index of the most relevant document used for generation
                                     (for fetching image path).

    Raises:
        RuntimeError: If no valid answer could be generated within the specified search attempts.
    """

    neighbor_index = 0  # Initialize index for tracking the most relevant document
    answer_found_flag = 0  # Flag to signal if an acceptable answer is found
    result = ""  # Initialize the answer string
    # Use a default image if the reference is not found
    page_source = "./Images/blank.jpg"  # Initialize the blank image
    query_embeddings = generate_text_embedding(
        query
    )  # Generate embeddings for the query

    response = my_index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embeddings],
        num_neighbors=5,
    )  # Retrieve up to 5 relevant documents from the vector store

    while answer_found_flag == 0 and neighbor_index < 4:
        context = embedding_df[
            embedding_df["id"] == response[0][neighbor_index].id
        ].text.values[
            0
        ]  # Extract text context from the relevant document

        prompt = get_prompt_text(
            query, context
        )  # Create a prompt using the question and context
        result = model.generate_content(prompt).text  # Generate an answer with the LLM

        if Test_LLM_Response(result):
            answer_found_flag = 1  # Exit loop when getting a valid response
        else:
            neighbor_index += (
                1  # Try the next retrieved document if the answer is unsatisfactory
            )

    if answer_found_flag == 1:
        page_source = embedding_df[
            embedding_df["id"] == response[0][neighbor_index].id
        ].page_source.values[
            0
        ]  # Extract image_path from the relevant document
    return result, page_source


query = (
    "what is the steps of Transformer Manufacturing Flow ?")

result, page_source = get_answer(query)
print(result)