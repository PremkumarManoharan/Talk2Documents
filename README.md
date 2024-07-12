Building a Question Answering System with Multimodal Retrieval Augmented Generation (RAG)
=================================================================================================

This repository provides a guide to building a question-answering system using multimodal retrieval augmented generation (RAG). The system leverages Google's Gemini models, the Vertex AI API, and text embeddings to perform Q&A over documents containing both text and images.

[Watch the video Part - 1](https://www.loom.com/share/0265665f257049668f0fa5771bedc219?sid=95a28345-4e0f-4759-a4b8-0ff03bba229e)

[Watch the video Part - 2](https://www.loom.com/share/032378a93aed4dbc9c52b353f646d76c?sid=726f16a5-285d-4362-a340-31811fc898d7)

[Watch the video Part - 3](https://www.loom.com/share/cb5be35a53f347798c9069b1f3ffab0b?sid=08dffe8c-6958-478e-8f17-3b0f21839790)

Table of Contents
-----------------

- [Building a Question Answering System with Multimodal Retrieval Augmented Generation (RAG)](#building-a-question-answering-system-with-multimodal-retrieval-augmented-generation-rag)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Overview](#overview)
  - [Objectives](#objectives)
  - [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Authentication](#authentication)
  - [Initializing Vertex AI](#initializing-vertex-ai)
  - [Extracting Data from PDFs](#extracting-data-from-pdfs)
  - [Generating Text Embeddings](#generating-text-embeddings)
  - [Creating and Deploying a Vector Search Index](#creating-and-deploying-a-vector-search-index)
  - [Asking Questions to the PDF](#asking-questions-to-the-pdf)
  - [Using Gradio UI for Q\&A](#using-gradio-ui-for-qa)
  - [References](#references)

Introduction
------------

Retrieval augmented generation (RAG) enhances the capabilities of large language models (LLMs) by providing access to external data, thus improving their knowledge base and mitigating hallucinations. This notebook demonstrates how to implement a multimodal RAG system to perform Q&A over a document filled with both text and images.

Overview
--------

This notebook guides you through the steps to build a question-answering system using the Vertex AI Gemini API and text embeddings. The system extracts data from documents, stores it in a vector store, searches the store with text queries, and generates answers using the Gemini Pro Model.

Objectives
----------

-   Extract data from documents containing both text and images using Gemini Vision Pro.
-   Generate embeddings of the data and store them in a vector store.
-   Search the vector store with text queries to find relevant data.
-   Generate answers to user queries using the Gemini Pro Model.

Getting Started
---------------

To get started, you need a Google Cloud project with the Vertex AI API enabled. This section will guide you through the necessary setup and installation steps.

Installation
------------

First, install the required packages:

`pip install --upgrade --quiet pymupdf langchain gradio google-cloud-aiplatform langchain_google_vertexai
pip install langchain_community`

After installing the packages, restart the runtime:

python


`import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)`

Authentication
--------------

If running on Google Colab, authenticate your environment:

`import sys
if "google.colab" in sys.modules:
    from google.colab import auth
    auth.authenticate_user()`

Initializing Vertex AI
----------------------

Define your Google Cloud project information and initialize Vertex AI:

`'PROJECT_ID = "your-project-id"
LOCATION = "us-east1"`

`import vertexai
vertexai.init(project=PROJECT_ID, location=LOCATION)`

Extracting Data from PDFs
-------------------------

Process the PDF to extract data:

`import fitz
PDF_FILENAME = "04a02.pdf"
doc = fitz.open(PDF_FILENAME)
for page in doc:
    pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))
    outpath = f"./Images/{PDF_FILENAME}_{page.number}.jpg"
    pix.save(outpath)`

Generating Text Embeddings
--------------------------

Generate text embeddings using the textembedding-gecko model:

`from vertexai.language_models import TextEmbeddingModel
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")`

`def generate_text_embedding(text) -> list:
    embeddings = text_embedding_model.get_embeddings([text])
    return embeddings[0].values`

Creating and Deploying a Vector Search Index
--------------------------------------------

Create a vector search index to store and search through embeddings:


`VECTOR_SEARCH_REGION = "us-central1"
VECTOR_SEARCH_INDEX_NAME = f"{PROJECT_ID}-vector-search-index-ht"
VECTOR_SEARCH_EMBEDDING_DIR = f"{PROJECT_ID}-vector-search-bucket-ht"
VECTOR_SEARCH_DIMENSIONS = 768`

`# Save embeddings to Big Query
embedding_df = ... # DataFrame containing your embeddings
jsonl_string = embedding_df[["id", "embedding"]].to_json(orient="records", lines=True)
with open("data.json", "w") as f:
    f.write(jsonl_string)`

`# Upload to Google Cloud Storage
BUCKET_URI = f"gs://{VECTOR_SEARCH_EMBEDDING_DIR}-{UID}"
! gsutil mb -l $LOCATION -p {PROJECT_ID} {BUCKET_URI}
! gsutil cp data.json {BUCKET_URI}`

`# Create the index
from google.cloud import aiplatform
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=VECTOR_SEARCH_INDEX_NAME,
    contents_delta_uri=BUCKET_URI,
    dimensions=768,
    approximate_neighbors_count=20,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
)`

`# Create an Index Endpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=VECTOR_SEARCH_INDEX_NAME,
    public_endpoint_enabled=True,
)`

`# Deploy the index
DEPLOYED_INDEX_ID = f"{VECTOR_SEARCH_INDEX_NAME.replace('-', '_')}_{UID}"
my_index_endpoint.deploy_index(index=my_index, deployed_index_id=DEPLOYED_INDEX_ID)`

Asking Questions to the PDF
---------------------------

Create functions to handle queries and generate answers using the Gemini Pro model:

`def get_prompt_text(question, context):
    return f"Answer the question using the context below. Question: {question} Context: {context}"`

`def get_answer(query):
    query_embeddings = generate_text_embedding(query)
    response = my_index_endpoint.find_neighbors(
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query_embeddings],
        num_neighbors=5,
    )`

`query = "What are the steps of Transformer Manufacturing Flow?"
result, page_source = get_answer(query)
print(result)`

Using Gradio UI for Q&A
-----------------------

Create a web-based frontend using Gradio:


`import gradio as gr
from PIL import Image as PIL_Image`

`def gradio_query(query):
    result, image_path = get_answer(query)
    image = PIL_Image.open(image_path) if os.path.exists(image_path) else PIL_Image.open("./Images/blank.jpg")
    return result, image`

`with gr.Blocks() as demo:
    query = gr.Textbox(label="Query", info="Enter your query")
    btn_enter = gr.Button("Process")
    answer = gr.Textbox(label="Response", interactive=False)
    image = gr.Image(label="Reference")
    btn_enter.click(fn=gradio_query, inputs=query, outputs=[answer, image])
    demo.launch(share=True, debug=True, inbrowser=True)`

References
----------

-   Vertex AI documentation
-   LangChain documentation
-   Gradio documentation

This README provides a comprehensive overview and instructions for setting up and running the multimodal RAG-based question-answering system using Google Cloud's Vertex AI and Gemini models. For more details, refer to the provided documentation links.