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

1.  **Install Required Packages:**

    -   Install packages such as `pymupdf`, `langchain`, `gradio`, `google-cloud-aiplatform`, and `langchain_google_vertexai`.
2.  **Restart Runtime:**

    -   After installing the packages, restart the runtime to ensure all packages are loaded correctly.

Authentication
--------------

1.  **Google Colab Authentication:**

    -   If running on Google Colab, authenticate your environment using Google Colab's authentication methods.
2.  **Vertex AI Workbench:**

    -   If using Vertex AI Workbench, skip the authentication step as it is not required.

Initializing Vertex AI
----------------------

1.  **Define Project Information:**

    -   Specify your Google Cloud project ID and location.
2.  **Initialize Vertex AI:**

    -   Initialize the Vertex AI environment with the specified project information.

Extracting Data from PDFs
-------------------------

1.  **Download Sample PDF and Images:**

    -   Download a sample PDF file and a default image to use when no results are found.
2.  **Process PDF to Images:**

    -   Split the PDF into images by rendering each page as an image using the `fitz` library.
3.  **Extract Data Using Gemini Vision Pro:**

    -   Load each image and use the Gemini Vision Pro model to extract text and tabular data from the images.
4.  **Store Extracted Information:**

    -   Store the extracted information in a Big Query for further processing.

Generating Text Embeddings
--------------------------

1.  **Initialize Text Embedding Model:**

    -   Use the `textembedding-gecko` model to generate embeddings for the extracted text data.
2.  **Generate Text Embeddings:**

    -   Create a function to generate text embeddings and apply it to the extracted text data to create a list of embeddings.
3.  **Store Embeddings:**

    -   Store the generated embeddings in a Big Query along with the corresponding text and image references.

Creating and Deploying a Vector Search Index
--------------------------------------------

1.  **Create Vector Search Index:**

    -   Define parameters for the vector search index, including the number of dimensions and distance measure type.
2.  **Save Embeddings to JSON:**

    -   Save the embeddings in JSONL format and upload them to a Google Cloud Storage bucket.
3.  **Create and Deploy Index:**

    -   Create a vector search index using the Vertex AI Matching Engine and deploy it to an endpoint.

Asking Questions to the PDF
---------------------------

1.  **Generate Query Embeddings:**

    -   Create a function to generate embeddings for the user's query.
2.  **Find Relevant Documents:**

    -   Use the vector search endpoint to find the most relevant documents based on the query embeddings.
3.  **Generate Answer:**

    -   Create a function to generate an answer by using the relevant documents as context for the Gemini Pro model.
4.  **Handle Multiple Attempts:**

    -   Implement logic to handle multiple attempts at finding a satisfactory answer, using the next most relevant document if necessary.

Using Gradio UI for Q&A
-----------------------

1.  **Set Up Gradio Interface:**

    -   Use Gradio to create a web-based interface for the question-answering system.
2.  **Define User Interactions:**

    -   Define how user inputs (queries) are processed and how the results (answers and images) are displayed.
3.  **Launch Gradio App:**

    -   Launch the Gradio app to make the question-answering system accessible via a web interface.

References
----------

-   Vertex AI documentation
-   LangChain documentation
-   Gradio documentation

This README provides a comprehensive overview and detailed instructions for setting up and running the multimodal RAG-based question-answering system using Google Cloud's Vertex AI and Gemini models. For more details, refer to the provided documentation links.