import numpy as np
import os
import time
import requests
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "webbot"

# Specify the Pinecone index configuration
spec = ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust region if necessary

# Streamlit UI setup
st.set_page_config(page_title="RAG App with LLaMA and Pinecone", page_icon="📚", layout="centered")
st.title("RAG App with LLaMA and Pinecone")

# Initialize session state for chat history if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Input field for website URLs
urls_input = st.text_area("Enter website URLs (comma-separated):", key="urls_input")
urls = [url_input.strip() for url_input in urls_input.split(",") if url_input.strip()]

# Initialize HuggingFaceEmbeddings with a specific model name
embeddings = HuggingFaceEmbeddings()  # Example model

# Check if the index exists and create it if necessary
if index_name not in pc.list_indexes().names():
    # Create the index with the specified dimension (e.g., 768 for HuggingFaceEmbeddings model)
    pc.create_index(
        name=index_name,
        dimension=768,
        metric="cosine",
        spec=spec
    )
index = pc.Index(index_name)

# Display the "Generate Embeddings" button if embeddings are not loaded
if st.button("Generate Embeddings"):
    if urls:
        st.write("Generating new vector embeddings...")

        # Load documents from URLs
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        # Split text into chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
        text_chunks = text_splitter.split_documents(data)

        # Generate embeddings and upsert to Pinecone
        vectors = []
        for i, chunk in enumerate(text_chunks):
            vector = embeddings.embed_documents([chunk.page_content])[0]
            vectors.append({
                "id": f"doc_{i}",
                "values": vector,
                "metadata": {"text": chunk.page_content}
            })

        # Upsert vectors to Pinecone
        index.upsert(vectors)
        st.write(f"Total documents in Pinecone: {len(vectors)}")

# If embeddings are available, show the chat interface
if index:
    # Pinecone retriever function
    def retrieve_relevant_docs(query_input_p, k=5):
        # Generate query embedding
        query_embedding = embeddings.embed_query(query_input_p)

        # Check if the embedding is a numpy array and convert to a list of floats
        if isinstance(query_embedding, np.ndarray):
            query_embedding = query_embedding.tolist()  # Convert numpy array to list
        elif isinstance(query_embedding, list):
            query_embedding = [float(val) for val in query_embedding]  # Ensure the list values are floats
        else:
            raise ValueError("Query embedding is neither a numpy array nor a list.")

        # Query Pinecone index
        results = index.query(
            vector=query_embedding,
            top_k=k,
            include_values=False,
            include_metadata=True
        )

        # Debug output
        # st.write("Query:", query_input)
        # st.write("Query Embedding:", query_embedding)
        # st.write("Pinecone Results:", results)
        #
        # stats = index.describe_index_stats()
        # st.write("Pinecone Index Stats:", stats)

        # Return the relevant metadata (documents)
        if results['matches']:
            return [result['metadata']['text'] for result in results['matches']]
        else:
            return ["No relevant documents found."]


    # Hugging Face Inference API URL for distilGPT2 model
    API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
    headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}


    # Function to send requests to the Hugging Face API
    # Function to send requests to the Hugging Face API
    def query(payload, max_length=500, min_length=150, length_penalty=1.0, num_beams=4):
        # Add more parameters to control the summary length and quality
        payload["parameters"] = {
            "max_length": max_length,  # Set the maximum length for the output
            "min_length": min_length,  # Set the minimum length for the output
            "length_penalty": length_penalty,  # Adjust the penalty for generating shorter sequences
            "num_beams": num_beams  # Use beam search for more coherent and diverse responses
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()


    # Function to get response with retries
    def get_response_with_retry(full_input, retries=3, delay=5):
        for attempt in range(retries):
            response = query({"inputs": full_input})
            if isinstance(response, list) and len(response) > 0 and "summary_text" in response[0]:
                return response[0]['summary_text']
            elif isinstance(response, dict) and "summary_text" in response:
                return response['generated_text']
            elif isinstance(response, dict) and "error" in response:
                if "loading" in response["error"]:
                    st.warning(f"Model is loading, retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    st.error(f"Unexpected error: {response['error']}")
                    break
            else:
                st.error("Unexpected response format.")
                break
        return "Error: No response generated after retries."


    # Chat interface for user input
    query_input = st.chat_input("Ask me anything:")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # When user sends a query, retrieve relevant documents and respond
    if query_input:
        st.chat_message("user").markdown(query_input)
        st.session_state.chat_history.append({"role": "user", "content": query_input})

        # Retrieve relevant document chunks based on the query
        relevant_docs = retrieve_relevant_docs(query_input)
        context = " ".join(relevant_docs) if relevant_docs else "No relevant context found."

        # Prepare the full input for the model query
        full_input = "\nQuery Input:\n" + query_input + "\nContext:\n" + context

        # Attempt to get a response with retry mechanism
        response_text = get_response_with_retry(full_input)

        # Append assistant's response to chat history
        st.session_state.chat_history.append({"role": "assistant", "content": response_text})

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.markdown(response_text)
