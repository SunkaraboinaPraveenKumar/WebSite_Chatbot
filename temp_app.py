# import os
# import time
# import json
# import streamlit as st
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from groq import Groq
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# # Streamlit UI setup
# st.set_page_config(
#     page_title="LLAMA WEB CHATBOT",
#     layout="centered"
# )
#
# # Initialize the chat history as streamlit session state if not present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Initialize the Groq client
# client = Groq()
#
# # Input field for website URLs (comma-separated)
# urls_input = st.text_area("Enter website URLs (comma-separated):")
# urls = [url_input.strip() for url_input in urls_input.split(",") if url_input.strip()]
#
# # Initialize HuggingFaceEmbeddings for document embedding (if needed)
# embeddings = HuggingFaceEmbeddings()  # Example model
#
# # Initialize Chroma parameters for storing vectors
# persist_directory = 'docs/chroma/'
# os.makedirs(persist_directory, exist_ok=True)
#
# # Load or create the vector store
# if os.path.exists(persist_directory) and os.listdir(persist_directory):
#     st.write("Loading existing vector embeddings from Chroma...")
#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# else:
#     vectordb = None
#
# # Function to generate embeddings and store them in Chroma
# def generate_embeddings():
#     if urls:
#         st.write("Generating new vector embeddings...")
#
#         # Load documents from URLs
#         loader = UnstructuredURLLoader(urls=urls)
#         data = loader.load()
#
#         # Split text into chunks for embedding
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#         text_chunks = text_splitter.split_documents(data)
#
#         # Store the documents and embeddings in Chroma
#         vectordb = Chroma.from_documents(
#             documents=text_chunks,
#             embedding=embeddings,  # Pass HuggingFace embeddings
#             persist_directory=persist_directory
#         )
#         st.write(f"Total documents in Chroma: {vectordb._collection.count()}")
#
# # Button to trigger the embedding generation
# if st.button("Generate Embeddings"):
#     generate_embeddings()
#
# # Create a retriever from the vector store if embeddings are generated
# retriever = None
# if vectordb:
#     retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#
# # Function to truncate text to fit within token limits
# def truncate_input(context, query_input, max_tokens=1024):
#     context_tokens = context.split()
#     query_tokens = query_input.split()
#     total_tokens = len(context_tokens) + len(query_tokens)
#
#     # If the total tokens exceed the max limit, truncate the context
#     if total_tokens > max_tokens:
#         available_tokens = max_tokens - len(query_tokens)  # Space for query tokens
#         truncated_context = ' '.join(context_tokens[:available_tokens])  # Truncate context
#         return truncated_context
#     return context
#
# # Function to interact with Groq's Llama-3.1 model
# def get_groq_response(messages):
#     response = client.chat.completions.create(
#         model="llama-3.1-8b-instant",
#         messages=messages
#     )
#     return response.choices[0].message.content
#
# # Streamlit chat interface for user input
# query_input = st.chat_input("Ask me anything:")
#
# system_prompt = (
#     "You are an assistant for question-answering tasks. "
#     "Use the following pieces of retrieved context to answer "
#     "the question. If you don't know the answer, say that you don't know. "
#     "Use three sentences maximum and keep the answer concise."
#     "\n\n"
# )
#
# # Display chat history
# for message in st.session_state.chat_history:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])
#
# # When user submits a query
# if query_input:
#     st.chat_message("user").markdown(query_input)
#     st.session_state.chat_history.append({"role": "user", "content": query_input})
#
#     # Retrieve relevant document chunks based on the query
#     if retriever:
#         relevant_docs = retriever.get_relevant_documents(query_input)
#         context = " ".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant context found."
#
#         # Truncate the context if it's too long considering the query
#         truncated_context = truncate_input(context, query_input)
#
#         # Prepare the full input for the model query
#         full_input = system_prompt + "\nContext:\n" + truncated_context + "\nUser Query:\n" + query_input
#
#         # Get response from Groq's Llama-3.1-8b-instant model
#         assistant_response = get_groq_response(messages=[{'role': "system", 'content': system_prompt}] + st.session_state.chat_history)
#
#         # Append assistant's response to the chat history
#         st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
#
#         # Display the assistant's response
#         with st.chat_message("assistant"):
#             st.markdown(assistant_response)








# import os
# import time
# import requests
# import streamlit as st
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from langchain.vectorstores import Chroma
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# # Streamlit UI setup
# st.set_page_config(page_title="RAG App Demo with LLaMA and Chroma", page_icon="📚", layout="centered")
# st.title("RAG App Demo with LLaMA and Chroma")
#
# # Initialize session state for chat history if not already present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Input field for website URLs
# urls_input = st.text_area("Enter website URLs (comma-separated):", key="urls_input")
# urls = [url_input.strip() for url_input in urls_input.split(",") if url_input.strip()]
#
# # Initialize HuggingFaceEmbeddings with a specific model name
# embeddings = HuggingFaceEmbeddings()  # Example model
#
# # Initialize Chroma parameters
# persist_directory = 'docs/chroma/'
# os.makedirs(persist_directory, exist_ok=True)
#
# # Load or create the vector store
# if os.path.exists(persist_directory) and os.listdir(persist_directory):
#     st.write("Loading existing vector embeddings from Chroma...")
#     vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#
# # Display the "Generate Embeddings" button if embeddings are not loaded
#
# if st.button("Generate Embeddings"):
#     if urls:
#         st.write("Generating new vector embeddings...")
#
#         # Load documents from URLs
#         loader = UnstructuredURLLoader(urls=urls)
#         data = loader.load()
#
#         # Split text into chunks for embedding
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#         text_chunks = text_splitter.split_documents(data)
#
#         # Store the documents and embeddings in Chroma
#         vectordb = Chroma.from_documents(
#             documents=text_chunks,
#             embedding=embeddings,  # Pass HuggingFace embeddings
#             persist_directory=persist_directory
#         )
#         st.write(f"Total documents in Chroma: {vectordb._collection.count()}")
#
# # If embeddings are available, show the chat interface
# if vectordb:
#     # Create a retriever from the vector store
#     retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
#
#     # Hugging Face Inference API URL for distilGPT2 model
#     API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
#     headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
#
#     # Function to send requests to the Hugging Face API
#     def query(payload, max_length=100):
#         payload["parameters"] = {"max_length": max_length}  # Set max_length for output
#         response = requests.post(API_URL, headers=headers, json=payload)
#         return response.json()
#
#     # Function to get response with retries
#     def get_response_with_retry(full_input, retries=3, delay=5):
#         for attempt in range(retries):
#             response = query({"inputs": full_input})
#             if isinstance(response, list) and len(response) > 0 and "summary_text" in response[0]:
#                 return response[0]['summary_text']
#             elif isinstance(response, dict) and "summary_text" in response:
#                 return response['generated_text']
#             elif isinstance(response, dict) and "error" in response:
#                 if "loading" in response["error"]:
#                     st.warning(f"Model is loading, retrying in {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     st.error(f"Unexpected error: {response['error']}")
#                     break
#             else:
#                 st.error("Unexpected response format.")
#                 break
#         return "Error: No response generated after retries."
#
#     # Chat interface for user input
#     query_input = st.chat_input("Ask me anything:")
#
#     # Display chat history
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     # When user sends a query, retrieve relevant documents and respond
#     if query_input:
#         st.chat_message("user").markdown(query_input)
#         st.session_state.chat_history.append({"role": "user", "content": query_input})
#
#         # Retrieve relevant document chunks based on the query
#         relevant_docs = retriever.get_relevant_documents(query_input)
#         context = " ".join([doc.page_content for doc in relevant_docs]) if relevant_docs else "No relevant context found."
#
#         # Prepare the full input for the model query
#         full_input = "\nQuery Input:\n" + query_input + "\nContext:\n" + context
#
#         # Attempt to get a response with retry mechanism
#         response_text = get_response_with_retry(full_input)
#
#         # Append assistant's response to chat history
#         st.session_state.chat_history.append({"role": "assistant", "content": response_text})
#
#         # Display the assistant's response
#         with st.chat_message("assistant"):
#             st.markdown(response_text)
#


# import numpy as np
# import os
# import time
# import requests
# import streamlit as st
# from langchain_community.document_loaders import UnstructuredURLLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceEmbeddings
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
#
# # Load environment variables
# load_dotenv()
#
# # Initialize Pinecone client
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index_name = "webbot"
#
# # Specify the Pinecone index configuration
# spec = ServerlessSpec(cloud='aws', region='us-east-1')  # Adjust region if necessary
#
# # Streamlit UI setup
# st.set_page_config(page_title="RAG App with LLaMA and Pinecone", page_icon="📚", layout="centered")
# st.title("RAG App with LLaMA and Pinecone")
#
# # Initialize session state for chat history if not already present
# if "chat_history" not in st.session_state:
#     st.session_state.chat_history = []
#
# # Input field for website URLs
# urls_input = st.text_area("Enter website URLs (comma-separated):", key="urls_input")
# urls = [url_input.strip() for url_input in urls_input.split(",") if url_input.strip()]
#
# # Initialize HuggingFaceEmbeddings with a specific model name
# embeddings = HuggingFaceEmbeddings()  # Example model
#
# # Check if the index exists and create it if necessary
# if index_name not in pc.list_indexes().names():
#     # Create the index with the specified dimension (e.g., 768 for HuggingFaceEmbeddings model)
#     pc.create_index(
#         name=index_name,
#         dimension=768,
#         metric="cosine",
#         spec=spec
#     )
# index = pc.Index(index_name)
#
# # Display the "Generate Embeddings" button if embeddings are not loaded
# if st.button("Generate Embeddings"):
#     if urls:
#         st.write("Generating new vector embeddings...")
#
#         # Load documents from URLs
#         loader = UnstructuredURLLoader(urls=urls)
#         data = loader.load()
#
#         # Split text into chunks for embedding
#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
#         text_chunks = text_splitter.split_documents(data)
#
#         # Generate embeddings and upsert to Pinecone
#         vectors = []
#         for i, chunk in enumerate(text_chunks):
#             vector = embeddings.embed_documents([chunk.page_content])[0]
#             vectors.append({
#                 "id": f"doc_{i}",
#                 "values": vector,
#                 "metadata": {"text": chunk.page_content}
#             })
#
#         # Upsert vectors to Pinecone
#         index.upsert(vectors)
#         st.write(f"Total documents in Pinecone: {len(vectors)}")
#
# # If embeddings are available, show the chat interface
# if index:
#     # Pinecone retriever function
#     def retrieve_relevant_docs(query_input_p, k=5):
#         # Generate query embedding
#         query_embedding = embeddings.embed_query(query_input_p)
#
#         # Check if the embedding is a numpy array and convert to a list of floats
#         if isinstance(query_embedding, np.ndarray):
#             query_embedding = query_embedding.tolist()  # Convert numpy array to list
#         elif isinstance(query_embedding, list):
#             query_embedding = [float(val) for val in query_embedding]  # Ensure the list values are floats
#         else:
#             raise ValueError("Query embedding is neither a numpy array nor a list.")
#
#
#         # Query Pinecone index
#         results = index.query(
#             vector=query_embedding,
#             top_k=k,
#             include_values=False,
#             include_metadata=True
#         )
#
#         # Debug output
#         # st.write("Query:", query_input)
#         # st.write("Query Embedding:", query_embedding)
#         # st.write("Pinecone Results:", results)
#         #
#         # stats = index.describe_index_stats()
#         # st.write("Pinecone Index Stats:", stats)
#
#         # Return the relevant metadata (documents)
#         if results['matches']:
#             return [result['metadata']['text'] for result in results['matches']]
#         else:
#             return ["No relevant documents found."]
#
#
#     # Hugging Face Inference API URL for distilGPT2 model
#     API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
#     headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}"}
#
#
#     # Function to send requests to the Hugging Face API
#     def query(payload, max_length=100):
#         payload["parameters"] = {"max_length": max_length}  # Set max_length for output
#         response = requests.post(API_URL, headers=headers, json=payload)
#         return response.json()
#
#
#     # Function to get response with retries
#     def get_response_with_retry(full_input, retries=3, delay=5):
#         for attempt in range(retries):
#             response = query({"inputs": full_input})
#             if isinstance(response, list) and len(response) > 0 and "summary_text" in response[0]:
#                 return response[0]['summary_text']
#             elif isinstance(response, dict) and "summary_text" in response:
#                 return response['generated_text']
#             elif isinstance(response, dict) and "error" in response:
#                 if "loading" in response["error"]:
#                     st.warning(f"Model is loading, retrying in {delay} seconds...")
#                     time.sleep(delay)
#                 else:
#                     st.error(f"Unexpected error: {response['error']}")
#                     break
#             else:
#                 st.error("Unexpected response format.")
#                 break
#         return "Error: No response generated after retries."
#
#
#     # Chat interface for user input
#     query_input = st.chat_input("Ask me anything:")
#
#     # Display chat history
#     for message in st.session_state.chat_history:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
#
#     # When user sends a query, retrieve relevant documents and respond
#     if query_input:
#         st.chat_message("user").markdown(query_input)
#         st.session_state.chat_history.append({"role": "user", "content": query_input})
#
#         # Retrieve relevant document chunks based on the query
#         relevant_docs = retrieve_relevant_docs(query_input)
#         context = " ".join(relevant_docs) if relevant_docs else "No relevant context found."
#
#         # Prepare the full input for the model query
#         full_input = "\nQuery Input:\n" + query_input + "\nContext:\n" + context
#
#         # Attempt to get a response with retry mechanism
#         response_text = get_response_with_retry(full_input)
#
#         # Append assistant's response to chat history
#         st.session_state.chat_history.append({"role": "assistant", "content": response_text})
#
#         # Display the assistant's response
#         with st.chat_message("assistant"):
#             st.markdown(response_text)
#
