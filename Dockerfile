# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the requirements.txt file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables if necessary (like streamlit or app configurations)
ENV STREAMLIT_SERVER_PORT 8501
ENV CHROMA_DIR ./docs/chroma

# Run the app
CMD ["streamlit", "run", "app.py"]
