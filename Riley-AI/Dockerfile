# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Riley-Agent requirements and install them
COPY Riley-Agent/requirements.txt ./Riley-Agent-requirements.txt
RUN pip install --no-cache-dir -r Riley-Agent-requirements.txt

# Copy the application code
COPY . .

# Expose the port for the Streamlit app
EXPOSE 7860

# Command to run the Streamlit app
CMD ["python3", "streamlit_app.py"]