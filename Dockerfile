# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install dependencies (no-cache-dir to keep image size small)
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code and models
COPY src/ /app/src/

# Expose port 8000 for FastAPI
EXPOSE 8000

# Set the pythonpath so it can find the modules
ENV PYTHONPATH=/app/src

# Run Uvicorn when the container launches
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
