FROM python:3.9-slim

# Install system dependencies required by OpenCV and other libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the current directory content into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install -r requirements.txt

# Run main.py when the container starts
CMD ["python", "main.py"]
