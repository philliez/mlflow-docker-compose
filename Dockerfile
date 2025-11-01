# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Make port 8050 available to the world outside this container
EXPOSE 8050

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Run the pipeline script when the container launches
CMD ["python", "tradestorun/Session 1/run_full_pipeline.py"]
