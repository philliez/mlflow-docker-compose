# Connecting Google Colab to Your Local MLflow Server

This guide explains how to run your training scripts on Google Colab and have them report back to your local MLflow server. This is useful for leveraging Colab's free GPU resources for model training.

## 1. Expose Your Local MLflow Server

To allow Colab to access your local MLflow server, you need to expose it to the internet. We will use `ngrok` for this.

### a. Install ngrok

If you don't have ngrok, you can download it from [ngrok.com](https://ngrok.com/download).

### b. Run Your MLflow Server

Start your MLflow server as you normally would. For example, using the script we created:

```bash
./run_mlflow_server.py
```

This will start the MLflow UI on `http://127.0.0.1:5000`.

### c. Expose the Server with ngrok

In a new terminal, run the following command to expose your local port 5000 to the internet:

```bash
ngrok http 5000
```

Ngrok will give you a public URL (e.g., `https://<random-string>.ngrok.io`). This is the URL you will use to connect from Colab.

## 2. Run the Training Script on Colab

### a. Upload Your Project to Google Drive

Upload your entire project directory to your Google Drive. This will make it easy to access your files from Colab.

### b. Create a New Colab Notebook

Create a new Colab notebook and mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

### c. Install Dependencies

Install the necessary libraries in your Colab notebook:

```python
!pip install mlflow xgboost pandas pandas-ta
```

### d. Run the Training Script

Navigate to your project directory and run the training script, passing the ngrok URL as the `--mlflow-tracking-uri` argument:

```python
import os

# --- Change this to the path of your project in Google Drive ---
project_path = '/content/drive/MyDrive/mlflow-docker-compose'
os.chdir(project_path)

# --- Replace with your ngrok URL ---
ngrok_url = 'https://<random-string>.ngrok.io'

!python tradestorun/2_train_model.py --mlflow-tracking-uri {ngrok_url}
```

Now, your training run on Colab will be logged to your local MLflow server. You can view the results in your browser at `http://127.0.0.1:5000`.
