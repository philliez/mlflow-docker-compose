# Connecting Google Colab to Your Local MLflow Server

This guide explains how to run your training scripts on Google Colab and have them report back to your local MLflow server. This is useful for leveraging Colab's free GPU resources for model training.

## 1. Expose Your Local MLflow Server

To allow Colab to access your local MLflow server, you need to expose it to the internet. We will use `ngrok` for this.

### a. Install ngrok

If you don't have ngrok, you can download it from [ngrok.com](https://ngrok.com/download).

### b. Configure ngrok Authtoken

Before using ngrok, you need to configure it with your authtoken. You only need to do this once.

```bash
ngrok config add-authtoken 34tqkL02OcDDXoiUV9JQX9UVF2m_3fTVKxdvi4wmc5CyAYmwr
```

### c. Start the ngrok Tunnel

In a new terminal, run the following command to expose your local port 5000 to the internet:

```bash
ngrok http 5000
```

Ngrok will give you a public URL. Use the one provided to you: `https://untrying-noninterpretatively-skyla.ngrok-free.dev`.

### d. Run Your MLflow Server

Start your MLflow server, passing the ngrok hostname to the `--host` argument. **Note:** Use only the hostname from your ngrok URL (without `https://`).

```bash
./run_mlflow_server.py --host untrying-noninterpretatively-skyla.ngrok-free.dev
```

This will start the MLflow UI on `http://127.0.0.1:5000` and allow it to accept requests from your ngrok URL.

## 2. Run the Training Script on Colab

### a. Upload Your Project to Google Drive

Upload your entire project directory, including the `ohlc_data.zip` file, to your Google Drive. This will make it easy to access your files from Colab.

### b. Run the Colab Notebook

Upload and run the `colab_runner.ipynb` notebook in Google Colab. It will handle mounting your drive, installing dependencies, and running the training script.
