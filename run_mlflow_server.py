import subprocess
import os

def run_mlflow_server():
    mlruns_path = os.path.join(os.getcwd(), "mlruns")
    command = f"mlflow ui --backend-store-uri {mlruns_path} --port 5000"
    print(f"Starting MLflow UI with command: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting MLflow UI: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
    except FileNotFoundError:
        print("Error: 'mlflow' command not found. Please ensure MLflow is installed and in your PATH.")

if __name__ == "__main__":
    run_mlflow_server()
