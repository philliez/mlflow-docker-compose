import subprocess
import os
import argparse

def run_mlflow_server(host):
    mlruns_path = os.path.join(os.getcwd(), "mlruns")
    command = f"mlflow ui --backend-store-uri {mlruns_path} --port 5000 --host {host}"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="127.0.0.1", help="The host to bind the MLflow UI to.")
    args = parser.parse_args()
    run_mlflow_server(args.host)
