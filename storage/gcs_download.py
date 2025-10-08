
# storage/gcs_download.py 

import os
from google.cloud import storage

# Configuración desde variables de entorno
GCS_BUCKET = os.getenv("GCS_BUCKET_NAME", "models_dl")
GCS_CREDENTIALS = os.getenv("GCS_CREDENTIALS_PATH", "storage/gcs-key.json")
MODELS = [
    "mba-api/baseline_lgb_model.txt",
    "mba-api/best_ncf_model.pt"
]
MODELS_DIR = os.getenv("MODELS_DIR", "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# Inicializar cliente de GCS
storage_client = storage.Client.from_service_account_json(GCS_CREDENTIALS)
bucket = storage_client.bucket(GCS_BUCKET)

def download_model(model_name):
    blob = bucket.blob(model_name)
    dest_path = os.path.join(MODELS_DIR, model_name)
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    print(f"Descargando {model_name} a {dest_path}...")
    blob.download_to_filename(dest_path)
    print(f"Descargado: {dest_path}")

if __name__ == "__main__":
    for model in MODELS:
        try:
            download_model(model)
        except Exception as e:
            print(f"Error al descargar {model}: {e}")
    print("Download complete")
