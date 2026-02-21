import os
from kaggle.api.kaggle_api_extended import KaggleApi


def get_kaggle_api():
    api = KaggleApi()
    api.authenticate()
    return api


def recommend_datasets(task_type, limit=5):
    if not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer.")

    normalized_task = str(task_type).strip().lower()
    query = "regression dataset" if normalized_task == "regression" else "classification dataset"

    api = get_kaggle_api()

    datasets = api.dataset_list(search=query)

    return [
        {"title": d.title, "ref": d.ref}
        for d in datasets[:limit]
    ]


def download_dataset(dataset_ref):
    if not isinstance(dataset_ref, str) or not dataset_ref.strip():
        raise ValueError("dataset_ref must be a non-empty string.")

    api = get_kaggle_api()

    save_path = os.path.join("datasets", "kaggle")
    os.makedirs(save_path, exist_ok=True)

    api.dataset_download_files(dataset_ref, path=save_path, unzip=True)

    return save_path
