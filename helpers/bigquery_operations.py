from google.cloud import bigquery

def create_dataset_if_not_exists(client, dataset_id):
    dataset_ref = client.dataset(dataset_id)
    try:
        client.get_dataset(dataset_ref)
        print(f"Dataset {dataset_id} already exists.")
    except Exception:
        dataset = bigquery.Dataset(dataset_ref)
        dataset = client.create_dataset(dataset)
        print(f"Created dataset {dataset_id}.")

def save_embeddings_to_bigquery(df, project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    create_dataset_if_not_exists(client, dataset_id)  # Ensure the dataset exists

    table_ref = client.dataset(dataset_id).table(table_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
    job.result()
    print(f"Loaded {job.output_rows} rows into {dataset_id}:{table_id}.")

def load_embeddings_from_bigquery(project_id, dataset_id, table_id):
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    return client.query(query).to_dataframe()
