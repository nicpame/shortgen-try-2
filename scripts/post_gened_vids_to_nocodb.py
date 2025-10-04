import json
import requests
from typing import List, Dict, Any
import os
from dotenv import load_dotenv
load_dotenv()
def prepare_nocodb_data(file_path: str) -> List[Dict[str, Any]]:
    """
    Reads data from a JSON file, extracts video records from the 'gened_vids' 
    object, and prepares them for NocoDB.
    
    Nested dictionaries (like 'gen_config') and lists (like 'thumbnail') 
    are converted into JSON strings, suitable for insertion into NocoDB's 
    JSON/TEXT columns. Other fields (source_vid_id, state) are included as is.

    Args:
        file_path: The path to the input JSON file.

    Returns:
        A list of dictionaries, where each dictionary represents a single row
        to be inserted into the NocoDB table.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return []
    
    gened_vids = data.get("gened_vids", {})
    processed_records = []
    
    # These keys hold nested dicts or lists and should be serialized to JSON strings
    json_keys = [
        "gen_config", "translation", "informalization", "vocalization", 
        "speech", "thumbnail", "metadata"
    ]
    
    for vid_key, record in gened_vids.items():
        new_record = {}
        
        # Include all top-level keys in the final record
        for key, value in record.items():
            if key in json_keys:
                # Convert nested object (dict/list) to a JSON string
                if value is not None:
                    # Use ensure_ascii=False to correctly handle non-ASCII characters (like Persian)
                    new_record[key] = json.dumps(value, ensure_ascii=False)
                else:
                    new_record[key] = None
            else:
                # Include simple, non-nested values (e.g., source_vid_id, state, created_at, vid_dir) as is
                new_record[key] = value
        
        # NOTE: You might want to include the 'vid_key' from the outer dict 
        # as a column if it represents a unique ID:
        # new_record['video_key_id'] = vid_key

        processed_records.append(new_record)
        
    return processed_records

def insert_data_into_nocodb(
    base_url: str, 
    table_id: str, 
    api_token: str, 
    records: List[Dict[str, Any]]
) -> requests.Response:
    """
    Inserts a list of records into a NocoDB table using the NocoDB API.

    Args:
        base_url: The base URL of your NocoDB instance (e.g., "https://app.nocodb.com").
        table_id: The ID of the target table (e.g., 't-xxxxxx').
        api_token: Your NocoDB API token ('xc-token').
        records: A list of record dictionaries prepared by prepare_nocodb_data.

    Returns:
        The requests.Response object from the API call, or None if an error occurred.
    """
    # NocoDB bulk insert endpoint for Table Records (POST)
    url = f"{base_url.rstrip('/')}/api/v2/tables/{table_id}/records"
    
    headers = {
        "Content-Type": "application/json",
        # Authentication is handled via the 'xc-token' header
        "xc-token": api_token 
    }
    
    # The API expects an array of records in the body for bulk insert
    payload = records

    print(f"Attempting to insert {len(records)} records into table {table_id}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status() # Raise an exception for 4xx or 5xx status codes
        print("Success: Records inserted successfully.")
        return response
    except requests.exceptions.RequestException as e:
        print(f"API Request Failed:")
        print(f"  URL: {url}")
        print(f"  Status Code: {response.status_code if 'response' in locals() else 'N/A'}")
        print(f"  Error Detail: {response.text if 'response' in locals() else e}")
        return None

# --- Example Usage ---

# 1. Define placeholders for connection details
NOCODB_BASE_URL = "https://app.nocodb.com" # Replace with your NocoDB instance URL
NOCODB_TABLE_ID = "m5w6sslaf11tgnv"              # Replace with your table's UUID (e.g., 't-u765a9')
NOCODB_API_TOKEN = os.environ.get("NOCODB_API_TOKEN")  # Read from environment variable
INPUT_FILE_PATH = "data/db/db.json"


# 2. Prepare the data
records_to_insert = prepare_nocodb_data(INPUT_FILE_PATH)

# 3. Insert data via API (uncomment to run the API call)
if records_to_insert and NOCODB_API_TOKEN != "your_xc_token_here":
    response = insert_data_into_nocodb(
        NOCODB_BASE_URL, 
        NOCODB_TABLE_ID, 
        NOCODB_API_TOKEN, 
        records_to_insert
    )
    if response:
        print(f"API Response JSON: {response.json()}")
    pass
elif records_to_insert:
    print("\n--- Data Preview (Ready for NocoDB) ---")
    print(json.dumps(records_to_insert[0], indent=2, ensure_ascii=False))