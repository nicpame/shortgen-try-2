import json
import requests
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv
load_dotenv()
# ==============================================================================
# ⚠️ NOCODB CONFIGURATION (REPLACE WITH YOUR ACTUAL VALUES) ⚠️
# ==============================================================================
NOCODB_URL = "https://app.nocodb.com"
BASE_ID = "pwtbb9v5ojsl3eq"
TABLE_NAME = "youtube_vids"
TABLE_ID = "mzs1uxne05zintu"
API_TOKEN = os.environ.get("NOCODB_API_TOKEN")
UNIQUE_ID_COLUMN = "gened_vid_id" 
PRIMARY_KEY_COLUMN = "Id" 
# ==============================================================================
DB_FILE_PATH = "data/db/db.json"
NUMBER_OF_RECORDS_TO_PROCESS = None  # Set to None to process all records
PUBLISH_CHANNEL_ID = 1 # Id of the channel to publish to in nocodb (id 1 is farsifaunfact)

# API Endpoints
API_RECORDS_ENDPOINT = f"{NOCODB_URL}/api/v2/tables/{TABLE_ID}/records"
API_STORAGE_UPLOAD_ENDPOINT = f"{NOCODB_URL}/api/v2/storage/upload"
HEADERS = {
    "xc-token": API_TOKEN,
    "Content-Type": "application/json",
}

def upload_file_to_nocodb(file_path: str, mimetype: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Uploads a local file to NocoDB's storage and returns the attachment metadata.
    NocoDB attachment fields expect a list of these metadata objects.
    """
    if not os.path.exists(file_path):
        print(f"  [Error] File not found: {file_path}")
        return None

    try:
        # File upload requires multipart/form-data. requests handles this if 'files' is used.
        upload_headers = {"xc-token": API_TOKEN}

        # Open the file in binary mode
        with open(file_path, 'rb') as f:
            files = {
                'file': (os.path.basename(file_path), f, mimetype)
            }

            response = requests.post(
                API_STORAGE_UPLOAD_ENDPOINT,
                headers=upload_headers,
                files=files,
            )
            response.raise_for_status()
        
        # NocoDB's upload response is typically a list of uploaded files, return the first item.
        return response.json()[0]

    except requests.exceptions.RequestException as e:
        print(f"  [HTTP Error] Failed to upload {file_path}: {e}")
        return None
    except Exception as e:
        print(f"  [Error] An unexpected error occurred during file upload: {e}")
        return None

def upsert_data_to_nocodb(json_file_path: str):
    """
    Reads data from a JSON file and performs an upsert operation
    (update if gened_vid_id exists, create otherwise) on NocoDB.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading JSON file {json_file_path}: {e}")
        return
    
    gened_vids = data.get('gened_vids', {})

    if not gened_vids:
        print("No 'gened_vids' object found in the JSON data. Exiting.")
        return

    if NUMBER_OF_RECORDS_TO_PROCESS is not None:
        gened_vids = dict(list(gened_vids.items())[:NUMBER_OF_RECORDS_TO_PROCESS])
        print(f"Processing only the first {NUMBER_OF_RECORDS_TO_PROCESS} records for testing.")

    for gened_vid_id, gened_vid in gened_vids.items():
        print(f"\nProcessing gened_vid_id: {gened_vid_id}")

        try:
            # --- 1. Extract and map data ---
            metadata_variations = gened_vid.get('metadata', {}).get('variations', [{}])
            metadata = metadata_variations[0] if metadata_variations else {}
            
            video_file_path = gened_vid.get('output_video_file_dir')
            thumbnail_variations = gened_vid.get('thumbnail', [{}])
            thumbnail_file_path = thumbnail_variations[0].get('file_path') if thumbnail_variations else None

            # --- 2. Upload files to NocoDB ---
            video_upload_data = None
            if video_file_path:
                print(f"  Uploading video: {video_file_path}")
                video_upload_data = upload_file_to_nocodb(video_file_path, mimetype='video/mp4')
            
            thumbnail_upload_data = None
            if thumbnail_file_path:
                print(f"  Uploading thumbnail: {thumbnail_file_path}")
                thumbnail_upload_data = upload_file_to_nocodb(thumbnail_file_path, mimetype='image/png')
            
            # --- 3. Construct the record payload ---
            record_payload = {
                "title": metadata.get('video_title'),
                "description": metadata.get('video_description'),
                "state": "candidate",
                "language": "fa",
                UNIQUE_ID_COLUMN: gened_vid_id,
                
                # Attachment fields require a list of file metadata
                "video_file": [video_upload_data] if video_upload_data else [],
                "thumbnail_file": [thumbnail_upload_data] if thumbnail_upload_data else [],
                "publish_channel": [PUBLISH_CHANNEL_ID]
            }
            
            # Remove keys where the value is None/empty list/etc. if the field is not meant to be updated.
            # Keeping file fields as empty list '[]' is fine if they are meant to be cleared/not touched.

            # --- 4. Check if the record already exists ---
            # Using NocoDB v2 filter syntax in the query string: ?where=({column_name},eq,{value})
            query_params = {'where': f'({UNIQUE_ID_COLUMN},eq,{gened_vid_id})'}
            
            response = requests.get(
                API_RECORDS_ENDPOINT,
                headers={"xc-token": API_TOKEN}, # GET requests don't need Content-Type: application/json
                params=query_params
            )
            response.raise_for_status()
            existing_records = response.json().get('list', [])

            # --- 5. Upsert (Update or Create) ---
            if existing_records:
                # Get the primary key value for the specific record
                pk_value = existing_records[0].get(PRIMARY_KEY_COLUMN)
                
                if not pk_value:
                    print(f"  [Warning] Could not find primary key ({PRIMARY_KEY_COLUMN}) for existing record. Skipping update.")
                    continue
                
                # UPDATE existing record (PATCH)
                update_endpoint = f"{API_RECORDS_ENDPOINT}/{pk_value}"
                print(f"  Record found (PK: {pk_value}). Updating...")
                
                update_response = requests.patch(
                    update_endpoint,
                    headers=HEADERS,
                    json=record_payload
                )
                update_response.raise_for_status()
                print(f"  Successfully UPDATED record {gened_vid_id}.")
                
            else:
                # CREATE new record (POST)
                print("  Record not found. Creating new record...")
                
                create_response = requests.post(
                    API_RECORDS_ENDPOINT,
                    headers=HEADERS,
                    json=record_payload
                )
                create_response.raise_for_status()
                print(f"  Successfully CREATED new record for {gened_vid_id}.")

        except requests.exceptions.HTTPError as e:
            print(f"  [HTTP Error] NocoDB operation failed for {gened_vid_id}. Status: {e.response.status_code}")
            # print(f"  Response content: {e.response.text}") # Uncomment for debugging
        except Exception as e:
            print(f"  [Error] An unexpected error occurred for {gened_vid_id}: {e}")

# Example Usage:
# To make this example runnable, you would need to:
# 1. Create a mock JSON file (e.g., 'data.json') with the structure you provided.
# 2. Create the mock video and thumbnail files referenced in the JSON path.
# 3. Replace the placeholder values in the NOCODB_CONFIGURATION section above.



if __name__ == "__main__":
    # Assuming 'path/to/your_data.json' is the path to your file
    upsert_data_to_nocodb(DB_FILE_PATH)