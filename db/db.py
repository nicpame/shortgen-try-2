from tinydb import TinyDB, Query
import json

DB = './data/db/db.json'
CANDIDATE_SOURCE_VIDS = './data/candidate_source_vids.json'

db = TinyDB(DB, indent=4)
source_vids_db = db.table('source_vids')

def import_candidate_source_vids_to_db():
    with open(CANDIDATE_SOURCE_VIDS, 'r', encoding='utf-8') as f:
        data = json.load(f)

    inserted_count = 0
    for item in data:
        url = item.get('url')
        if url is None:
            continue  # skip items without a url
        if not source_vids_db.search(Query().url == url):
            item['state'] = 'candidate'
            source_vids_db.insert(item)
            inserted_count += 1
    print(f"Inserted {inserted_count} new items into the source_vids_db table.")
    
def get_candidate_source_vids():
    return source_vids_db.search(Query().state == 'candidate')


