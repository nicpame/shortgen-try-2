import db.db as db
import dl.dl as dl
from state.update_state_after_batch_dl import update_state_after_batch_dl

# step 1: Import candidate source vids to the database - source_vids table
# db.import_candidate_source_vids_to_db()


# step 2: dl vids in source_vids table
# dl.dl_batch_vids(db.get_candidate_source_vids())

# step 3: Process the downloaded videos and update the database
update_state_after_batch_dl(db.get_candidate_source_vids())
