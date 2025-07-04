def video_dir_name_from_id(vid_id: int) -> str:
    return str(vid_id).zfill(3)  # Pad with zeros to ensure at least 3 digits