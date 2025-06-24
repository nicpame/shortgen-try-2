#!/usr/bin/env python3
"""
YouTube Video Downloader with Metadata and Transcript Extraction
Supports both regular YouTube URLs and YouTube Shorts URLs
"""

import json
import os
import sys
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import yt_dlp


#  config
yt_cookies_file_path = os.path.abspath('./dl/yt_cookies2.txt') 
print(f"Using cookies file: {yt_cookies_file_path}")

class YouTubeDownloader:
    def __init__(self, output_dir: str = "downloads"):
        """Initialize the YouTube downloader with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.default_ydl_opts = {
            'cookiefile': yt_cookies_file_path,
        }
    def set_output_dir(self, output_dir: str):
        """Set a new output directory for downloads."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to be safe for filesystem."""
        # Remove or replace invalid characters
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Remove extra whitespace and dots
        filename = re.sub(r'\s+', ' ', filename).strip()
        filename = filename.strip('.')
        # Limit length
        if len(filename) > 200:
            filename = filename[:200]
        return filename
    
    def extract_video_id(self, url: str) -> str:
        """Extract video ID from various YouTube URL formats."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([^&\n?#]+)',
            r'youtube\.com/embed/([^&\n?#]+)',
            r'youtube\.com/v/([^&\n?#]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If no pattern matches, assume it's already a video ID
        return url
    
    def get_video_info(self, url: str) -> Dict[str, Any]:
        """Extract video information without downloading."""
        ydl_opts = {
            **self.default_ydl_opts,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                return info
        except Exception as e:
            raise Exception(f"Failed to extract video info: {str(e)}")
    
    def extract_transcript(self, url: str) -> Optional[Dict[str, Any]]:
        """Extract transcript/subtitles from the video."""
        ydl_opts = {
            **self.default_ydl_opts,
            'quiet': True,
            'no_warnings': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en', 'en-US', 'en-GB'],
            'skip_download': True,
        }
        
        transcript_data = {
            'available': False,
            'manual_subtitles': {},
            'automatic_subtitles': {},
            'combined_text': ''
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                # Check for manual subtitles
                if 'subtitles' in info and info['subtitles']:
                    transcript_data['available'] = True
                    transcript_data['manual_subtitles'] = info['subtitles']
                
                # Check for automatic subtitles
                if 'automatic_captions' in info and info['automatic_captions']:
                    transcript_data['available'] = True
                    transcript_data['automatic_subtitles'] = info['automatic_captions']
                
                # Try to get the actual transcript text
                if transcript_data['available']:
                    # Prefer manual subtitles over automatic
                    subs_source = info.get('subtitles', {}) or info.get('automatic_captions', {})
                    
                    # Look for English subtitles
                    for lang in ['en', 'en-US', 'en-GB']:
                        if lang in subs_source:
                            # Get the subtitle URL and try to extract text
                            sub_info = subs_source[lang]
                            if sub_info and len(sub_info) > 0:
                                transcript_data['subtitle_info'] = sub_info[0]
                                break
                
        except Exception as e:
            print(f"Warning: Could not extract transcript: {str(e)}")
        
        return transcript_data
    
    def download_video(self, url: str, quality: str = 'best') -> Dict[str, Any]:
        """Download video and return information about the downloaded file."""
        try:
            # First, get video info
            info = self.get_video_info(url)
            
            # Create safe filename
            title = info.get('title', 'Unknown')
            safe_title = self.sanitize_filename(title)
            video_id = info.get('id', 'unknown')
            
            # Set up download options
            ydl_opts = {
                **self.default_ydl_opts,
                'format': quality,
                'outtmpl': str(self.output_dir / f"source_vid.%(ext)s"),
                'writethumbnail': True,
                'writeinfojson': False,  # We'll create our own JSON
            }
            
            # Download the video
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            
            return {
                'success': True,
                'title': title,
                'safe_title': safe_title,
                'video_id': video_id,
                'info': info
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_metadata_json(self, info: Dict[str, Any], transcript: Dict[str, Any], 
                           output_filename: str) -> str:
        """Create comprehensive metadata JSON file."""
        
        # Extract key metadata
        metadata = {
            'extraction_date': datetime.now().isoformat(),
            'video_info': {
                'id': info.get('id'),
                'title': info.get('title'),
                'uploader': info.get('uploader'),
                'uploader_id': info.get('uploader_id'),
                'upload_date': info.get('upload_date'),
                'description': info.get('description'),
                'duration': info.get('duration'),
                'view_count': info.get('view_count'),
                'like_count': info.get('like_count'),
                'comment_count': info.get('comment_count'),
                'age_limit': info.get('age_limit'),
                'webpage_url': info.get('webpage_url'),
                'thumbnail': info.get('thumbnail'),
                'tags': info.get('tags', []),
                'categories': info.get('categories', []),
                'is_live': info.get('is_live'),
                'was_live': info.get('was_live'),
            },
            'format_info': {
                'format': info.get('format'),
                'format_id': info.get('format_id'),
                'ext': info.get('ext'),
                'resolution': info.get('resolution'),
                'fps': info.get('fps'),
                'vcodec': info.get('vcodec'),
                'acodec': info.get('acodec'),
                'filesize': info.get('filesize'),
                'filesize_approx': info.get('filesize_approx'),
            },
            'channel_info': {
                'channel': info.get('channel'),
                'channel_id': info.get('channel_id'),
                'channel_url': info.get('channel_url'),
                'channel_follower_count': info.get('channel_follower_count'),
            },
            'transcript': transcript,
            'technical_info': {
                'extractor': info.get('extractor'),
                'extractor_key': info.get('extractor_key'),
                'epoch': info.get('epoch'),
                'webpage_url_basename': info.get('webpage_url_basename'),
            }
        }
        
        # Save metadata to JSON file
        json_filename = output_filename.replace('.mp4', '_metadata.json').replace('.webm', '_metadata.json')
        json_path = self.output_dir / json_filename
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
        
        return str(json_path)
    
    def process_video(self, url: str, quality: str = 'best', vid_output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Complete processing pipeline for a YouTube video."""
        print("Processing video for download...")
        print(f"url: {url}")
        print(f"dir: {vid_output_dir}")
        
        # Optionally set a different output directory for this video
        original_output_dir = self.output_dir
        if vid_output_dir:
            self.set_output_dir(vid_output_dir)

        try:
            # Download video
            print("Downloading video...")
            download_result = self.download_video(url, quality)

            if not download_result['success']:
                return {
                    'success': False,
                    'error': download_result['error']
                }

            # Create metadata JSON
            print("Creating metadata file...")
            safe_title = download_result['safe_title']
            video_id = download_result['video_id']

            json_path = self.create_metadata_json(
                info=download_result['info'],
                transcript={},
                output_filename='source_vid_metadata.json'
            )

            result = {
                'success': True,
                'title': download_result['title'],
                'video_id': video_id,
                'files': {
                    'video': f"{safe_title}_{video_id}.*",  # Extension depends on format
                    'metadata': json_path,
                    'thumbnail': f"{safe_title}_{video_id}.jpg"
                },
                # 'transcript_available': transcript['available']
            }

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

