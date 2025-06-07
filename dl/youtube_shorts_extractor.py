#!/usr/bin/env python3
"""
YouTube Channel Shorts Extractor
Extracts all YouTube Shorts from a channel and sorts them by date (newest first)
"""

import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import yt_dlp


class YouTubeShortsExtractor:
    def __init__(self, output_dir: str = "channel_shorts"):
        """Initialize the YouTube Shorts extractor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_channel_id(self, url: str) -> str:
        """Extract channel ID or handle from various YouTube channel URL formats."""
        patterns = [
            r'youtube\.com/channel/([^/?]+)',
            r'youtube\.com/c/([^/?]+)', 
            r'youtube\.com/user/([^/?]+)',
            r'youtube\.com/@([^/?]+)',
            r'youtube\.com/([^/?]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        
        # If no pattern matches, assume it's already a channel ID/handle
        return url.split('/')[-1]
    
    def get_channel_info(self, channel_url: str) -> Dict[str, Any]:
        """Get basic channel information."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                return {
                    'channel_id': info.get('id'),
                    'channel_title': info.get('title'),
                    'channel_url': info.get('webpage_url'),
                    'uploader': info.get('uploader'),
                    'subscriber_count': info.get('subscriber_count'),
                    'video_count': info.get('video_count', 0)
                }
        except Exception as e:
            raise Exception(f"Failed to extract channel info: {str(e)}")
    
    def get_all_videos(self, channel_url: str, max_videos: Optional[int] = None) -> List[Dict[str, Any]]:
        """Extract all videos from a channel."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True,
            'ignoreerrors': True,
        }
        
        # Add playlistend if max_videos is specified
        if max_videos:
            ydl_opts['playlistend'] = max_videos
        
        try:
            print(f"Extracting videos from channel: {channel_url}")
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(channel_url, download=False)
                
                if 'entries' not in info:
                    return []
                
                videos = []
                total_videos = len(info['entries'])
                print(f"Found {total_videos} total videos in channel")
                
                for i, entry in enumerate(info['entries'], 1):
                    if entry is None:
                        continue
                    
                    video_info = {
                        'id': entry.get('id'),
                        'title': entry.get('title'),
                        'url': entry.get('url'),
                        'webpage_url': entry.get('webpage_url', f"https://www.youtube.com/watch?v={entry.get('id')}"),
                        'upload_date': entry.get('upload_date'),
                        'duration': entry.get('duration'),
                        'view_count': entry.get('view_count'),
                        'uploader': entry.get('uploader'),
                    }
                    
                    videos.append(video_info)
                    
                    # Progress indicator
                    if i % 50 == 0:
                        print(f"Processed {i}/{total_videos} videos...")
                
                print(f"Extracted {len(videos)} video entries")
                return videos
                
        except Exception as e:
            raise Exception(f"Failed to extract videos: {str(e)}")
    
    def get_video_details(self, video_url: str) -> Dict[str, Any]:
        """Get detailed information about a single video to determine if it's a Short."""
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(video_url, download=False)
                return info
        except Exception as e:
            print(f"Warning: Could not get details for {video_url}: {str(e)}")
            return {}
    
    def is_youtube_short(self, video_info: Dict[str, Any], detailed_info: Dict[str, Any] = None) -> bool:
        """
        Determine if a video is a YouTube Short based on various criteria.
        
        YouTube Shorts criteria:
        1. Duration <= 60 seconds
        2. Aspect ratio is vertical (height > width)
        3. URL contains 'shorts' (not always reliable)
        """
        # Check duration from basic info first
        duration = video_info.get('duration')
        if duration and duration <= 60:
            return True
        
        # If we have detailed info, check more criteria
        if detailed_info:
            detailed_duration = detailed_info.get('duration')
            if detailed_duration and detailed_duration <= 60:
                # Additional checks for aspect ratio
                width = detailed_info.get('width')
                height = detailed_info.get('height')
                
                if width and height:
                    # Vertical aspect ratio indicates Short
                    if height > width:
                        return True
                    # Square aspect ratio might also be a Short if duration <= 60
                    elif height == width and detailed_duration <= 60:
                        return True
        
        # Check if URL contains 'shorts' (less reliable but sometimes helpful)
        webpage_url = video_info.get('webpage_url', '')
        if 'shorts' in webpage_url:
            return True
        
        return False
    
    def filter_shorts(self, videos: List[Dict[str, Any]], 
                     check_detailed: bool = True,
                     max_concurrent: int = 5) -> List[Dict[str, Any]]:
        """Filter videos to only include YouTube Shorts."""
        shorts = []
        
        print(f"Filtering {len(videos)} videos to find YouTube Shorts...")
        
        # First pass: filter by duration from basic info
        potential_shorts = []
        for video in videos:
            duration = video.get('duration')
            if duration and duration <= 60:
                potential_shorts.append(video)
            elif not duration:  # Unknown duration, might be a short
                potential_shorts.append(video)
        
        print(f"Found {len(potential_shorts)} potential shorts (duration <= 60s or unknown)")
        
        if not check_detailed:
            return potential_shorts
        
        # Second pass: get detailed info for more accurate filtering
        import concurrent.futures
        
        def check_single_video(video):
            try:
                detailed_info = self.get_video_details(video['webpage_url'])
                if self.is_youtube_short(video, detailed_info):
                    # Add detailed info to the video
                    video.update({
                        'detailed_duration': detailed_info.get('duration'),
                        'width': detailed_info.get('width'),
                        'height': detailed_info.get('height'),
                        'aspect_ratio': f"{detailed_info.get('width', 0)}x{detailed_info.get('height', 0)}",
                        'view_count': detailed_info.get('view_count', video.get('view_count')),
                        'like_count': detailed_info.get('like_count'),
                        'upload_date': detailed_info.get('upload_date', video.get('upload_date')),
                        'description': detailed_info.get('description', '')[:200] + '...' if detailed_info.get('description', '') else '',
                        'tags': detailed_info.get('tags', []),
                    })
                    return video
            except Exception as e:
                print(f"Error checking video {video.get('id', 'unknown')}: {str(e)}")
            return None
        
        # Use ThreadPoolExecutor for concurrent processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Process videos in batches to avoid overwhelming the API
            batch_size = 20
            for i in range(0, len(potential_shorts), batch_size):
                batch = potential_shorts[i:i + batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(potential_shorts) + batch_size - 1)//batch_size}")
                
                # Submit batch for processing
                future_to_video = {executor.submit(check_single_video, video): video for video in batch}
                
                # Collect results
                for future in concurrent.futures.as_completed(future_to_video):
                    result = future.result()
                    if result:
                        shorts.append(result)
        
        print(f"Confirmed {len(shorts)} YouTube Shorts after detailed analysis")
        return shorts
    
    def sort_by_date(self, shorts: List[Dict[str, Any]], newest_first: bool = True) -> List[Dict[str, Any]]:
        """Sort shorts by upload date."""
        def parse_date(date_str):
            if not date_str:
                return datetime.min
            try:
                # YouTube date format: YYYYMMDD
                return datetime.strptime(str(date_str), '%Y%m%d')
            except (ValueError, TypeError):
                return datetime.min
        
        sorted_shorts = sorted(
            shorts,
            key=lambda x: parse_date(x.get('upload_date')),
            reverse=newest_first
        )
        
        return sorted_shorts
    
    def save_results(self, shorts: List[Dict[str, Any]], channel_info: Dict[str, Any], 
                    filename: str = None) -> str:
        """Save the shorts data to a JSON file."""
        if not filename:
            channel_name = channel_info.get('channel_title', 'unknown_channel')
            safe_name = re.sub(r'[<>:"/\\|?*]', '_', channel_name)
            filename = f"{safe_name}_shorts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        result_data = {
            'extraction_info': {
                'extracted_at': datetime.now().isoformat(),
                'total_shorts_found': len(shorts),
                'channel_info': channel_info
            },
            'shorts': shorts
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
        
        return str(output_path)
    
    def extract_channel_shorts(self, channel_url: str, 
                              max_videos: Optional[int] = None,
                              check_detailed: bool = False,
                              save_to_file: bool = True) -> Dict[str, Any]:
        """
        Main method to extract all shorts from a channel.
        
        Args:
            channel_url: YouTube channel URL
            max_videos: Maximum number of videos to check (None for all)
            check_detailed: Whether to get detailed info for accurate filtering
            save_to_file: Whether to save results to JSON file
        
        Returns:
            Dictionary containing channel info and shorts list
        """
        try:
            print(f"Starting extraction for channel: {channel_url}")
            
            # Get channel information
            print("Getting channel information...")
            channel_info = self.get_channel_info(channel_url)
            print(f"Channel: {channel_info.get('channel_title')} ({channel_info.get('subscriber_count', 'unknown')} subscribers)")
            
            # Get all videos from channel
            print("Extracting all videos from channel...")
            all_videos = self.get_all_videos(channel_url, max_videos)
            
            if not all_videos:
                print("No videos found in channel")
                return {
                    'success': False,
                    'error': 'No videos found in channel',
                    'channel_info': channel_info,
                    'shorts': []
                }
            
            # Filter for shorts
            print("Filtering for YouTube Shorts...")
            shorts = self.filter_shorts(all_videos, check_detailed)
            
            # Sort by date (newest first)
            print("Sorting by date (newest first)...")
            sorted_shorts = self.sort_by_date(shorts, newest_first=True)
            
            # Save results if requested
            saved_file = None
            if save_to_file:
                print("Saving results to file...")
                saved_file = self.save_results(sorted_shorts, channel_info)
                print(f"Results saved to: {saved_file}")
            
            result = {
                'success': True,
                'channel_info': channel_info,
                'shorts': sorted_shorts,
                'total_shorts': len(sorted_shorts),
                'saved_file': saved_file
            }
            
            print(f"\n✅ Extraction completed!")
            print(f"Channel: {channel_info.get('channel_title')}")
            print(f"Total shorts found: {len(sorted_shorts)}")
            print(f"Date range: {sorted_shorts[0].get('upload_date') if sorted_shorts else 'N/A'} to {sorted_shorts[-1].get('upload_date') if sorted_shorts else 'N/A'}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error during extraction: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'channel_info': None,
                'shorts': []
            }
    
    def print_shorts_summary(self, shorts: List[Dict[str, Any]]):
        """Print a summary of found shorts."""
        if not shorts:
            print("No shorts found.")
            return
        
        print(f"\n📹 Found {len(shorts)} YouTube Shorts:")
        print("=" * 80)
        
        for i, short in enumerate(shorts[:10], 1):  # Show first 10
            title = short.get('title', 'Unknown Title')[:50]
            upload_date = short.get('upload_date', 'Unknown')
            duration = short.get('detailed_duration') or short.get('duration', 'Unknown')
            view_count = short.get('view_count', 'Unknown')
            
            # Format upload date
            try:
                if upload_date != 'Unknown':
                    formatted_date = datetime.strptime(str(upload_date), '%Y%m%d').strftime('%Y-%m-%d')
                else:
                    formatted_date = 'Unknown'
            except:
                formatted_date = str(upload_date)
            
            print(f"{i:2d}. {title}")
            print(f"    📅 {formatted_date} | ⏱️ {duration}s | 👀 {view_count} views")
            print(f"    🔗 {short.get('webpage_url', '')}")
            print()
        
        if len(shorts) > 10:
            print(f"... and {len(shorts) - 10} more shorts")


