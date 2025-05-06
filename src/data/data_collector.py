"""
Data collection utilities for VerbalVector, focusing on downloading data from YouTube.

This module provides functionality to collect TED talk data for training the presentation feedback model.
"""

import os
import json
import logging
from typing import List, Dict, Optional
import requests
from pydub import AudioSegment
import yt_dlp
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from pathlib import Path
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TEDTalkCollector:
    """Class for collecting TED talk data."""
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Initialize the TED talk collector.
        
        Args:
            output_dir: Directory to save downloaded content
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "transcripts"), exist_ok=True)
        
    def download_talk(self, video_url: str, output_filename: Optional[str] = None) -> Dict[str, str]:
        """
        Download a TED talk video and extract audio.
        
        Args:
            video_url: URL of the TED talk video
            output_filename: Base filename to use for saved files (without extension)
            
        Returns:
            Dictionary with paths to downloaded files
        """
        logger.info(f"Downloading TED talk: {video_url}")
        
        # Use yt-dlp to download video
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': os.path.join(self.output_dir, 'tmp_%(title)s.%(ext)s'),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'skip_download': False,
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=True)
            title = info_dict.get('title', 'unknown_title').replace(' ', '_')
            
            if output_filename is None:
                output_filename = title
                
            # Get downloaded files
            audio_path = os.path.join(self.output_dir, f"tmp_{title}.wav")
            subtitle_path = os.path.join(self.output_dir, f"tmp_{title}.en.vtt")
            
            # Move files to final locations
            final_audio_path = os.path.join(self.output_dir, "audio", f"{output_filename}.wav")
            final_transcript_path = os.path.join(self.output_dir, "transcripts", f"{output_filename}.txt")
            
            # Convert audio if needed
            audio = AudioSegment.from_wav(audio_path)
            audio.export(final_audio_path, format="wav")
            
            # Convert subtitles to plain text
            transcript = self._convert_vtt_to_text(subtitle_path)
            with open(final_transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)
                
            # Clean up temporary files
            os.remove(audio_path)
            os.remove(subtitle_path)
            
            return {
                'audio_path': final_audio_path,
                'transcript_path': final_transcript_path,
                'title': title
            }
    
    def download_multiple_talks(self, video_urls: List[str]) -> List[Dict[str, str]]:
        """
        Download multiple TED talks.
        
        Args:
            video_urls: List of TED talk URLs
            
        Returns:
            List of dictionaries with paths to downloaded files
        """
        results = []
        for url in video_urls:
            try:
                result = self.download_talk(url)
                results.append(result)
            except Exception as e:
                logger.error(f"Error downloading {url}: {str(e)}")
        
        return results
    
    def _convert_vtt_to_text(self, vtt_path: str) -> str:
        """
        Convert VTT subtitle file to plain text.
        
        Args:
            vtt_path: Path to VTT file
            
        Returns:
            Plain text transcript
        """
        with open(vtt_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip header and timing information
        text_lines = []
        for line in lines:
            line = line.strip()
            if (line and not line.startswith('WEBVTT') and 
                not line.startswith('NOTE') and 
                not '-->' in line and 
                not line.isdigit()):
                text_lines.append(line)
        
        return ' '.join(text_lines)


def get_ted_talk_urls(num_talks: int = 10, category: str = "technology") -> List[str]:
    """
    Get URLs for popular TED talks.
    
    Args:
        num_talks: Number of talks to retrieve
        category: Category of talks to retrieve
        
    Returns:
        List of TED talk URLs
    """
    # For the PoC, we could hardcode a few high-quality TED talk URLs
    # In a real implementation, this would scrape TED.com or use an API
    
    sample_urls = [
        "https://www.youtube.com/watch?v=8S0FDjFBj8o",  # The power of vulnerability | Brené Brown
        "https://www.youtube.com/watch?v=iCvmsMzlF7o",  # How great leaders inspire action | Simon Sinek
        "https://www.youtube.com/watch?v=H14bBuluwB8",  # Your body language may shape who you are | Amy Cuddy
        "https://www.youtube.com/watch?v=Unzc731iCUY",  # How to speak so that people want to listen | Julian Treasure
        "https://www.youtube.com/watch?v=qp0HIF3SfI4",  # The skill of self confidence | Dr. Ivan Joseph
    ]
    
    return sample_urls[:min(num_talks, len(sample_urls))]


def download_youtube_audio(url: str, output_dir: str, file_prefix: str) -> (Path | None):
    """
    Downloads the best audio stream from a YouTube URL using yt-dlp.

    Args:
        url: The YouTube video URL.
        output_dir: The directory to save the downloaded audio.
        file_prefix: The base name for the output file (e.g., 'talk_1').

    Returns:
        The Path object to the downloaded audio file, or None if download failed.
    """
    output_template = str(Path(output_dir) / f"{file_prefix}.%(ext)s")
    ydl_opts = {
        'format': 'bestaudio/best', # Download best audio quality
        'outtmpl': output_template, # Define output filename template
        'quiet': True, # Suppress yt-dlp console output
        'noplaylist': True, # Ensure only single video is downloaded
        'postprocessors': [{ # Optional: Convert to a specific format if needed
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav', # Extract to WAV format
            # 'preferredquality': '192', # Specify quality if needed
        }],
        'keepvideo': False, # Delete original downloaded file after processing
    }

    try:
        logger.info(f"Attempting to download audio for {url} using yt-dlp...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get info first to check availability and title (optional but good practice)
            info_dict = ydl.extract_info(url, download=False)
            title = info_dict.get('title', 'Unknown Title')
            logger.info(f"Found video: '{title}'. Proceeding with audio download...")

            # Perform the download and processing
            ydl.download([url])

            # Construct the final expected path after postprocessing
            # yt-dlp replaces %(ext)s with the final extension (wav in this case)
            final_extension = ydl_opts['postprocessors'][0]['preferredcodec']
            final_filename = f"{file_prefix}.{final_extension}"
            final_path = Path(output_dir) / final_filename

        if final_path.exists():
            logger.info(f"Successfully downloaded and processed audio: {final_path.name}")
            return final_path
        else:
            # This case might occur if postprocessing failed silently
            logger.error(f"Audio download command executed but output file not found: {final_path}")
            # Attempt to find *any* file with the prefix in case extension was unexpected
            found_files = list(Path(output_dir).glob(f"{file_prefix}.*"))
            if found_files:
                logger.warning(f"Found potential output file(s): {found_files}. Returning the first one.")
                return found_files[0]
            return None

    except yt_dlp.utils.DownloadError as e:
        # Handle specific yt-dlp download errors (e.g., video unavailable)
        logger.error(f"yt-dlp download error for {url}: {e}", exc_info=False)
        return None
    except Exception as e:
        # Handle other potential errors
        logger.error(f"Unexpected error downloading audio for {url}: {e}", exc_info=True) # Log full traceback
        return None

def download_youtube_transcript(url: str, output_dir: str, file_prefix: str) -> (Path | None):
    """
    Downloads the transcript for a YouTube URL.

    Args:
        url: The YouTube video URL.
        output_dir: The directory to save the transcript.
        file_prefix: The base name for the output file (e.g., 'talk_1').

    Returns:
        The Path object to the saved transcript JSON file, or None if failed.
    """
    try:
        video_id = url.split("watch?v=")[1].split('&')[0] # Handle extra URL params
        logger.info(f"Fetching transcript for video ID: {video_id} ({url})")

        # Attempt to get the transcript
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

        output_filename = f"{file_prefix}_transcript.json"
        output_path = Path(output_dir) / output_filename

        logger.info(f"Saving transcript to {output_path}...")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(transcript_list, f, indent=2)
        logger.info(f"Successfully saved transcript: {output_path.name}")
        return output_path

    except (TranscriptsDisabled, NoTranscriptFound):
        logger.warning(f"Transcript not available or disabled for {url}")
        return None
    except Exception as e:
        logger.error(f"Error downloading transcript for {url}: {e}", exc_info=False)
        return None

def download_batch_youtube(urls: List[str], output_dir: str) -> Dict[str, Dict[str, Path]]:
    """
    Downloads audio and transcripts for a list of YouTube URLs.

    Args:
        urls: A list of YouTube video URLs.
        output_dir: The base directory to save downloaded files.

    Returns:
        A dictionary mapping the original URL to a dict containing paths
        to the downloaded 'audio' and 'transcript' files (or None if failed).
        e.g., {'url1': {'audio': Path(...), 'transcript': Path(...)}, ...}
    """
    results = {}
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    logger.info(f"Starting batch download for {len(urls)} URLs to directory: '{output_dir}'")

    for i, url in enumerate(urls):
        logger.info(f"--- Processing URL {i+1}/{len(urls)}: {url} ---")
        file_prefix = f"youtube_video_{i+1}" # Generic prefix
        url_result = {'audio': None, 'transcript': None}

        # Download Audio
        audio_path = download_youtube_audio(url, output_dir, file_prefix)
        if audio_path:
            url_result['audio'] = audio_path

        # Download Transcript
        transcript_path = download_youtube_transcript(url, output_dir, file_prefix)
        if transcript_path:
            url_result['transcript'] = transcript_path

        results[url] = url_result

        # Optional: Add a small delay between requests to be polite to servers
        if i < len(urls) - 1:
            time.sleep(0.5) # Wait half a second

    logger.info("Batch download complete.")
    return results


if __name__ == "__main__":
    # Example usage
    collector = TEDTalkCollector()
    urls = get_ted_talk_urls(num_talks=2)
    results = collector.download_multiple_talks(urls)
    
    for result in results:
        print(f"Downloaded: {result['title']}")
        print(f"  Audio: {result['audio_path']}")
        print(f"  Transcript: {result['transcript_path']}")

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger.info("Running data_collector example...")

    # Sample list of short videos
    test_urls = [
        "https://www.youtube.com/watch?v=eIho2S0ZahI",  # Short TED talk
        "https://www.youtube.com/watch?v=4q1dgn_C0AU",  # Short TED talk
        "https://www.youtube.com/watch?v=arj7oStGLkU",  # Short talk (No transcript?)
        "https://www.youtube.com/watch?v=invalid_url_test" # Example of a bad URL
    ]

    output_directory = "data/raw_download_test"

    download_results = download_batch_youtube(test_urls, output_directory)

    print("\nDownload Summary:")
    for url, paths in download_results.items():
        audio_status = f"OK ('{paths['audio'].name}')" if paths['audio'] else "Failed"
        transcript_status = f"OK ('{paths['transcript'].name}')" if paths['transcript'] else "Failed/Unavailable"
        print(f"- {url}:")
        print(f"    Audio: {audio_status}")
        print(f"    Transcript: {transcript_status}")

    print(f"\nFiles saved in: {output_directory}") 