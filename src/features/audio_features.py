"""
Audio feature extraction for VerbalVector.

This module extracts paralinguistic features from audio files for analysis.
"""

import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from typing import Dict, Any, List, Tuple


class AudioFeatureExtractor:
    """Class for extracting presentation-relevant features from audio files."""
    
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize the audio feature extractor.
        
        Args:
            sample_rate: Target sample rate for audio processing
        """
        self.sample_rate = sample_rate
    
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """
        Extract full set of presentation features from an audio file.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary of extracted features
        """
        # Load audio file
        y, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Extract core features
        features = {}
        
        # Duration
        features['duration'] = librosa.get_duration(y=y, sr=sr)
        
        # Pitch features
        pitch_features = self._extract_pitch_features(y, sr)
        features.update(pitch_features)
        
        # Volume/energy features
        volume_features = self._extract_volume_features(y, sr)
        features.update(volume_features)
        
        # Speech rate features
        rate_features = self._extract_speech_rate_features(y, sr)
        features.update(rate_features)
        
        # Pause features
        pause_features = self._extract_pause_features(y, sr)
        features.update(pause_features)
        
        # Voice quality features
        quality_features = self._extract_voice_quality_features(y, sr)
        features.update(quality_features)
        
        return features
    
    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract pitch-related features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of pitch features
        """
        # Extract pitch (F0) using PYIN algorithm
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Filter out unvoiced segments
        f0_voiced = f0[voiced_flag]
        
        if len(f0_voiced) == 0:  # Handle case with no voiced segments
            return {
                'pitch_mean': 0.0,
                'pitch_std': 0.0,
                'pitch_min': 0.0, 
                'pitch_max': 0.0,
                'pitch_range': 0.0
            }
            
        # Calculate pitch statistics
        pitch_mean = np.mean(f0_voiced)
        pitch_std = np.std(f0_voiced)
        pitch_min = np.min(f0_voiced)
        pitch_max = np.max(f0_voiced)
        pitch_range = pitch_max - pitch_min
        
        return {
            'pitch_mean': float(pitch_mean),
            'pitch_std': float(pitch_std),
            'pitch_min': float(pitch_min),
            'pitch_max': float(pitch_max),
            'pitch_range': float(pitch_range)
        }
    
    def _extract_volume_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract volume/energy related features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of volume features
        """
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        
        # Calculate volume statistics
        volume_mean = np.mean(rms)
        volume_std = np.std(rms)
        volume_min = np.min(rms)
        volume_max = np.max(rms)
        volume_range = volume_max - volume_min
        
        # Calculate volume dynamics (rate of change)
        volume_gradient = np.gradient(rms)
        volume_changes = np.abs(volume_gradient)
        volume_change_rate = np.mean(volume_changes)
        
        return {
            'volume_mean': float(volume_mean),
            'volume_std': float(volume_std),
            'volume_min': float(volume_min),
            'volume_max': float(volume_max),
            'volume_range': float(volume_range),
            'volume_change_rate': float(volume_change_rate)
        }
    
    def _extract_speech_rate_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract speech rate features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of speech rate features
        """
        # Detect onsets (syllable-like units)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
        
        # Calculate speech rate metrics
        duration_seconds = librosa.get_duration(y=y, sr=sr)
        
        if len(onsets) == 0:
            speech_rate = 0
        else:
            # Approximate syllables per second
            speech_rate = len(onsets) / duration_seconds
        
        # Zero crossing rate can help identify speech rate patterns
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        zcr_mean = np.mean(zcr)
        
        return {
            'speech_rate': float(speech_rate),
            'zero_crossing_rate': float(zcr_mean)
        }
    
    def _extract_pause_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract pause-related features.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of pause features
        """
        # RMS energy for silence detection
        rms = librosa.feature.rms(y=y)[0]
        
        # Define silence threshold (can be adjusted based on recording conditions)
        silence_threshold = 0.01
        
        # Find silent regions
        is_silent = rms < silence_threshold
        
        # --- Replacement for librosa.util.find_runs --- 
        # Pad is_silent to correctly find runs at the edges
        padded_silence = np.concatenate(([False], is_silent, [False]))
        # Find where the silence state changes
        diffs = np.diff(padded_silence.astype(int))
        # Get the start indices (transition from False to True)
        starts = np.where(diffs == 1)[0]
        # Get the end indices (transition from True to False)
        ends = np.where(diffs == -1)[0]
        # Calculate runs: (start_index, run_length)
        # Note: ends[i] is the index *after* the run ends, so length is ends[i] - starts[i]
        silent_regions = [(starts[i], ends[i] - starts[i]) for i in range(len(starts))]
        # --- End replacement ---
        
        # Convert frames to seconds
        frame_length = 2048  # Default for librosa.feature.rms
        hop_length = 512     # Default for librosa.feature.rms
        
        # Filter for pauses longer than 0.5 seconds
        min_pause_frames = int(0.5 * sr / hop_length)
        pauses = [region for region in silent_regions if region[1] >= min_pause_frames]
        
        # Calculate pause metrics
        num_pauses = len(pauses)
        
        if num_pauses == 0:
            total_pause_duration = 0
            mean_pause_duration = 0
            pause_frequency = 0
        else:
            total_pause_duration = sum(region[1] for region in pauses) * hop_length / sr
            mean_pause_duration = total_pause_duration / num_pauses
            pause_frequency = num_pauses / librosa.get_duration(y=y, sr=sr)
        
        return {
            'num_pauses': num_pauses,
            'total_pause_duration': float(total_pause_duration),
            'mean_pause_duration': float(mean_pause_duration),
            'pause_frequency': float(pause_frequency)
        }
    
    def _extract_voice_quality_features(self, y: np.ndarray, sr: int) -> Dict[str, float]:
        """
        Extract voice quality features related to timbre and expressiveness.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of voice quality features
        """
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # MFCCs capture vocal tract configuration and timbre
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Statistical summaries
        centroid_mean = np.mean(spectral_centroid)
        bandwidth_mean = np.mean(spectral_bandwidth)
        contrast_mean = np.mean(np.mean(spectral_contrast, axis=1))
        
        # Summarize MFCCs
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_stds = np.std(mfccs, axis=1)
        
        # Create features dictionary
        features = {
            'spectral_centroid_mean': float(centroid_mean),
            'spectral_bandwidth_mean': float(bandwidth_mean),
            'spectral_contrast_mean': float(contrast_mean),
        }
        
        # Add MFCC features
        for i, (mean, std) in enumerate(zip(mfcc_means, mfcc_stds)):
            features[f'mfcc{i+1}_mean'] = float(mean)
            features[f'mfcc{i+1}_std'] = float(std)
        
        return features


def process_presentation(audio_path: str) -> Dict[str, Any]:
    """
    Process a presentation audio file and extract all features.
    
    Args:
        audio_path: Path to audio file
        
    Returns:
        Dictionary of extracted features
    """
    extractor = AudioFeatureExtractor()
    features = extractor.extract_features(audio_path)
    return features


if __name__ == "__main__":
    # Example usage
    import json
    
    # Path to test audio file
    test_audio = "data/raw/audio/sample_presentation.wav"
    
    if os.path.exists(test_audio):
        # Extract features
        features = process_presentation(test_audio)
        
        # Print features
        print(json.dumps(features, indent=2))
    else:
        print(f"Test file not found: {test_audio}")
        print("Please download a presentation audio file first.") 