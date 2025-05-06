"""
Text feature extraction for VerbalVector.

This module extracts linguistic features from presentation transcripts.
"""

import os
import re
import json
import string
import numpy as np
from typing import Dict, Any, List, Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist


# Download required NLTK data
def download_nltk_data():
    """Download required NLTK data if not already present."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
        
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')


class TextFeatureExtractor:
    """Class for extracting presentation-relevant features from transcripts."""
    
    def __init__(self):
        """Initialize the text feature extractor."""
        # Download required data
        download_nltk_data()
        
        # Common filler words in presentations
        self.filler_words = [
            'um', 'uh', 'ah', 'er', 'like', 'you know', 'sort of', 'kind of',
            'i mean', 'basically', 'actually', 'literally', 'so', 'right',
            'well', 'anyway', 'okay', 'mmmm', 'hmm'
        ]
    
    def extract_features(self, transcript_path: str) -> Dict[str, Any]:
        """
        Extract presentation-relevant features from a transcript.
        
        Args:
            transcript_path: Path to transcript file
            
        Returns:
            Dictionary of extracted text features
        """
        # Read transcript
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read()
        
        # Extract all features
        features = {}
        
        # Basic metrics
        basic_metrics = self._extract_basic_metrics(transcript)
        features.update(basic_metrics)
        
        # Filler word analysis
        filler_metrics = self._extract_filler_word_metrics(transcript)
        features.update(filler_metrics)
        
        # Vocabulary richness
        vocab_metrics = self._extract_vocabulary_metrics(transcript)
        features.update(vocab_metrics)
        
        # Sentence structure
        sentence_metrics = self._extract_sentence_metrics(transcript)
        features.update(sentence_metrics)
        
        # Readability
        readability_metrics = self._calculate_readability_metrics(transcript)
        features.update(readability_metrics)
        
        return features
        
    def _extract_basic_metrics(self, text: str) -> Dict[str, Any]:
        """
        Extract basic text metrics.
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary of basic metrics
        """
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Filter out punctuation
        words = [word for word in words if word not in string.punctuation]
        
        # Calculate metrics
        word_count = len(words)
        sentence_count = len(sentences)
        avg_words_per_sentence = word_count / max(1, sentence_count)
        avg_word_length = sum(len(word) for word in words) / max(1, word_count)
        
        # Count unique words (case-insensitive)
        unique_word_count = len(set(word.lower() for word in words))
        
        return {
            'word_count': word_count,
            'sentence_count': sentence_count,
            'avg_words_per_sentence': float(avg_words_per_sentence),
            'avg_word_length': float(avg_word_length),
            'unique_word_count': unique_word_count
        }
    
    def _extract_filler_word_metrics(self, text: str) -> Dict[str, Any]:
        """
        Analyze filler word usage.
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary of filler word metrics
        """
        # Convert to lowercase
        text_lower = text.lower()
        
        # Count filler words
        filler_counts = {}
        total_fillers = 0
        
        for filler in self.filler_words:
            # Use word boundary regex to find whole words/phrases
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, text_lower))
            filler_counts[filler] = count
            total_fillers += count
        
        # Get word count
        words = word_tokenize(text)
        word_count = len([word for word in words if word not in string.punctuation])
        
        # Calculate filler word rate
        filler_rate = total_fillers / max(1, word_count) * 100  # as percentage
        
        return {
            'filler_word_counts': filler_counts,
            'total_filler_words': total_fillers,
            'filler_word_rate': float(filler_rate)
        }
    
    def _extract_vocabulary_metrics(self, text: str) -> Dict[str, Any]:
        """
        Analyze vocabulary richness.
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary of vocabulary metrics
        """
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Filter out punctuation and stop words
        stop_words = set(stopwords.words('english'))
        content_words = [word for word in words if word not in string.punctuation and word not in stop_words]
        
        # Calculate vocabulary metrics
        total_words = len(content_words)
        unique_words = len(set(content_words))
        
        # Type-token ratio (measure of lexical diversity)
        ttr = unique_words / max(1, total_words)
        
        # Calculate word frequency statistics
        freq_dist = FreqDist(content_words)
        most_common = freq_dist.most_common(10)
        
        # Calculate hapax legomena (words that appear only once)
        hapax = [word for word, freq in freq_dist.items() if freq == 1]
        hapax_percentage = len(hapax) / max(1, unique_words) * 100
        
        return {
            'lexical_diversity': float(ttr),
            'most_common_words': dict(most_common),
            'unique_content_words': unique_words,
            'hapax_percentage': float(hapax_percentage)
        }
    
    def _extract_sentence_metrics(self, text: str) -> Dict[str, Any]:
        """
        Analyze sentence structure.
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary of sentence metrics
        """
        # Tokenize sentences
        sentences = sent_tokenize(text)
        
        # Calculate sentence length statistics
        sentence_lengths = [len(word_tokenize(sentence)) for sentence in sentences]
        
        # Sentence length statistics
        mean_length = np.mean(sentence_lengths)
        median_length = np.median(sentence_lengths)
        min_length = np.min(sentence_lengths)
        max_length = np.max(sentence_lengths)
        std_length = np.std(sentence_lengths)
        
        # Distribution of sentence lengths
        length_distribution = {}
        for length in sentence_lengths:
            bin_key = f"{(length // 5) * 5}-{(length // 5) * 5 + 4}"
            if bin_key in length_distribution:
                length_distribution[bin_key] += 1
            else:
                length_distribution[bin_key] = 1
                
        # Normalize to percentages
        for key in length_distribution:
            length_distribution[key] = (length_distribution[key] / len(sentences)) * 100
        
        return {
            'mean_sentence_length': float(mean_length),
            'median_sentence_length': float(median_length),
            'min_sentence_length': int(min_length),
            'max_sentence_length': int(max_length),
            'std_sentence_length': float(std_length),
            'sentence_length_distribution': length_distribution
        }
    
    def _calculate_readability_metrics(self, text: str) -> Dict[str, Any]:
        """
        Calculate readability metrics.
        
        Args:
            text: Transcript text
            
        Returns:
            Dictionary of readability metrics
        """
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        
        # Filter out punctuation
        words = [word for word in words if word not in string.punctuation]
        
        # Count syllables (simplified approach)
        def count_syllables(word):
            # This is a simplified syllable counter
            word = word.lower()
            if len(word) <= 3:
                return 1
            
            # Count vowel groups
            vowels = "aeiouy"
            count = 0
            prev_is_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_is_vowel:
                    count += 1
                prev_is_vowel = is_vowel
                
            # Adjust for common patterns
            if word.endswith('e'):
                count -= 1
            if word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
                count += 1
            if count == 0:
                count = 1
                
            return count
        
        # Calculate total syllables
        total_syllables = sum(count_syllables(word) for word in words)
        
        # Count complex words (3+ syllables)
        complex_words = [word for word in words if count_syllables(word) >= 3]
        complex_word_count = len(complex_words)
        
        # Flesch-Kincaid Grade Level
        if len(sentences) == 0 or len(words) == 0:
            fk_grade = 0
        else:
            fk_grade = 0.39 * (len(words) / len(sentences)) + 11.8 * (total_syllables / len(words)) - 15.59
        
        # Simplified Gunning Fog Index
        if len(sentences) == 0 or len(words) == 0:
            fog_index = 0
        else:
            fog_index = 0.4 * ((len(words) / len(sentences)) + 100 * (complex_word_count / len(words)))
        
        return {
            'flesch_kincaid_grade': float(fk_grade),
            'gunning_fog_index': float(fog_index),
            'complex_word_percentage': float(complex_word_count / max(1, len(words)) * 100),
            'avg_syllables_per_word': float(total_syllables / max(1, len(words)))
        }


def process_transcript(transcript_path: str) -> Dict[str, Any]:
    """
    Process a presentation transcript and extract all features.
    
    Args:
        transcript_path: Path to transcript file
        
    Returns:
        Dictionary of extracted features
    """
    extractor = TextFeatureExtractor()
    features = extractor.extract_features(transcript_path)
    return features


if __name__ == "__main__":
    # Example usage
    
    # Path to test transcript file
    test_transcript = "data/raw/transcripts/sample_presentation.txt"
    
    if os.path.exists(test_transcript):
        # Extract features
        features = process_transcript(test_transcript)
        
        # Print features
        print(json.dumps(features, indent=2))
    else:
        print(f"Test file not found: {test_transcript}")
        print("Please download a presentation transcript first.") 