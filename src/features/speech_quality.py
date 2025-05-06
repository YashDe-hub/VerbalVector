"""
Speech quality analysis for VerbalVector.

This module analyzes pronunciation and grammar in speech transcripts.
"""

import re
import logging
from typing import Dict, Any, List, Tuple
import pronouncing
import ollama

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SpeechQualityAnalyzer:
    """Class for analyzing pronunciation and grammar in speech."""
    
    def __init__(self, ollama_client: ollama.Client = None):
        """Initialize the speech quality analyzer."""
        self.ollama_client = ollama_client
        
        # Common words that might be mispronounced
        self.common_words = {
            'data': ['DAA-tuh', 'DAY-tuh'],
            'either': ['EE-thur', 'EYE-thur'],
            'neither': ['NEE-thur', 'NYE-thur'],
            'schedule': ['SKED-jool', 'SHED-jool'],
            'tomato': ['tuh-MAY-toh', 'tuh-MAH-toh'],
            'potato': ['puh-TAY-toh', 'puh-TAH-toh'],
            'vase': ['vays', 'vahz'],
            'route': ['root', 'rowt'],
            'caramel': ['KAR-uh-muhl', 'KAR-muhl'],
            'pecan': ['pih-KAN', 'PEE-kan'],
            'almond': ['AH-mund', 'AL-mund'],
            'coyote': ['KAI-ote', 'kai-OH-tee'],
            'envelope': ['EN-vuh-lope', 'AHN-vuh-lope'],
            'garage': ['guh-RAHZH', 'GAR-ij'],
            'mobile': ['MOH-buhl', 'MOH-beel'],
            'often': ['AW-fuhn', 'OFF-tuhn'],
            'pajamas': ['puh-JAH-muhz', 'puh-JAM-uhz'],
            'pasta': ['PAH-stuh', 'PAS-tuh'],
            'pizza': ['PEET-suh', 'PIT-suh'],
            'prescription': ['prih-SKRIP-shuhn', 'pruh-SKRIP-shuhn'],
            'realtor': ['REE-uhl-tur', 'REEL-tur'],
            'realtor': ['REE-uhl-tur', 'REEL-tur'],
            'sherbet': ['SHUR-buht', 'SHUR-bert'],
            'status': ['STAY-tuhs', 'STAT-uhs'],
            'suite': ['sweet', 'soot'],
            'supposedly': ['suh-POH-zid-lee', 'suh-POZ-id-lee'],
            'temperature': ['TEM-pruh-chur', 'TEM-puh-chur'],
            'theater': ['THEE-uh-tur', 'THEE-ay-tur'],
            'through': ['throo', 'thru'],
            'tomorrow': ['tuh-MOR-oh', 'tuh-MAH-row'],
            'umbrella': ['uhm-BRELL-uh', 'uhm-BRELL-uh'],
            'usually': ['YOO-zhuh-lee', 'YOO-zhuh-wuh-lee'],
            'vehicle': ['VEE-uh-kuhl', 'VEE-hi-kuhl'],
            'vitamin': ['VAI-tuh-min', 'VIH-tuh-min'],
            'water': ['WAW-tur', 'WAH-tur'],
            'Wednesday': ['WENZ-day', 'WED-uhnz-day'],
            'zebra': ['ZEE-bruh', 'ZEB-ruh']
        }
    
    def analyze_speech(self, transcript: str, model_name: str = "gemma2:9b") -> Dict[str, Any]:
        """
        Analyze speech quality including pronunciation and grammar.
        
        Args:
            transcript: The speech transcript text
            model_name: The Ollama model to use for grammar analysis
            
        Returns:
            Dictionary containing pronunciation and grammar analysis
        """
        # Analyze pronunciation
        pronunciation_analysis = self._analyze_pronunciation(transcript)
        
        # Analyze grammar using LLM
        grammar_analysis = self._analyze_grammar(transcript, model_name)
        
        return {
            'pronunciation': pronunciation_analysis,
            'grammar': grammar_analysis
        }
    
    def _analyze_pronunciation(self, transcript: str) -> Dict[str, Any]:
        """
        Analyze pronunciation in the transcript.
        
        Args:
            transcript: The speech transcript text
            
        Returns:
            Dictionary containing pronunciation analysis
        """
        # Split into words and clean
        words = re.findall(r'\b\w+\b', transcript.lower())
        
        # Track mispronounced words
        mispronounced_words = []
        
        # Check each word
        for word in words:
            # Skip if word is too short or contains numbers
            if len(word) < 3 or any(c.isdigit() for c in word):
                continue
                
            # Get pronunciations from CMU dictionary
            pronunciations = pronouncing.phones_for_word(word)
            
            if not pronunciations:
                # Word not found in dictionary, check common words
                if word in self.common_words:
                    mispronounced_words.append({
                        'word': word,
                        'correct_pronunciations': self.common_words[word],
                        'context': self._get_word_context(transcript, word)
                    })
                continue
            
            # Check if pronunciation is unusual
            # This is a simple check - in practice, you'd want more sophisticated analysis
            if len(pronunciations) > 1:
                # Word has multiple valid pronunciations
                mispronounced_words.append({
                    'word': word,
                    'correct_pronunciations': pronunciations,
                    'context': self._get_word_context(transcript, word)
                })
        
        return {
            'mispronounced_words': mispronounced_words,
            'total_words_checked': len(words),
            'words_with_issues': len(mispronounced_words)
        }
    
    def _analyze_grammar(self, transcript: str, model_name: str) -> Dict[str, Any]:
        """
        Analyze grammar in the transcript using an LLM.
        
        Args:
            transcript: The speech transcript text
            model_name: The Ollama model to use
            
        Returns:
            Dictionary containing grammar analysis
        """
        if not self.ollama_client:
            logger.error("Ollama client not provided to SpeechQualityAnalyzer.")
            return {
                'grammar_issues': [],
                'total_issues': 0,
                'issues_by_type': {},
                'error': 'Ollama client not configured.'
            }
            
        logger.info(f"Analyzing grammar using Ollama model: {model_name}")

        # --- LLM Call Placeholder ---
        # TODO: Construct prompt, call self.ollama_client.chat(), parse response
        prompt = f"""Analyze the grammar of the following text. Identify any grammatical errors, explain them clearly, and suggest corrections. Focus only on grammar, not style or spelling unless it affects grammar. Present the findings as a list of issues.

Text:
{transcript}

Respond with a JSON list where each item has 'error', 'explanation', and 'suggestion' keys. If no errors, return an empty list []."""

        try:
            response = self.ollama_client.chat(
                model=model_name,
                messages=[{'role': 'user', 'content': prompt}],
                format='json' # Request JSON output
            )
            
            # Attempt to parse the JSON content from the response
            # Note: Ollama might return the JSON string within response['message']['content']
            # Error handling needed here based on actual Ollama client behavior
            llm_output_content = response.get('message', {}).get('content', '[]')
            
            # Basic validation/parsing - enhance as needed
            import json
            try:
                grammar_issues = json.loads(llm_output_content)
                if not isinstance(grammar_issues, list):
                    raise ValueError("LLM response is not a JSON list.")
            except (json.JSONDecodeError, ValueError) as e:
                 logger.error(f"Failed to parse LLM grammar response: {e}\nResponse content: {llm_output_content}")
                 grammar_issues = [{'error': 'LLM Response Parsing Error', 'explanation': str(e), 'suggestion': None}]


        except Exception as e:
            logger.error(f"Error during Ollama grammar analysis: {e}")
            grammar_issues = [{'error': 'LLM Call Failed', 'explanation': str(e), 'suggestion': None}]
            
        # --- End LLM Call Placeholder ---

        # Structure the output similarly to before, but based on LLM response
        return {
            'grammar_issues': grammar_issues,
            'total_issues': len(grammar_issues),
            'issues_by_type': self._count_issues_by_type(grammar_issues)
        }
    
    def _get_word_context(self, text: str, word: str, context_size: int = 50) -> str:
        """
        Get context around a word in the text.
        
        Args:
            text: The full text
            word: The word to find context for
            context_size: Number of characters to include before and after
            
        Returns:
            Context string
        """
        # Find all occurrences of the word
        pattern = r'\b' + re.escape(word) + r'\b'
        matches = list(re.finditer(pattern, text.lower()))
        
        if not matches:
            return ""
            
        # Get context around the first occurrence
        match = matches[0]
        start = max(0, match.start() - context_size)
        end = min(len(text), match.end() + context_size)
        
        return text[start:end]
    
    def _count_issues_by_type(self, issues: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Count grammar issues by type (adapts based on LLM output structure).
        Uses the 'error' key from the LLM response if available, otherwise 'type'.
        """
        counts = {}
        for issue in issues:
            # Try to use 'error' key first, fall back to 'type' if defined differently
            issue_type = issue.get('error', issue.get('type', 'Unknown')) 
            counts[issue_type] = counts.get(issue_type, 0) + 1
        return counts


def analyze_speech_quality(transcript: str, ollama_client: ollama.Client, model_name: str) -> Dict[str, Any]:
    """
    Analyze speech quality including pronunciation and grammar.
    
    Args:
        transcript: The speech transcript text
        ollama_client: The Ollama client instance
        model_name: The Ollama model to use
        
    Returns:
        Dictionary containing speech quality analysis
    """
    logger.info("Analyzing speech quality (pronunciation, grammar)...")
    analyzer = SpeechQualityAnalyzer(ollama_client=ollama_client)
    return analyzer.analyze_speech(transcript, model_name=model_name) 