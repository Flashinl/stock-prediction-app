"""
Grammar Processor for converting ASL word sequences to proper English grammar.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

class GrammarProcessor:
    """
    Processes ASL word sequences and applies English grammar rules
    to produce natural, readable text output.
    """
    
    def __init__(self):
        """Initialize the grammar processor with ASL-to-English rules."""
        
        # ASL grammar patterns and their English equivalents
        self.grammar_rules = {
            # Question patterns
            "wh_questions": {
                "what": ["what", "is", "that"],
                "where": ["where", "is"],
                "when": ["when", "will"],
                "who": ["who", "is"],
                "why": ["why", "do"],
                "how": ["how", "do"]
            },
            
            # Pronoun corrections
            "pronouns": {
                "me": "I",  # ASL often uses "me" where English uses "I"
                "point-self": "I",
                "point-you": "you",
                "point-there": "he/she/it"
            },
            
            # Verb tense markers
            "tense_markers": {
                "finish": "past",
                "will": "future",
                "now": "present",
                "yesterday": "past",
                "tomorrow": "future",
                "today": "present"
            },
            
            # Common ASL-to-English transformations
            "transformations": [
                # Remove redundant words common in ASL
                (r"\b(point-self|me)\s+(name)\b", "my name is"),
                (r"\b(you)\s+(name)\s+(what)\b", "what is your name"),
                (r"\b(finish)\s+", ""),  # Remove ASL past tense marker
                (r"\b(will)\s+", "will "),  # Keep future marker
                
                # Fix word order (ASL: Object-Subject-Verb, English: Subject-Verb-Object)
                (r"\b(water|food|book)\s+(I|you|he|she)\s+(want|need|like)\b", 
                 r"\2 \3 \1"),
                
                # Add articles where needed
                (r"\b(want|need|like)\s+(book|car|house|food)\b", r"\1 a \2"),
                (r"\b(see|watch)\s+(movie|tv|game)\b", r"\1 the \2"),
            ]
        }
        
        # Common English articles and prepositions to add
        self.articles = ["a", "an", "the"]
        self.prepositions = ["in", "on", "at", "to", "from", "with", "by"]
        
        # Word categories for better processing
        self.word_categories = {
            "pronouns": ["I", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"],
            "verbs": ["go", "come", "eat", "drink", "sleep", "work", "study", "play", "love", "like", "want", "need", "see", "watch"],
            "nouns": ["water", "food", "book", "car", "house", "movie", "tv", "game", "mother", "father", "sister", "brother"],
            "adjectives": ["good", "bad", "big", "small", "hot", "cold", "happy", "sad", "red", "blue", "green"],
            "time": ["today", "tomorrow", "yesterday", "now", "later", "morning", "afternoon", "evening", "night"]
        }
        
        logger.info("Grammar processor initialized")
    
    def process_sentence(self, words: List[str]) -> str:
        """
        Process a sequence of ASL words and convert to proper English grammar.
        
        Args:
            words: List of ASL words in order
            
        Returns:
            Grammatically correct English sentence
        """
        if not words:
            return ""
        
        # Clean and normalize input
        cleaned_words = self._clean_words(words)
        
        # Apply ASL-specific transformations
        transformed = self._apply_transformations(cleaned_words)
        
        # Fix word order
        reordered = self._fix_word_order(transformed)
        
        # Add missing articles and prepositions
        enhanced = self._add_missing_words(reordered)
        
        # Apply capitalization and punctuation
        final_sentence = self._finalize_sentence(enhanced)
        
        logger.debug(f"Processed: {words} -> {final_sentence}")
        return final_sentence
    
    def _clean_words(self, words: List[str]) -> List[str]:
        """Clean and normalize the input words."""
        cleaned = []
        
        for word in words:
            # Convert to lowercase for processing
            word = word.lower().strip()
            
            # Skip empty words
            if not word:
                continue
            
            # Handle special ASL markers
            if word in ["point-self", "me-point"]:
                word = "I"
            elif word in ["point-you", "you-point"]:
                word = "you"
            elif word in ["point-there", "point-him", "point-her"]:
                word = "he"  # Default to he, could be improved with context
            
            cleaned.append(word)
        
        return cleaned
    
    def _apply_transformations(self, words: List[str]) -> List[str]:
        """Apply ASL-to-English transformation rules."""
        sentence = " ".join(words)
        
        # Apply regex transformations
        for pattern, replacement in self.grammar_rules["transformations"]:
            sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
        
        # Handle question words
        sentence = self._process_questions(sentence)
        
        # Handle tense markers
        sentence = self._process_tense(sentence)
        
        return sentence.split()
    
    def _process_questions(self, sentence: str) -> str:
        """Process question patterns specific to ASL."""
        # ASL often puts question words at the end
        words = sentence.split()
        
        question_words = ["what", "where", "when", "who", "why", "how"]
        
        # Check if question word is at the end
        if words and words[-1] in question_words:
            question_word = words[-1]
            rest_of_sentence = words[:-1]
            
            # Move question word to the beginning
            if question_word == "what" and "name" in rest_of_sentence:
                return "what is your name"
            elif question_word == "where":
                return f"where is {' '.join(rest_of_sentence)}"
            elif question_word == "what":
                return f"what is {' '.join(rest_of_sentence)}"
            else:
                return f"{question_word} {' '.join(rest_of_sentence)}"
        
        return sentence
    
    def _process_tense(self, sentence: str) -> str:
        """Process tense markers and adjust verbs accordingly."""
        words = sentence.split()
        processed_words = []
        tense = "present"  # default
        
        i = 0
        while i < len(words):
            word = words[i]
            
            # Check for tense markers
            if word in self.grammar_rules["tense_markers"]:
                tense = self.grammar_rules["tense_markers"][word]
                
                # Skip the tense marker word (except "will" for future)
                if word != "will":
                    i += 1
                    continue
            
            # Adjust verbs based on tense
            if word in self.word_categories["verbs"]:
                word = self._conjugate_verb(word, tense)
            
            processed_words.append(word)
            i += 1
        
        return " ".join(processed_words)
    
    def _conjugate_verb(self, verb: str, tense: str) -> str:
        """Simple verb conjugation for common verbs."""
        if tense == "past":
            # Simple past tense rules
            past_forms = {
                "go": "went",
                "come": "came",
                "eat": "ate",
                "drink": "drank",
                "sleep": "slept",
                "see": "saw"
            }
            return past_forms.get(verb, verb + "ed")
        
        elif tense == "future":
            return f"will {verb}"
        
        # Present tense (default)
        return verb
    
    def _fix_word_order(self, words: List[str]) -> List[str]:
        """Fix word order from ASL (often SOV) to English (SVO)."""
        if len(words) < 3:
            return words
        
        # Simple pattern matching for common structures
        reordered = []
        i = 0
        
        while i < len(words):
            # Look for Object-Subject-Verb pattern
            if i + 2 < len(words):
                word1, word2, word3 = words[i], words[i+1], words[i+2]
                
                # Check if we have noun-pronoun-verb (ASL order)
                if (word1 in self.word_categories["nouns"] and 
                    word2 in self.word_categories["pronouns"] and 
                    word3 in self.word_categories["verbs"]):
                    
                    # Reorder to Subject-Verb-Object (English order)
                    reordered.extend([word2, word3, word1])
                    i += 3
                    continue
            
            # Default: keep the word as is
            reordered.append(words[i])
            i += 1
        
        return reordered
    
    def _add_missing_words(self, words: List[str]) -> List[str]:
        """Add missing articles, prepositions, and other function words."""
        enhanced = []
        
        for i, word in enumerate(words):
            # Add article before nouns when appropriate
            if (word in self.word_categories["nouns"] and 
                i > 0 and 
                words[i-1] in self.word_categories["verbs"]):
                
                # Add "a" or "the" based on context
                if word in ["water", "food"]:  # Uncountable nouns
                    enhanced.append(word)
                else:
                    enhanced.extend(["a", word])
            else:
                enhanced.append(word)
        
        return enhanced
    
    def _finalize_sentence(self, words: List[str]) -> str:
        """Apply final capitalization and punctuation."""
        if not words:
            return ""
        
        # Join words
        sentence = " ".join(words)
        
        # Capitalize first letter
        sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
        
        # Add period if no punctuation exists
        if not sentence.endswith(('.', '!', '?')):
            # Add question mark for questions
            if any(sentence.lower().startswith(qw) for qw in ["what", "where", "when", "who", "why", "how"]):
                sentence += "?"
            else:
                sentence += "."
        
        # Clean up extra spaces
        sentence = re.sub(r'\s+', ' ', sentence).strip()
        
        return sentence
    
    def get_grammar_suggestions(self, words: List[str]) -> List[str]:
        """
        Get alternative grammar suggestions for a word sequence.
        
        Args:
            words: List of ASL words
            
        Returns:
            List of alternative sentence structures
        """
        suggestions = []
        
        # Original processing
        original = self.process_sentence(words)
        suggestions.append(original)
        
        # Try different word orders
        if len(words) >= 3:
            # Try reversing the order
            reversed_words = words[::-1]
            reversed_sentence = self.process_sentence(reversed_words)
            if reversed_sentence != original:
                suggestions.append(reversed_sentence)
        
        # Try with different tense interpretations
        if "finish" not in words and "will" not in words:
            # Try past tense
            past_words = ["finish"] + words
            past_sentence = self.process_sentence(past_words)
            if past_sentence != original:
                suggestions.append(past_sentence)
        
        return list(set(suggestions))  # Remove duplicates
