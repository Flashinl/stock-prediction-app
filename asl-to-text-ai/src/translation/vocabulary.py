"""
ASL Vocabulary Management System for sign-to-word mapping.
"""

import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import sqlite3
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ASLWord:
    """Data class for ASL word information."""
    class_id: int
    word: str
    category: str
    description: str
    difficulty: str
    regional_variants: List[str]
    frequency: float
    confidence_threshold: float

class ASLVocabulary:
    """
    Manages the ASL vocabulary database with efficient lookup and search capabilities.
    """
    
    def __init__(self, vocabulary_path: Optional[str] = None):
        """
        Initialize the vocabulary system.
        
        Args:
            vocabulary_path: Path to vocabulary database file
        """
        self.vocabulary_path = vocabulary_path
        self.words_by_id = {}
        self.words_by_text = {}
        self.categories = set()
        self.total_words = 0
        
        # Load vocabulary
        if vocabulary_path and Path(vocabulary_path).exists():
            self.load_vocabulary(vocabulary_path)
        else:
            self._create_default_vocabulary()
        
        logger.info(f"ASL Vocabulary initialized with {self.total_words} words")
    
    def load_vocabulary(self, vocabulary_path: str) -> bool:
        """
        Load vocabulary from a file (JSON or SQLite).
        
        Args:
            vocabulary_path: Path to vocabulary file
            
        Returns:
            True if loaded successfully
        """
        try:
            path = Path(vocabulary_path)
            
            if path.suffix.lower() == '.json':
                return self._load_from_json(vocabulary_path)
            elif path.suffix.lower() == '.db':
                return self._load_from_sqlite(vocabulary_path)
            else:
                logger.error(f"Unsupported vocabulary format: {path.suffix}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load vocabulary from {vocabulary_path}: {e}")
            return False
    
    def _load_from_json(self, json_path: str) -> bool:
        """Load vocabulary from JSON file."""
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.words_by_id.clear()
        self.words_by_text.clear()
        self.categories.clear()
        
        for item in data.get('vocabulary', []):
            word = ASLWord(
                class_id=item['class_id'],
                word=item['word'],
                category=item.get('category', 'general'),
                description=item.get('description', ''),
                difficulty=item.get('difficulty', 'medium'),
                regional_variants=item.get('regional_variants', []),
                frequency=item.get('frequency', 0.5),
                confidence_threshold=item.get('confidence_threshold', 0.7)
            )
            
            self.words_by_id[word.class_id] = word
            self.words_by_text[word.word.lower()] = word
            self.categories.add(word.category)
        
        self.total_words = len(self.words_by_id)
        logger.info(f"Loaded {self.total_words} words from JSON")
        return True
    
    def _load_from_sqlite(self, db_path: str) -> bool:
        """Load vocabulary from SQLite database."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT class_id, word, category, description, difficulty, 
                   regional_variants, frequency, confidence_threshold
            FROM vocabulary
        """)
        
        self.words_by_id.clear()
        self.words_by_text.clear()
        self.categories.clear()
        
        for row in cursor.fetchall():
            regional_variants = json.loads(row[5]) if row[5] else []
            
            word = ASLWord(
                class_id=row[0],
                word=row[1],
                category=row[2] or 'general',
                description=row[3] or '',
                difficulty=row[4] or 'medium',
                regional_variants=regional_variants,
                frequency=row[6] or 0.5,
                confidence_threshold=row[7] or 0.7
            )
            
            self.words_by_id[word.class_id] = word
            self.words_by_text[word.word.lower()] = word
            self.categories.add(word.category)
        
        conn.close()
        self.total_words = len(self.words_by_id)
        logger.info(f"Loaded {self.total_words} words from SQLite")
        return True
    
    def _create_default_vocabulary(self):
        """Create a default vocabulary with common ASL words."""
        default_words = [
            # Basic greetings and common words
            {"class_id": 0, "word": "hello", "category": "greeting", "frequency": 0.9},
            {"class_id": 1, "word": "goodbye", "category": "greeting", "frequency": 0.8},
            {"class_id": 2, "word": "thank you", "category": "courtesy", "frequency": 0.9},
            {"class_id": 3, "word": "please", "category": "courtesy", "frequency": 0.8},
            {"class_id": 4, "word": "yes", "category": "response", "frequency": 0.9},
            {"class_id": 5, "word": "no", "category": "response", "frequency": 0.9},
            {"class_id": 6, "word": "sorry", "category": "courtesy", "frequency": 0.7},
            {"class_id": 7, "word": "help", "category": "action", "frequency": 0.8},
            {"class_id": 8, "word": "water", "category": "noun", "frequency": 0.7},
            {"class_id": 9, "word": "food", "category": "noun", "frequency": 0.7},
            
            # Pronouns
            {"class_id": 10, "word": "I", "category": "pronoun", "frequency": 0.9},
            {"class_id": 11, "word": "you", "category": "pronoun", "frequency": 0.9},
            {"class_id": 12, "word": "he", "category": "pronoun", "frequency": 0.8},
            {"class_id": 13, "word": "she", "category": "pronoun", "frequency": 0.8},
            {"class_id": 14, "word": "we", "category": "pronoun", "frequency": 0.8},
            {"class_id": 15, "word": "they", "category": "pronoun", "frequency": 0.8},
            
            # Common verbs
            {"class_id": 16, "word": "go", "category": "verb", "frequency": 0.8},
            {"class_id": 17, "word": "come", "category": "verb", "frequency": 0.8},
            {"class_id": 18, "word": "eat", "category": "verb", "frequency": 0.7},
            {"class_id": 19, "word": "drink", "category": "verb", "frequency": 0.7},
            {"class_id": 20, "word": "sleep", "category": "verb", "frequency": 0.6},
            {"class_id": 21, "word": "work", "category": "verb", "frequency": 0.7},
            {"class_id": 22, "word": "study", "category": "verb", "frequency": 0.6},
            {"class_id": 23, "word": "play", "category": "verb", "frequency": 0.6},
            {"class_id": 24, "word": "love", "category": "verb", "frequency": 0.7},
            {"class_id": 25, "word": "like", "category": "verb", "frequency": 0.7},
            
            # Time-related
            {"class_id": 26, "word": "today", "category": "time", "frequency": 0.8},
            {"class_id": 27, "word": "tomorrow", "category": "time", "frequency": 0.7},
            {"class_id": 28, "word": "yesterday", "category": "time", "frequency": 0.7},
            {"class_id": 29, "word": "now", "category": "time", "frequency": 0.8},
            {"class_id": 30, "word": "later", "category": "time", "frequency": 0.7},
            
            # Numbers (0-10)
            {"class_id": 31, "word": "zero", "category": "number", "frequency": 0.6},
            {"class_id": 32, "word": "one", "category": "number", "frequency": 0.8},
            {"class_id": 33, "word": "two", "category": "number", "frequency": 0.8},
            {"class_id": 34, "word": "three", "category": "number", "frequency": 0.8},
            {"class_id": 35, "word": "four", "category": "number", "frequency": 0.7},
            {"class_id": 36, "word": "five", "category": "number", "frequency": 0.7},
            {"class_id": 37, "word": "six", "category": "number", "frequency": 0.7},
            {"class_id": 38, "word": "seven", "category": "number", "frequency": 0.7},
            {"class_id": 39, "word": "eight", "category": "number", "frequency": 0.7},
            {"class_id": 40, "word": "nine", "category": "number", "frequency": 0.7},
            {"class_id": 41, "word": "ten", "category": "number", "frequency": 0.7},
            
            # Family
            {"class_id": 42, "word": "mother", "category": "family", "frequency": 0.7},
            {"class_id": 43, "word": "father", "category": "family", "frequency": 0.7},
            {"class_id": 44, "word": "sister", "category": "family", "frequency": 0.6},
            {"class_id": 45, "word": "brother", "category": "family", "frequency": 0.6},
            {"class_id": 46, "word": "family", "category": "family", "frequency": 0.7},
            
            # Colors
            {"class_id": 47, "word": "red", "category": "color", "frequency": 0.6},
            {"class_id": 48, "word": "blue", "category": "color", "frequency": 0.6},
            {"class_id": 49, "word": "green", "category": "color", "frequency": 0.6},
            {"class_id": 50, "word": "yellow", "category": "color", "frequency": 0.6},
            {"class_id": 51, "word": "black", "category": "color", "frequency": 0.6},
            {"class_id": 52, "word": "white", "category": "color", "frequency": 0.6},
        ]
        
        for word_data in default_words:
            word = ASLWord(
                class_id=word_data["class_id"],
                word=word_data["word"],
                category=word_data["category"],
                description=f"Common ASL sign for '{word_data['word']}'",
                difficulty="easy",
                regional_variants=[],
                frequency=word_data["frequency"],
                confidence_threshold=0.7
            )
            
            self.words_by_id[word.class_id] = word
            self.words_by_text[word.word.lower()] = word
            self.categories.add(word.category)
        
        self.total_words = len(self.words_by_id)
        logger.info(f"Created default vocabulary with {self.total_words} words")
    
    def get_word(self, class_id: int) -> Optional[Dict[str, Any]]:
        """
        Get word information by class ID.
        
        Args:
            class_id: Model output class ID
            
        Returns:
            Dictionary with word information or None if not found
        """
        word = self.words_by_id.get(class_id)
        if word:
            return {
                "class_id": word.class_id,
                "word": word.word,
                "category": word.category,
                "description": word.description,
                "difficulty": word.difficulty,
                "regional_variants": word.regional_variants,
                "frequency": word.frequency,
                "confidence_threshold": word.confidence_threshold
            }
        return None
    
    def search_word(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Search for a word by text.
        
        Args:
            text: Word text to search for
            
        Returns:
            Dictionary with word information or None if not found
        """
        word = self.words_by_text.get(text.lower())
        if word:
            return self.get_word(word.class_id)
        return None
    
    def get_words_by_category(self, category: str) -> List[Dict[str, Any]]:
        """
        Get all words in a specific category.
        
        Args:
            category: Category name
            
        Returns:
            List of word dictionaries
        """
        words = []
        for word in self.words_by_id.values():
            if word.category == category:
                words.append(self.get_word(word.class_id))
        return words
    
    def get_categories(self) -> List[str]:
        """Get list of all categories."""
        return sorted(list(self.categories))
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """Get vocabulary statistics."""
        category_counts = {}
        for word in self.words_by_id.values():
            category_counts[word.category] = category_counts.get(word.category, 0) + 1
        
        return {
            "total_words": self.total_words,
            "total_categories": len(self.categories),
            "categories": list(self.categories),
            "category_counts": category_counts,
            "most_common_category": max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None
        }
    
    def add_word(self, class_id: int, word: str, category: str = "general", 
                 description: str = "", difficulty: str = "medium") -> bool:
        """
        Add a new word to the vocabulary.
        
        Args:
            class_id: Unique class ID for the word
            word: Word text
            category: Word category
            description: Word description
            difficulty: Difficulty level
            
        Returns:
            True if word added successfully
        """
        if class_id in self.words_by_id:
            logger.warning(f"Word with class_id {class_id} already exists")
            return False
        
        new_word = ASLWord(
            class_id=class_id,
            word=word,
            category=category,
            description=description,
            difficulty=difficulty,
            regional_variants=[],
            frequency=0.5,
            confidence_threshold=0.7
        )
        
        self.words_by_id[class_id] = new_word
        self.words_by_text[word.lower()] = new_word
        self.categories.add(category)
        self.total_words += 1
        
        logger.info(f"Added new word: {word} (ID: {class_id})")
        return True
