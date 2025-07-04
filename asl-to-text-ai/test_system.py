#!/usr/bin/env python3
"""
Simple test script to verify ASL-to-Text AI system components.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all core modules can be imported."""
    print("Testing imports...")
    
    try:
        from utils.config import get_config, validate_config
        print("✓ Config module imported successfully")
        
        from translation.vocabulary import ASLVocabulary
        print("✓ Vocabulary module imported successfully")
        
        from translation.grammar_processor import GrammarProcessor
        print("✓ Grammar processor imported successfully")
        
        from models.asl_detector import ASLDetector
        print("✓ ASL detector imported successfully")
        
        from translation.asl_translator import ASLTranslator
        print("✓ ASL translator imported successfully")
        
        from preprocessing.video_processor import VideoProcessor
        print("✓ Video processor imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_config():
    """Test configuration validation."""
    print("\nTesting configuration...")
    
    try:
        from utils.config import get_config, validate_config
        
        config = get_config()
        print(f"✓ Configuration loaded with {len(config)} sections")
        
        if validate_config():
            print("✓ Configuration validation passed")
            return True
        else:
            print("✗ Configuration validation failed")
            return False
            
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_vocabulary():
    """Test vocabulary system."""
    print("\nTesting vocabulary...")
    
    try:
        from translation.vocabulary import ASLVocabulary
        
        vocab = ASLVocabulary()
        stats = vocab.get_vocabulary_stats()
        
        print(f"✓ Vocabulary loaded with {stats['total_words']} words")
        print(f"✓ Found {stats['total_categories']} categories")
        
        # Test word lookup
        word_info = vocab.get_word(0)  # Should be "hello"
        if word_info and word_info['word'] == 'hello':
            print("✓ Word lookup working correctly")
            return True
        else:
            print("✗ Word lookup failed")
            return False
            
    except Exception as e:
        print(f"✗ Vocabulary test failed: {e}")
        return False

def test_grammar_processor():
    """Test grammar processing."""
    print("\nTesting grammar processor...")
    
    try:
        from translation.grammar_processor import GrammarProcessor
        
        processor = GrammarProcessor()
        
        # Test sentence processing
        test_words = ["I", "love", "you"]
        result = processor.process_sentence(test_words)
        
        if result and len(result) > 0:
            print(f"✓ Grammar processing working: '{' '.join(test_words)}' -> '{result}'")
            return True
        else:
            print("✗ Grammar processing failed")
            return False
            
    except Exception as e:
        print(f"✗ Grammar processor test failed: {e}")
        return False

def test_video_processor():
    """Test video processing capabilities."""
    print("\nTesting video processor...")
    
    try:
        from preprocessing.video_processor import VideoProcessor
        import numpy as np
        
        processor = VideoProcessor()
        
        # Create a dummy frame
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test frame processing
        result = processor.process_real_time_frame(dummy_frame)
        
        if result and 'frame' in result:
            print(f"✓ Video processing working (quality: {result['quality_score']:.2f})")
            return True
        else:
            print("✗ Video processing failed")
            return False
            
    except Exception as e:
        print(f"✗ Video processor test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("ASL-to-Text AI System Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config,
        test_vocabulary,
        test_grammar_processor,
        test_video_processor
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        return 0
    else:
        print("❌ Some tests failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
