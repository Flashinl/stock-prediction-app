#!/usr/bin/env python3
"""
Mega ASL Dataset Creator
Creates 1000 classes with 25 samples each with detailed progress updates.
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
import argparse
import time
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MegaASLDatasetCreator:
    """Create a mega ASL dataset with 1000 classes."""
    
    def __init__(self, data_dir: str = "data/mega_asl"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Create 1000 comprehensive ASL words
        self.vocabulary = self.create_1000_vocabulary()
        self.num_classes = len(self.vocabulary)
        
        print(f"🎯 Created vocabulary with {self.num_classes} words")
    
    def create_1000_vocabulary(self):
        """Create exactly 1000 ASL words."""
        
        # Base categories
        basic = ["hello", "goodbye", "please", "thank_you", "sorry", "yes", "no", "help", "love", "family"]
        
        # Numbers 0-99
        numbers = [f"number_{i}" for i in range(100)]
        
        # Colors and variations
        colors = ["red", "blue", "green", "yellow", "orange", "purple", "pink", "brown", "black", "white", 
                 "gray", "silver", "gold", "dark", "light", "bright", "pale", "deep", "vivid", "dull"]
        
        # Animals (100 animals)
        animals = [
            "dog", "cat", "bird", "fish", "horse", "cow", "pig", "sheep", "goat", "chicken",
            "duck", "rabbit", "mouse", "elephant", "lion", "tiger", "bear", "wolf", "fox", "deer",
            "monkey", "snake", "frog", "turtle", "butterfly", "bee", "spider", "ant", "fly", "mosquito",
            "eagle", "hawk", "owl", "parrot", "penguin", "dolphin", "whale", "shark", "octopus", "crab",
            "lobster", "shrimp", "salmon", "tuna", "bass", "trout", "catfish", "goldfish", "hamster", "guinea_pig",
            "ferret", "chinchilla", "hedgehog", "squirrel", "chipmunk", "raccoon", "skunk", "opossum", "beaver", "otter",
            "seal", "walrus", "polar_bear", "panda", "koala", "kangaroo", "wallaby", "platypus", "echidna", "sloth",
            "armadillo", "anteater", "pangolin", "aardvark", "meerkat", "mongoose", "hyena", "jackal", "coyote", "dingo",
            "lynx", "bobcat", "cheetah", "leopard", "jaguar", "puma", "cougar", "ocelot", "serval", "caracal",
            "zebra", "giraffe", "hippo", "rhino", "buffalo", "bison", "yak", "llama", "alpaca", "camel"
        ]
        
        # Food items (150 foods)
        foods = [
            "apple", "banana", "orange", "grape", "strawberry", "blueberry", "raspberry", "blackberry", "cherry", "peach",
            "pear", "plum", "apricot", "mango", "pineapple", "coconut", "lemon", "lime", "grapefruit", "watermelon",
            "cantaloupe", "honeydew", "kiwi", "papaya", "passion_fruit", "dragon_fruit", "star_fruit", "lychee", "rambutan", "durian",
            "bread", "rice", "pasta", "noodles", "cereal", "oatmeal", "quinoa", "barley", "wheat", "corn",
            "potato", "sweet_potato", "carrot", "onion", "garlic", "ginger", "tomato", "cucumber", "lettuce", "spinach",
            "broccoli", "cauliflower", "cabbage", "kale", "brussels_sprouts", "asparagus", "celery", "bell_pepper", "chili", "eggplant",
            "zucchini", "squash", "pumpkin", "beet", "radish", "turnip", "parsnip", "leek", "scallion", "chive",
            "beef", "pork", "chicken", "turkey", "duck", "lamb", "veal", "venison", "rabbit", "fish",
            "salmon", "tuna", "cod", "halibut", "sole", "trout", "bass", "mackerel", "sardine", "anchovy",
            "shrimp", "crab", "lobster", "scallop", "oyster", "mussel", "clam", "squid", "octopus", "eel",
            "milk", "cheese", "butter", "yogurt", "cream", "ice_cream", "egg", "honey", "sugar", "salt",
            "pepper", "cinnamon", "vanilla", "chocolate", "coffee", "tea", "juice", "water", "soda", "wine",
            "beer", "whiskey", "vodka", "rum", "gin", "tequila", "brandy", "champagne", "cocktail", "smoothie",
            "soup", "stew", "curry", "chili", "salad", "sandwich", "burger", "pizza", "taco", "burrito",
            "sushi", "ramen", "pho", "pad_thai", "fried_rice", "stir_fry", "barbecue", "roast", "grill", "bake"
        ]
        
        # Body parts (50)
        body = [
            "head", "face", "eye", "nose", "mouth", "ear", "hair", "neck", "shoulder", "arm",
            "elbow", "wrist", "hand", "finger", "thumb", "nail", "chest", "breast", "back", "spine",
            "waist", "hip", "leg", "thigh", "knee", "calf", "ankle", "foot", "toe", "heel",
            "skin", "muscle", "bone", "joint", "heart", "lung", "liver", "kidney", "stomach", "brain",
            "blood", "nerve", "vein", "artery", "tooth", "tongue", "lip", "cheek", "chin", "forehead"
        ]
        
        # Actions (100)
        actions = [
            "walk", "run", "jump", "hop", "skip", "dance", "swim", "dive", "climb", "crawl",
            "sit", "stand", "lie", "sleep", "wake", "eat", "drink", "chew", "swallow", "taste",
            "smell", "see", "look", "watch", "stare", "glance", "peek", "wink", "blink", "cry",
            "laugh", "smile", "frown", "grin", "giggle", "chuckle", "sob", "weep", "sigh", "yawn",
            "speak", "talk", "whisper", "shout", "scream", "sing", "hum", "whistle", "cough", "sneeze",
            "breathe", "inhale", "exhale", "pant", "gasp", "hiccup", "burp", "snore", "grunt", "moan",
            "push", "pull", "lift", "carry", "hold", "grab", "catch", "throw", "toss", "drop",
            "pick", "place", "put", "set", "lay", "lean", "bend", "stretch", "reach", "touch",
            "feel", "squeeze", "pinch", "poke", "pat", "rub", "scratch", "tickle", "massage", "hug",
            "kiss", "shake", "wave", "point", "clap", "snap", "knock", "tap", "hit", "slap"
        ]
        
        # Objects and tools (100)
        objects = [
            "chair", "table", "desk", "bed", "sofa", "couch", "bench", "stool", "shelf", "cabinet",
            "drawer", "closet", "wardrobe", "mirror", "picture", "painting", "frame", "clock", "lamp", "light",
            "candle", "torch", "flashlight", "bulb", "switch", "outlet", "plug", "cord", "wire", "cable",
            "phone", "computer", "laptop", "tablet", "keyboard", "mouse", "screen", "monitor", "speaker", "headphone",
            "camera", "video", "television", "radio", "stereo", "microphone", "recorder", "player", "remote", "battery",
            "charger", "adapter", "converter", "transformer", "generator", "motor", "engine", "machine", "robot", "tool",
            "hammer", "screwdriver", "wrench", "pliers", "saw", "drill", "nail", "screw", "bolt", "nut",
            "knife", "fork", "spoon", "plate", "bowl", "cup", "glass", "bottle", "jar", "can",
            "box", "bag", "sack", "basket", "bucket", "barrel", "tank", "container", "package", "wrapper",
            "book", "magazine", "newspaper", "journal", "diary", "notebook", "paper", "pen", "pencil", "eraser"
        ]
        
        # Places (50)
        places = [
            "home", "house", "apartment", "room", "kitchen", "bathroom", "bedroom", "living_room", "dining_room", "garage",
            "basement", "attic", "balcony", "porch", "yard", "garden", "park", "playground", "field", "forest",
            "mountain", "hill", "valley", "desert", "beach", "ocean", "sea", "lake", "river", "stream",
            "city", "town", "village", "neighborhood", "street", "road", "highway", "bridge", "tunnel", "building",
            "school", "university", "library", "hospital", "clinic", "store", "shop", "market", "restaurant", "cafe"
        ]
        
        # Weather and nature (50)
        weather = [
            "sun", "moon", "star", "planet", "sky", "cloud", "rain", "snow", "ice", "frost",
            "wind", "breeze", "storm", "thunder", "lightning", "rainbow", "fog", "mist", "dew", "hail",
            "tornado", "hurricane", "cyclone", "blizzard", "drought", "flood", "earthquake", "volcano", "fire", "smoke",
            "tree", "branch", "leaf", "flower", "grass", "bush", "shrub", "vine", "moss", "fern",
            "rock", "stone", "pebble", "sand", "dirt", "soil", "mud", "clay", "mineral", "crystal"
        ]
        
        # Technology (50)
        technology = [
            "internet", "website", "email", "software", "hardware", "program", "app", "application", "system", "network",
            "server", "database", "file", "folder", "document", "data", "information", "code", "algorithm", "function",
            "variable", "array", "string", "number", "boolean", "object", "class", "method", "property", "attribute",
            "interface", "protocol", "standard", "format", "version", "update", "upgrade", "download", "upload", "backup",
            "security", "password", "encryption", "firewall", "virus", "malware", "spam", "phishing", "hacking", "debugging"
        ]
        
        # Emotions and abstract (50)
        emotions = [
            "happy", "sad", "angry", "excited", "calm", "nervous", "worried", "scared", "brave", "confident",
            "proud", "ashamed", "guilty", "innocent", "honest", "dishonest", "kind", "mean", "nice", "rude",
            "polite", "impolite", "patient", "impatient", "generous", "selfish", "humble", "arrogant", "wise", "foolish",
            "smart", "stupid", "clever", "dull", "creative", "boring", "interesting", "funny", "serious", "playful",
            "lazy", "active", "energetic", "tired", "sleepy", "awake", "alert", "focused", "distracted", "confused"
        ]
        
        # Transportation (50)
        transport = [
            "car", "truck", "van", "bus", "taxi", "limousine", "motorcycle", "scooter", "bicycle", "tricycle",
            "train", "subway", "tram", "trolley", "monorail", "plane", "airplane", "jet", "helicopter", "glider",
            "boat", "ship", "yacht", "sailboat", "motorboat", "canoe", "kayak", "raft", "ferry", "cruise",
            "rocket", "spaceship", "satellite", "drone", "balloon", "parachute", "skateboard", "rollerblade", "sled", "sleigh",
            "cart", "wagon", "trailer", "ambulance", "fire_truck", "police_car", "garbage_truck", "delivery_truck", "moving_van", "tow_truck"
        ]
        
        # Combine all categories to reach exactly 1000
        all_words = (
            basic + numbers + colors + animals + foods + body + actions + 
            objects + places + weather + technology + emotions + transport
        )
        
        # Ensure exactly 1000 words
        if len(all_words) > 1000:
            all_words = all_words[:1000]
        elif len(all_words) < 1000:
            # Add more generic words to reach 1000
            remaining = 1000 - len(all_words)
            extra_words = [f"word_{i:03d}" for i in range(remaining)]
            all_words.extend(extra_words)
        
        return all_words
    
    def create_synthetic_video(self, word: str, class_id: int, sample_id: int):
        """Create a single synthetic ASL video."""
        sequence_length = 30
        height, width = 224, 224
        
        # Create video sequence
        video = np.zeros((sequence_length, height, width, 3), dtype=np.float32)
        
        # Word-specific parameters
        np.random.seed(hash(word + str(sample_id)) % 2**32)
        
        center_x = 112 + int(30 * np.cos(class_id * 0.1))
        center_y = 112 + int(30 * np.sin(class_id * 0.1))
        movement_type = class_id % 5
        amplitude = 15 + (class_id % 25)
        
        for frame_idx in range(sequence_length):
            progress = frame_idx / sequence_length
            
            # Create base frame
            frame = np.random.normal(0.3, 0.1, (height, width, 3)).astype(np.float32)
            frame = np.clip(frame, 0, 1)
            
            # Calculate hand positions based on movement type
            if movement_type == 0:  # Circular
                angle = progress * 2 * np.pi
                hand_x = center_x + int(amplitude * np.cos(angle))
                hand_y = center_y + int(amplitude * np.sin(angle))
            elif movement_type == 1:  # Linear
                hand_x = center_x + int(amplitude * np.sin(progress * np.pi))
                hand_y = center_y
            elif movement_type == 2:  # Up-down
                hand_x = center_x
                hand_y = center_y + int(amplitude * np.sin(progress * 2 * np.pi))
            elif movement_type == 3:  # Figure-8
                t = progress * 2 * np.pi
                hand_x = center_x + int(amplitude * np.sin(t))
                hand_y = center_y + int(amplitude * 0.5 * np.sin(2 * t))
            else:  # Complex
                t = progress * 2 * np.pi
                hand_x = center_x + int(amplitude * (np.cos(t) + 0.3 * np.cos(3*t)))
                hand_y = center_y + int(amplitude * (np.sin(t) + 0.3 * np.sin(3*t)))
            
            # Draw person
            frame_uint8 = (frame * 255).astype(np.uint8)
            
            # Head
            head_pos = (center_x, center_y - 60)
            if 0 <= head_pos[0] < width and 0 <= head_pos[1] < height:
                cv2.circle(frame_uint8, head_pos, 20, (200, 180, 160), -1)
            
            # Body
            body_start = (center_x, center_y - 40)
            body_end = (center_x, center_y + 80)
            cv2.line(frame_uint8, body_start, body_end, (100, 150, 120), 8)
            
            # Arms and hands
            shoulder_y = center_y - 20
            if (0 <= hand_x < width and 0 <= hand_y < height):
                cv2.line(frame_uint8, (center_x + 15, shoulder_y), (hand_x, hand_y), (200, 180, 160), 6)
                cv2.circle(frame_uint8, (hand_x, hand_y), 10, (200, 180, 160), -1)
            
            # Left hand (static)
            left_hand = (center_x - 60, center_y + 20)
            if (0 <= left_hand[0] < width and 0 <= left_hand[1] < height):
                cv2.line(frame_uint8, (center_x - 15, shoulder_y), left_hand, (200, 180, 160), 6)
                cv2.circle(frame_uint8, left_hand, 10, (200, 180, 160), -1)
            
            # Add variation
            noise = np.random.normal(0, 0.02, frame_uint8.shape).astype(np.int16)
            frame_uint8 = np.clip(frame_uint8.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            video[frame_idx] = frame_uint8.astype(np.float32) / 255.0
        
        return video
    
    def create_dataset(self, samples_per_class: int = 25):
        """Create the mega dataset with detailed progress."""
        print(f"\n🚀 CREATING MEGA ASL DATASET")
        print(f"📊 Classes: {self.num_classes}")
        print(f"📊 Samples per class: {samples_per_class}")
        print(f"📊 Total samples: {self.num_classes * samples_per_class}")
        print(f"📁 Output directory: {self.data_dir}")
        print("="*60)
        
        videos_dir = self.data_dir / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        metadata = []
        start_time = time.time()
        total_samples_created = 0
        
        for class_id, word in enumerate(self.vocabulary):
            class_start_time = time.time()
            class_dir = videos_dir / word
            class_dir.mkdir(exist_ok=True)
            
            print(f"\n📝 CLASS {class_id + 1:4d}/{self.num_classes}: '{word}'")
            
            for sample_id in range(samples_per_class):
                sample_start_time = time.time()
                
                # Create video
                video_data = self.create_synthetic_video(word, class_id, sample_id)
                
                # Save video
                video_filename = f"{word}_{sample_id:03d}.npy"
                video_path = class_dir / video_filename
                np.save(video_path, video_data)
                
                # Add to metadata
                metadata.append({
                    "video_path": str(video_path.relative_to(self.data_dir)),
                    "word": word,
                    "label": class_id,
                    "sample_id": sample_id,
                    "duration": 1.0,
                    "sequence_length": 30
                })
                
                total_samples_created += 1
                sample_time = time.time() - sample_start_time
                
                # Progress for each sample
                progress = total_samples_created / (self.num_classes * samples_per_class)
                elapsed = time.time() - start_time
                eta = (elapsed / progress - elapsed) if progress > 0 else 0
                
                print(f"   ✅ Sample {sample_id + 1:2d}/{samples_per_class} "
                      f"({sample_time:.2f}s) | "
                      f"Total: {total_samples_created:5d}/{self.num_classes * samples_per_class} "
                      f"({progress:.1%}) | "
                      f"ETA: {eta/60:.1f}min")
            
            class_time = time.time() - class_start_time
            print(f"🎯 Completed class '{word}' in {class_time:.1f}s")
        
        # Save metadata files
        print(f"\n💾 Saving metadata files...")
        
        # Vocabulary
        vocab_data = {
            "vocabulary": self.vocabulary,
            "num_classes": self.num_classes,
            "samples_per_class": samples_per_class,
            "total_samples": len(metadata),
            "sequence_length": 30,
            "frame_size": [224, 224, 3],
            "created_at": datetime.now().isoformat()
        }
        
        with open(self.data_dir / "vocabulary.json", 'w') as f:
            json.dump(vocab_data, f, indent=2)
        
        # Metadata
        with open(self.data_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Training config
        config = {
            "model_config": {
                "num_classes": self.num_classes,
                "sequence_length": 30,
                "embed_dim": 512,
                "num_heads": 8,
                "num_transformer_blocks": 6,
                "dropout": 0.1
            },
            "training_config": {
                "epochs": 200,
                "learning_rate": 1e-4,
                "weight_decay": 1e-4,
                "target_accuracy": 0.95,
                "early_stopping_patience": 25
            },
            "data_config": {
                "data_dir": str(self.data_dir),
                "batch_size": 16,
                "validation_split": 0.15,
                "num_workers": 4
            }
        }
        
        with open(self.data_dir / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        total_time = time.time() - start_time
        
        print(f"\n🎉 DATASET CREATION COMPLETE!")
        print(f"⏱️  Total time: {total_time/60:.1f} minutes")
        print(f"📊 Classes: {self.num_classes}")
        print(f"📊 Total samples: {len(metadata)}")
        print(f"📁 Location: {self.data_dir}")
        print("="*60)
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Create Mega ASL Dataset')
    parser.add_argument('--data-dir', type=str, default='data/mega_asl', help='Output directory')
    parser.add_argument('--samples-per-class', type=int, default=25, help='Samples per class')
    
    args = parser.parse_args()
    
    creator = MegaASLDatasetCreator(args.data_dir)
    creator.create_dataset(args.samples_per_class)

if __name__ == "__main__":
    main()
