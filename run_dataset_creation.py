#!/usr/bin/env python3
"""
Script to run the comprehensive test dataset creation
"""

import sys
import os
import logging
from create_test_dataset import TestDatasetCreator

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_creation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def main():
    """Main function to create the comprehensive test dataset"""
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE STOCK PREDICTION TEST DATASET CREATION")
    logger.info("=" * 60)
    
    try:
        # Create the dataset creator
        creator = TestDatasetCreator()
        
        # Get the stock list first to show what we're working with
        all_symbols = creator.get_comprehensive_stock_list()
        logger.info(f"Total available symbols: {len(all_symbols)}")
        
        # Show breakdown by category
        large_cap_count = len([s for s in all_symbols if s in [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'NFLX'
        ]])
        logger.info(f"Large cap stocks: ~{large_cap_count}")
        
        # Create the comprehensive dataset
        logger.info("Starting dataset creation...")
        dataset = creator.create_comprehensive_dataset(
            max_stocks=300,  # Process 300 stocks for better training data
            save_to_file=True
        )
        
        if dataset:
            logger.info("=" * 60)
            logger.info("DATASET CREATION COMPLETED SUCCESSFULLY!")
            logger.info("=" * 60)
            logger.info(f"✅ Total samples created: {len(dataset)}")
            logger.info(f"❌ Failed symbols: {len(creator.failed_symbols)}")
            
            if dataset:
                sample = dataset[0]
                logger.info(f"📊 Features per sample: {len(sample.keys())}")
                
                # Show feature categories
                technical_features = [k for k in sample.keys() if any(x in k.lower() for x in ['rsi', 'macd', 'sma', 'volume', 'price', 'bb_', 'momentum'])]
                fundamental_features = [k for k in sample.keys() if any(x in k.lower() for x in ['revenue', 'ebitda', 'debt', 'ratio', 'margin', 'growth', 'pe_', 'eps'])]
                
                logger.info(f"🔧 Technical features: {len(technical_features)}")
                logger.info(f"💰 Fundamental features: {len(fundamental_features)}")
                
                # Show label distribution
                labels = [item.get('label', 'Unknown') for item in dataset]
                label_counts = {}
                for label in labels:
                    label_counts[label] = label_counts.get(label, 0) + 1
                
                logger.info("📈 Label distribution:")
                for label, count in sorted(label_counts.items()):
                    percentage = (count / len(dataset)) * 100
                    logger.info(f"   {label}: {count} ({percentage:.1f}%)")
            
            logger.info("=" * 60)
            logger.info("FILES CREATED:")
            logger.info("📁 datasets/comprehensive_stock_dataset.json")
            logger.info("📁 datasets/comprehensive_stock_dataset.csv") 
            logger.info("📁 datasets/dataset_summary.json")
            logger.info("=" * 60)
            
            if creator.failed_symbols:
                logger.warning("Failed symbols:")
                for symbol in creator.failed_symbols[:10]:  # Show first 10
                    logger.warning(f"   ❌ {symbol}")
                if len(creator.failed_symbols) > 10:
                    logger.warning(f"   ... and {len(creator.failed_symbols) - 10} more")
        
        else:
            logger.error("❌ Dataset creation failed - no data collected")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Dataset creation failed with error: {e}")
        logger.error("Check the logs for more details")
        return 1
    
    logger.info("🎉 Dataset creation process completed!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
