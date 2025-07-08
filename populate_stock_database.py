#!/usr/bin/env python3
"""
Populate Stock Database
Adds comprehensive stock list to database for training
"""

import logging
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_comprehensive_stock_data():
    """Get comprehensive stock data for database population"""
    return [
        # Large Cap Tech
        {'symbol': 'AAPL', 'name': 'Apple Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Consumer Electronics'},
        {'symbol': 'MSFT', 'name': 'Microsoft Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Software'},
        {'symbol': 'GOOGL', 'name': 'Alphabet Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Internet Services'},
        {'symbol': 'AMZN', 'name': 'Amazon.com Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Discretionary', 'industry': 'E-commerce'},
        {'symbol': 'NVDA', 'name': 'NVIDIA Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Semiconductors'},
        {'symbol': 'META', 'name': 'Meta Platforms Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Social Media'},
        {'symbol': 'TSLA', 'name': 'Tesla Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Discretionary', 'industry': 'Electric Vehicles'},
        {'symbol': 'NFLX', 'name': 'Netflix Inc.', 'exchange': 'NASDAQ', 'sector': 'Communication Services', 'industry': 'Streaming'},
        {'symbol': 'ADBE', 'name': 'Adobe Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Software'},
        {'symbol': 'CRM', 'name': 'Salesforce Inc.', 'exchange': 'NYSE', 'sector': 'Technology', 'industry': 'Software'},
        
        # Financial Services
        {'symbol': 'JPM', 'name': 'JPMorgan Chase & Co.', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Banking'},
        {'symbol': 'BAC', 'name': 'Bank of America Corp.', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Banking'},
        {'symbol': 'WFC', 'name': 'Wells Fargo & Company', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Banking'},
        {'symbol': 'GS', 'name': 'Goldman Sachs Group Inc.', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Investment Banking'},
        {'symbol': 'MS', 'name': 'Morgan Stanley', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Investment Banking'},
        {'symbol': 'C', 'name': 'Citigroup Inc.', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Banking'},
        {'symbol': 'AXP', 'name': 'American Express Company', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Credit Services'},
        {'symbol': 'BLK', 'name': 'BlackRock Inc.', 'exchange': 'NYSE', 'sector': 'Financial Services', 'industry': 'Asset Management'},
        
        # Healthcare
        {'symbol': 'JNJ', 'name': 'Johnson & Johnson', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Pharmaceuticals'},
        {'symbol': 'PFE', 'name': 'Pfizer Inc.', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Pharmaceuticals'},
        {'symbol': 'UNH', 'name': 'UnitedHealth Group Inc.', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Health Insurance'},
        {'symbol': 'ABBV', 'name': 'AbbVie Inc.', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Pharmaceuticals'},
        {'symbol': 'MRK', 'name': 'Merck & Co. Inc.', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Pharmaceuticals'},
        {'symbol': 'TMO', 'name': 'Thermo Fisher Scientific Inc.', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Life Sciences'},
        {'symbol': 'ABT', 'name': 'Abbott Laboratories', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Medical Devices'},
        {'symbol': 'DHR', 'name': 'Danaher Corporation', 'exchange': 'NYSE', 'sector': 'Healthcare', 'industry': 'Life Sciences'},
        
        # Consumer & Retail
        {'symbol': 'WMT', 'name': 'Walmart Inc.', 'exchange': 'NYSE', 'sector': 'Consumer Staples', 'industry': 'Retail'},
        {'symbol': 'HD', 'name': 'Home Depot Inc.', 'exchange': 'NYSE', 'sector': 'Consumer Discretionary', 'industry': 'Home Improvement'},
        {'symbol': 'PG', 'name': 'Procter & Gamble Company', 'exchange': 'NYSE', 'sector': 'Consumer Staples', 'industry': 'Personal Care'},
        {'symbol': 'KO', 'name': 'Coca-Cola Company', 'exchange': 'NYSE', 'sector': 'Consumer Staples', 'industry': 'Beverages'},
        {'symbol': 'PEP', 'name': 'PepsiCo Inc.', 'exchange': 'NASDAQ', 'sector': 'Consumer Staples', 'industry': 'Beverages'},
        {'symbol': 'COST', 'name': 'Costco Wholesale Corporation', 'exchange': 'NASDAQ', 'sector': 'Consumer Staples', 'industry': 'Retail'},
        {'symbol': 'NKE', 'name': 'Nike Inc.', 'exchange': 'NYSE', 'sector': 'Consumer Discretionary', 'industry': 'Apparel'},
        {'symbol': 'MCD', 'name': 'McDonald\'s Corporation', 'exchange': 'NYSE', 'sector': 'Consumer Discretionary', 'industry': 'Restaurants'},
        
        # Industrial & Energy
        {'symbol': 'CAT', 'name': 'Caterpillar Inc.', 'exchange': 'NYSE', 'sector': 'Industrials', 'industry': 'Machinery'},
        {'symbol': 'BA', 'name': 'Boeing Company', 'exchange': 'NYSE', 'sector': 'Industrials', 'industry': 'Aerospace'},
        {'symbol': 'GE', 'name': 'General Electric Company', 'exchange': 'NYSE', 'sector': 'Industrials', 'industry': 'Conglomerate'},
        {'symbol': 'MMM', 'name': '3M Company', 'exchange': 'NYSE', 'sector': 'Industrials', 'industry': 'Diversified'},
        {'symbol': 'HON', 'name': 'Honeywell International Inc.', 'exchange': 'NASDAQ', 'sector': 'Industrials', 'industry': 'Aerospace'},
        {'symbol': 'XOM', 'name': 'Exxon Mobil Corporation', 'exchange': 'NYSE', 'sector': 'Energy', 'industry': 'Oil & Gas'},
        {'symbol': 'CVX', 'name': 'Chevron Corporation', 'exchange': 'NYSE', 'sector': 'Energy', 'industry': 'Oil & Gas'},
        {'symbol': 'COP', 'name': 'ConocoPhillips', 'exchange': 'NYSE', 'sector': 'Energy', 'industry': 'Oil & Gas'},
        
        # Growth Stocks
        {'symbol': 'SNOW', 'name': 'Snowflake Inc.', 'exchange': 'NYSE', 'sector': 'Technology', 'industry': 'Cloud Computing'},
        {'symbol': 'PLTR', 'name': 'Palantir Technologies Inc.', 'exchange': 'NYSE', 'sector': 'Technology', 'industry': 'Data Analytics'},
        {'symbol': 'CRWD', 'name': 'CrowdStrike Holdings Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Cybersecurity'},
        {'symbol': 'ZS', 'name': 'Zscaler Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Cybersecurity'},
        {'symbol': 'DDOG', 'name': 'Datadog Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Software'},
        {'symbol': 'NET', 'name': 'Cloudflare Inc.', 'exchange': 'NYSE', 'sector': 'Technology', 'industry': 'Cloud Services'},
        {'symbol': 'ROKU', 'name': 'Roku Inc.', 'exchange': 'NASDAQ', 'sector': 'Communication Services', 'industry': 'Streaming'},
        {'symbol': 'UBER', 'name': 'Uber Technologies Inc.', 'exchange': 'NYSE', 'sector': 'Technology', 'industry': 'Transportation'},
        
        # Semiconductors
        {'symbol': 'INTC', 'name': 'Intel Corporation', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Semiconductors'},
        {'symbol': 'AMD', 'name': 'Advanced Micro Devices Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Semiconductors'},
        {'symbol': 'QCOM', 'name': 'QUALCOMM Incorporated', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Semiconductors'},
        {'symbol': 'AVGO', 'name': 'Broadcom Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Semiconductors'},
        {'symbol': 'TXN', 'name': 'Texas Instruments Incorporated', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Semiconductors'},
        {'symbol': 'MU', 'name': 'Micron Technology Inc.', 'exchange': 'NASDAQ', 'sector': 'Technology', 'industry': 'Memory'},
        
        # Utilities & REITs
        {'symbol': 'NEE', 'name': 'NextEra Energy Inc.', 'exchange': 'NYSE', 'sector': 'Utilities', 'industry': 'Electric Utilities'},
        {'symbol': 'DUK', 'name': 'Duke Energy Corporation', 'exchange': 'NYSE', 'sector': 'Utilities', 'industry': 'Electric Utilities'},
        {'symbol': 'SO', 'name': 'Southern Company', 'exchange': 'NYSE', 'sector': 'Utilities', 'industry': 'Electric Utilities'},
        {'symbol': 'AMT', 'name': 'American Tower Corporation', 'exchange': 'NYSE', 'sector': 'Real Estate', 'industry': 'REITs'},
        {'symbol': 'PLD', 'name': 'Prologis Inc.', 'exchange': 'NYSE', 'sector': 'Real Estate', 'industry': 'REITs'},
        {'symbol': 'CCI', 'name': 'Crown Castle International Corp.', 'exchange': 'NYSE', 'sector': 'Real Estate', 'industry': 'REITs'},
        
        # Communication Services
        {'symbol': 'VZ', 'name': 'Verizon Communications Inc.', 'exchange': 'NYSE', 'sector': 'Communication Services', 'industry': 'Telecom'},
        {'symbol': 'T', 'name': 'AT&T Inc.', 'exchange': 'NYSE', 'sector': 'Communication Services', 'industry': 'Telecom'},
        {'symbol': 'DIS', 'name': 'Walt Disney Company', 'exchange': 'NYSE', 'sector': 'Communication Services', 'industry': 'Entertainment'},
        {'symbol': 'CMCSA', 'name': 'Comcast Corporation', 'exchange': 'NASDAQ', 'sector': 'Communication Services', 'industry': 'Media'},
        
        # ETFs
        {'symbol': 'SPY', 'name': 'SPDR S&P 500 ETF Trust', 'exchange': 'NYSE', 'sector': 'ETF', 'industry': 'Index Fund'},
        {'symbol': 'QQQ', 'name': 'Invesco QQQ Trust', 'exchange': 'NASDAQ', 'sector': 'ETF', 'industry': 'Index Fund'},
        {'symbol': 'IWM', 'name': 'iShares Russell 2000 ETF', 'exchange': 'NYSE', 'sector': 'ETF', 'industry': 'Index Fund'},
        {'symbol': 'GLD', 'name': 'SPDR Gold Shares', 'exchange': 'NYSE', 'sector': 'ETF', 'industry': 'Commodity'},
        {'symbol': 'SLV', 'name': 'iShares Silver Trust', 'exchange': 'NYSE', 'sector': 'ETF', 'industry': 'Commodity'},
        {'symbol': 'TLT', 'name': 'iShares 20+ Year Treasury Bond ETF', 'exchange': 'NASDAQ', 'sector': 'ETF', 'industry': 'Bond Fund'},
    ]

def populate_database():
    """Populate database with comprehensive stock data"""
    with app.app_context():
        stock_data = get_comprehensive_stock_data()
        
        added_count = 0
        updated_count = 0
        
        for stock_info in stock_data:
            try:
                # Check if stock already exists
                existing_stock = Stock.query.filter_by(symbol=stock_info['symbol']).first()
                
                if existing_stock:
                    # Update existing stock
                    for key, value in stock_info.items():
                        if hasattr(existing_stock, key):
                            setattr(existing_stock, key, value)
                    updated_count += 1
                    logger.info(f"Updated {stock_info['symbol']}")
                else:
                    # Create new stock
                    new_stock = Stock(
                        symbol=stock_info['symbol'],
                        name=stock_info['name'],
                        exchange=stock_info['exchange'],
                        sector=stock_info['sector'],
                        industry=stock_info['industry'],
                        is_active=True
                    )
                    db.session.add(new_stock)
                    added_count += 1
                    logger.info(f"Added {stock_info['symbol']}")
                
            except Exception as e:
                logger.error(f"Error processing {stock_info['symbol']}: {e}")
                continue
        
        try:
            db.session.commit()
            logger.info(f"Database population complete!")
            logger.info(f"Added: {added_count} stocks")
            logger.info(f"Updated: {updated_count} stocks")
            logger.info(f"Total: {added_count + updated_count} stocks processed")
            
        except Exception as e:
            logger.error(f"Error committing to database: {e}")
            db.session.rollback()

if __name__ == "__main__":
    logger.info("Starting database population...")
    populate_database()
    logger.info("Database population completed!")
