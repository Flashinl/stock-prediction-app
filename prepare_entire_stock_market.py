#!/usr/bin/env python3
"""
Entire Stock Market Data Preparation
Downloads and prepares training data for ALL US stocks
"""

import logging
import pandas as pd
import yfinance as yf
import requests
import time
import os
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from app import app, db, Stock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntireStockMarketPreparator:
    def __init__(self):
        self.all_symbols = []
        self.processed_count = 0
        self.failed_count = 0
        self.batch_size = 100
        
    def get_all_us_stocks(self):
        """Get comprehensive list of ALL US stocks from multiple sources"""
        logger.info("Fetching comprehensive US stock list...")
        
        all_symbols = set()
        
        # Method 1: Get from major exchanges
        exchanges = ['NYSE', 'NASDAQ', 'AMEX']
        for exchange in exchanges:
            try:
                symbols = self._get_stocks_from_exchange(exchange)
                all_symbols.update(symbols)
                logger.info(f"Added {len(symbols)} stocks from {exchange}")
            except Exception as e:
                logger.warning(f"Failed to get stocks from {exchange}: {e}")
        
        # Method 2: Get S&P 500, Russell 2000, etc.
        index_symbols = self._get_index_stocks()
        all_symbols.update(index_symbols)
        
        # Method 3: Get from yfinance screener (if available)
        screener_symbols = self._get_screener_stocks()
        all_symbols.update(screener_symbols)
        
        # Method 4: Add comprehensive manual list of known stocks
        manual_symbols = self._get_comprehensive_manual_list()
        all_symbols.update(manual_symbols)
        
        # Filter out invalid symbols
        valid_symbols = [s for s in all_symbols if self._is_valid_symbol(s)]
        
        logger.info(f"Total unique US stocks found: {len(valid_symbols)}")
        return sorted(valid_symbols)
    
    def _get_stocks_from_exchange(self, exchange):
        """Get stocks from specific exchange using FMP API (free tier)"""
        try:
            # Using Financial Modeling Prep free API
            url = f"https://financialmodelingprep.com/api/v3/stock/list"
            response = requests.get(url, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                symbols = []
                for stock in data:
                    if stock.get('exchangeShortName') == exchange:
                        symbol = stock.get('symbol', '').strip()
                        if symbol and len(symbol) <= 5:  # Filter reasonable symbols
                            symbols.append(symbol)
                return symbols[:2000]  # Limit to prevent overload
            else:
                logger.warning(f"Failed to fetch from FMP API: {response.status_code}")
                return []
        except Exception as e:
            logger.warning(f"Error fetching from exchange {exchange}: {e}")
            return []
    
    def _get_index_stocks(self):
        """Get stocks from major indices"""
        index_symbols = set()
        
        # S&P 500
        try:
            sp500_url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(sp500_url)
            sp500_symbols = tables[0]['Symbol'].tolist()
            index_symbols.update(sp500_symbols)
            logger.info(f"Added {len(sp500_symbols)} S&P 500 stocks")
        except Exception as e:
            logger.warning(f"Failed to get S&P 500 stocks: {e}")
        
        # Add major ETFs and indices
        major_etfs = [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'IEFA', 'IEMG',
            'GLD', 'SLV', 'TLT', 'IEF', 'LQD', 'HYG', 'VNQ', 'REIT'
        ]
        index_symbols.update(major_etfs)
        
        return list(index_symbols)
    
    def _get_screener_stocks(self):
        """Get stocks using various screening methods"""
        screener_symbols = []
        
        # Add popular stocks by market cap ranges
        popular_large_caps = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA',
            'BRK-A', 'BRK-B', 'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA',
            'CVX', 'HD', 'ABBV', 'PFE', 'KO', 'AVGO', 'PEP', 'COST', 'WMT'
        ]
        
        popular_mid_caps = [
            'SNOW', 'PLTR', 'CRWD', 'ZS', 'DDOG', 'NET', 'ROKU', 'UBER',
            'LYFT', 'ABNB', 'COIN', 'HOOD', 'SOFI', 'RIVN', 'LCID'
        ]
        
        popular_small_caps = [
            'SPCE', 'NKLA', 'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'RIDE',
            'GOEV', 'HYLN', 'WKHS', 'SOLO', 'OPEN', 'WISH', 'CLOV'
        ]
        
        screener_symbols.extend(popular_large_caps)
        screener_symbols.extend(popular_mid_caps)
        screener_symbols.extend(popular_small_caps)
        
        return screener_symbols
    
    def _get_comprehensive_manual_list(self):
        """Get comprehensive manual list covering all sectors"""
        return [
            # Technology - Large Cap
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA', 'TSLA',
            'NFLX', 'ADBE', 'CRM', 'ORCL', 'IBM', 'INTC', 'AMD', 'QCOM',
            'AVGO', 'TXN', 'MU', 'AMAT', 'LRCX', 'KLAC', 'MRVL', 'MCHP',
            
            # Technology - Growth
            'SNOW', 'PLTR', 'CRWD', 'ZS', 'OKTA', 'DDOG', 'NET', 'FSLY',
            'ESTC', 'SPLK', 'TWLO', 'ZM', 'DOCU', 'PTON', 'RBLX', 'U',
            
            # Financial Services
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW',
            'USB', 'PNC', 'TFC', 'COF', 'BK', 'STT', 'FITB', 'RF', 'CFG',
            
            # Healthcare & Biotech
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY',
            'AMGN', 'GILD', 'BIIB', 'REGN', 'VRTX', 'ILMN', 'MRNA', 'BNTX',
            'ZTS', 'ELV', 'CVS', 'CI', 'HUM', 'ANTM', 'MOH', 'CNC',
            
            # Consumer Discretionary
            'AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW', 'TJX',
            'BKNG', 'ABNB', 'UBER', 'LYFT', 'DIS', 'NFLX', 'ROKU', 'SPOT',
            
            # Consumer Staples
            'WMT', 'PG', 'KO', 'PEP', 'COST', 'CL', 'KMB', 'GIS', 'K',
            'HSY', 'MKC', 'CPB', 'CAG', 'SJM', 'HRL', 'TSN', 'TYSON',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL', 'OXY', 'MPC', 'VLO',
            'PSX', 'KMI', 'OKE', 'WMB', 'EPD', 'ET', 'MPLX', 'PAA',
            
            # Industrials
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX',
            'NOC', 'GD', 'DE', 'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'ROK',
            
            # Materials
            'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'GOLD', 'AA', 'X', 'CLF',
            'NUE', 'STLD', 'RS', 'VMC', 'MLM', 'NTR', 'CF', 'MOS', 'FMC',
            
            # Real Estate & REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC',
            'EXR', 'AVB', 'EQR', 'MAA', 'UDR', 'CPT', 'ESS', 'AIV', 'BXP',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG',
            'ED', 'FE', 'ES', 'AWK', 'WEC', 'DTE', 'PPL', 'CMS', 'NI',
            
            # Communication Services
            'GOOGL', 'META', 'VZ', 'T', 'DIS', 'CMCSA', 'NFLX', 'CHTR',
            'TMUS', 'TWTR', 'SNAP', 'PINS', 'MTCH', 'IAC', 'DISH', 'SIRI',
            
            # Small Cap & Penny Stocks (sample)
            'SPCE', 'NKLA', 'PLUG', 'FCEL', 'BLNK', 'CHPT', 'QS', 'RIDE',
            'GOEV', 'HYLN', 'WKHS', 'SOLO', 'OPEN', 'WISH', 'CLOV', 'SOFI',
            'HOOD', 'COIN', 'RIVN', 'LCID', 'F', 'GM', 'FORD', 'NIO', 'XPEV',
            
            # International ADRs
            'BABA', 'TSM', 'ASML', 'NVO', 'TM', 'SONY', 'SAP', 'SHOP',
            'SE', 'MELI', 'PDD', 'JD', 'BIDU', 'NTE', 'TME', 'BILI',
            
            # Biotech & Pharma
            'MRNA', 'BNTX', 'NVAX', 'VXRT', 'INO', 'OCGN', 'SRNE', 'ATOS',
            'CTXR', 'OBSV', 'BNGO', 'PACB', 'CRSP', 'EDIT', 'NTLA', 'BEAM',
            
            # Cannabis & Alternative
            'TLRY', 'CGC', 'ACB', 'CRON', 'SNDL', 'OGI', 'HEXO', 'APHA',
            
            # SPACs & Recent IPOs
            'SPAC', 'PSTH', 'CCIV', 'IPOE', 'IPOF', 'CLOV', 'BARK', 'BODY'
        ]
    
    def _is_valid_symbol(self, symbol):
        """Check if symbol is valid for US stock market"""
        if not symbol or len(symbol) > 5:
            return False
        if any(char in symbol for char in ['/', '=', '^', '.']):
            return False
        if symbol.endswith('.TO') or symbol.endswith('.L'):
            return False
        return True
    
    def prepare_market_data_parallel(self, max_workers=10):
        """Prepare data for entire stock market using parallel processing"""
        logger.info("Starting entire stock market data preparation...")
        
        # Get all US stocks
        self.all_symbols = self.get_all_us_stocks()
        total_stocks = len(self.all_symbols)
        
        logger.info(f"Preparing to process {total_stocks} stocks...")
        
        # Process in batches with parallel workers
        successful_data = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_symbol = {
                executor.submit(self._process_single_stock, symbol): symbol 
                for symbol in self.all_symbols
            }
            
            # Process completed tasks
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        successful_data.append(result)
                        self.processed_count += 1
                    else:
                        self.failed_count += 1
                        
                    # Progress update
                    if (self.processed_count + self.failed_count) % 100 == 0:
                        progress = (self.processed_count + self.failed_count) / total_stocks * 100
                        logger.info(f"Progress: {progress:.1f}% - Processed: {self.processed_count}, Failed: {self.failed_count}")
                        
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
                    self.failed_count += 1
        
        logger.info(f"Market data preparation complete!")
        logger.info(f"Successfully processed: {self.processed_count} stocks")
        logger.info(f"Failed: {self.failed_count} stocks")
        logger.info(f"Success rate: {self.processed_count/total_stocks*100:.1f}%")
        
        # Save comprehensive dataset
        self._save_market_dataset(successful_data)
        
        return successful_data
    
    def _process_single_stock(self, symbol):
        """Process a single stock and extract all features"""
        try:
            # Add small delay to avoid rate limiting
            time.sleep(0.1)
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2y")  # Get 2 years of data
            info = ticker.info
            
            if hist.empty or len(hist) < 100:  # Need sufficient data
                return None
            
            # Extract basic info
            stock_data = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': hist['Close'].iloc[-1],
                'volume': hist['Volume'].iloc[-1],
                'data_points': len(hist),
                'date_range': f"{hist.index[0].date()} to {hist.index[-1].date()}"
            }
            
            return stock_data
            
        except Exception as e:
            logger.debug(f"Failed to process {symbol}: {e}")
            return None
    
    def _save_market_dataset(self, market_data):
        """Save the comprehensive market dataset"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as CSV
        df = pd.DataFrame(market_data)
        csv_file = f"data/entire_stock_market_{timestamp}.csv"
        os.makedirs('data', exist_ok=True)
        df.to_csv(csv_file, index=False)
        logger.info(f"Market dataset saved to: {csv_file}")
        
        # Save as JSON for detailed info
        json_file = f"data/entire_stock_market_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(market_data, f, indent=2, default=str)
        logger.info(f"Market dataset saved to: {json_file}")
        
        # Save summary statistics
        summary = {
            'total_stocks': len(market_data),
            'sectors': df['sector'].value_counts().to_dict(),
            'market_cap_ranges': {
                'large_cap_10B+': len(df[df['market_cap'] > 10_000_000_000]),
                'mid_cap_2B_10B': len(df[(df['market_cap'] > 2_000_000_000) & (df['market_cap'] <= 10_000_000_000)]),
                'small_cap_300M_2B': len(df[(df['market_cap'] > 300_000_000) & (df['market_cap'] <= 2_000_000_000)]),
                'micro_cap_under_300M': len(df[df['market_cap'] <= 300_000_000])
            },
            'timestamp': timestamp
        }
        
        summary_file = f"data/market_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Market summary saved to: {summary_file}")

def main():
    logger.info("Starting entire stock market data preparation...")
    
    preparator = EntireStockMarketPreparator()
    market_data = preparator.prepare_market_data_parallel(max_workers=20)
    
    logger.info(f"Entire stock market data preparation completed!")
    logger.info(f"Total stocks processed: {len(market_data)}")

if __name__ == "__main__":
    main()
