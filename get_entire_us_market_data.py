#!/usr/bin/env python3
"""
Get Entire US Stock Market Historical Data
Uses reliable free sources for comprehensive market data
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
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntireUSMarketData:
    def __init__(self):
        self.all_symbols = []
        self.valid_symbols = []
        self.market_data = {}
        
    def get_sp500_stocks(self):
        """Get S&P 500 stocks from Wikipedia (most reliable)"""
        try:
            logger.info("Fetching S&P 500 stocks...")
            url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            tables = pd.read_html(url)
            sp500_df = tables[0]
            
            symbols = sp500_df['Symbol'].tolist()
            logger.info(f"Found {len(symbols)} S&P 500 stocks")
            return symbols
        except Exception as e:
            logger.error(f"Failed to get S&P 500 stocks: {e}")
            return []
    
    def get_nasdaq_stocks(self):
        """Get NASDAQ stocks from NASDAQ's official API"""
        try:
            logger.info("Fetching NASDAQ stocks...")
            url = "https://api.nasdaq.com/api/screener/stocks"
            params = {
                'tableonly': 'true',
                'limit': '5000',
                'offset': '0',
                'download': 'true'
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers)
            if response.status_code == 200:
                data = response.json()
                if 'data' in data and 'rows' in data['data']:
                    symbols = [row['symbol'] for row in data['data']['rows']]
                    logger.info(f"Found {len(symbols)} NASDAQ stocks")
                    return symbols
            
            logger.warning("NASDAQ API failed, using fallback list")
            return self._get_nasdaq_fallback()
            
        except Exception as e:
            logger.warning(f"NASDAQ API error: {e}, using fallback")
            return self._get_nasdaq_fallback()
    
    def _get_nasdaq_fallback(self):
        """Fallback NASDAQ stock list"""
        return [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO', 'COST',
            'NFLX', 'ADBE', 'PEP', 'TMUS', 'CSCO', 'CMCSA', 'TXN', 'QCOM', 'AMAT', 'INTU',
            'AMD', 'ISRG', 'BKNG', 'HON', 'AMGN', 'VRTX', 'ADP', 'SBUX', 'GILD', 'MU',
            'INTC', 'ADI', 'LRCX', 'PYPL', 'REGN', 'KLAC', 'PANW', 'SNPS', 'CDNS', 'MRVL',
            'ORLY', 'CRWD', 'FTNT', 'CSX', 'ABNB', 'DXCM', 'ADSK', 'ROP', 'NXPI', 'WDAY',
            'FANG', 'TEAM', 'CTAS', 'CHTR', 'PCAR', 'MNST', 'AEP', 'ROST', 'FAST', 'ODFL',
            'BKR', 'EA', 'VRSK', 'EXC', 'XEL', 'CTSH', 'GEHC', 'KDP', 'LULU', 'CCEP',
            'DDOG', 'ZS', 'SNOW', 'NET', 'ROKU', 'UBER', 'LYFT', 'DOCU', 'ZM', 'PTON'
        ]
    
    def get_nyse_stocks(self):
        """Get NYSE stocks using multiple sources"""
        try:
            logger.info("Fetching NYSE stocks...")
            
            # Try NYSE API first
            url = "https://www.nyse.com/api/quotes/filter"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, list):
                    symbols = [item.get('symbolTicker', '') for item in data if item.get('symbolTicker')]
                    logger.info(f"Found {len(symbols)} NYSE stocks from API")
                    return symbols
            
            logger.warning("NYSE API failed, using comprehensive fallback")
            return self._get_nyse_fallback()
            
        except Exception as e:
            logger.warning(f"NYSE API error: {e}, using fallback")
            return self._get_nyse_fallback()
    
    def _get_nyse_fallback(self):
        """Comprehensive NYSE stock fallback list"""
        return [
            # Large Cap
            'JPM', 'JNJ', 'V', 'PG', 'UNH', 'MA', 'HD', 'CVX', 'ABBV', 'BAC', 'XOM', 'WMT',
            'LLY', 'KO', 'AVGO', 'MRK', 'PFE', 'TMO', 'ABT', 'CRM', 'ACN', 'COST', 'DHR',
            'VZ', 'NKE', 'DIS', 'ADBE', 'MCD', 'BMY', 'PM', 'NFLX', 'T', 'UPS', 'LOW',
            'QCOM', 'HON', 'UNP', 'IBM', 'GS', 'SPGI', 'CAT', 'INTU', 'AXP', 'BKNG', 'DE',
            
            # Mid Cap
            'MMM', 'BA', 'GE', 'WFC', 'MS', 'C', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC', 'COF',
            'BK', 'STT', 'FITB', 'RF', 'CFG', 'KEY', 'ZION', 'CMA', 'HBAN', 'MTB', 'SIVB',
            'WAL', 'PB', 'EWBC', 'PACW', 'SBNY', 'OZK', 'CBSH', 'UBSI', 'FFIN', 'FULT',
            
            # Energy
            'XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'KMI', 'OKE', 'WMB',
            'EPD', 'ET', 'MPLX', 'PAA', 'ENPH', 'SEDG', 'RUN', 'NOVA', 'FSLR', 'SPWR',
            
            # Healthcare
            'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK', 'TMO', 'ABT', 'DHR', 'BMY', 'AMGN', 'GILD',
            'CVS', 'CI', 'HUM', 'ANTM', 'MOH', 'CNC', 'ZTS', 'ELV', 'DXCM', 'ISRG', 'SYK',
            
            # Financial
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'USB', 'PNC', 'TFC',
            'COF', 'BK', 'STT', 'FITB', 'RF', 'CFG', 'KEY', 'ZION', 'CMA', 'HBAN', 'MTB',
            
            # Industrial
            'CAT', 'BA', 'GE', 'MMM', 'HON', 'UPS', 'FDX', 'LMT', 'RTX', 'NOC', 'GD', 'DE',
            'EMR', 'ETN', 'ITW', 'PH', 'CMI', 'ROK', 'DOV', 'FTV', 'IR', 'CARR', 'OTIS',
            
            # Consumer
            'WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'NKE', 'MCD', 'SBUX', 'TGT', 'LOW',
            'TJX', 'DIS', 'F', 'GM', 'FORD', 'NIO', 'XPEV', 'LI', 'RIVN', 'LCID',
            
            # Utilities
            'NEE', 'DUK', 'SO', 'D', 'EXC', 'XEL', 'SRE', 'AEP', 'PCG', 'ED', 'FE', 'ES',
            'AWK', 'WEC', 'DTE', 'PPL', 'CMS', 'NI', 'LNT', 'EVRG', 'ATO', 'CNP', 'NRG',
            
            # REITs
            'AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'WELL', 'DLR', 'O', 'SBAC', 'EXR', 'AVB',
            'EQR', 'MAA', 'UDR', 'CPT', 'ESS', 'AIV', 'BXP', 'VTR', 'PEAK', 'ARE', 'BDN'
        ]
    
    def get_etf_list(self):
        """Get major ETFs"""
        return [
            'SPY', 'QQQ', 'IWM', 'VTI', 'VOO', 'VEA', 'VWO', 'IEFA', 'IEMG', 'AGG',
            'BND', 'VNQ', 'GLD', 'SLV', 'TLT', 'IEF', 'LQD', 'HYG', 'EMB', 'VCIT',
            'VXUS', 'IXUS', 'ITOT', 'IEUR', 'EFA', 'EEM', 'VGK', 'VPL', 'VGT', 'XLK',
            'XLF', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLRE', 'XLB', 'XME'
        ]
    
    def compile_entire_market(self):
        """Compile entire US stock market from all sources"""
        logger.info("Compiling entire US stock market...")
        
        all_symbols = set()
        
        # Get from all sources
        sp500_symbols = self.get_sp500_stocks()
        nasdaq_symbols = self.get_nasdaq_stocks()
        nyse_symbols = self.get_nyse_stocks()
        etf_symbols = self.get_etf_list()
        
        all_symbols.update(sp500_symbols)
        all_symbols.update(nasdaq_symbols)
        all_symbols.update(nyse_symbols)
        all_symbols.update(etf_symbols)
        
        # Clean and filter symbols
        cleaned_symbols = []
        for symbol in all_symbols:
            if symbol and isinstance(symbol, str):
                symbol = symbol.strip().upper()
                if len(symbol) <= 5 and symbol.replace('-', '').replace('.', '').isalnum():
                    cleaned_symbols.append(symbol)
        
        # Remove duplicates and sort
        unique_symbols = sorted(list(set(cleaned_symbols)))
        
        logger.info(f"Compiled {len(unique_symbols)} unique US market symbols")
        logger.info(f"Sources: S&P 500 ({len(sp500_symbols)}), NASDAQ ({len(nasdaq_symbols)}), NYSE ({len(nyse_symbols)}), ETFs ({len(etf_symbols)})")
        
        self.all_symbols = unique_symbols
        return unique_symbols
    
    def download_historical_data(self, symbols, period="2y", max_workers=10):
        """Download historical data for all symbols"""
        logger.info(f"Downloading historical data for {len(symbols)} symbols...")
        
        successful_downloads = {}
        failed_downloads = []
        
        def download_single_stock(symbol):
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period=period)
                info = ticker.info
                
                if len(hist) >= 100:  # Minimum data requirement
                    return symbol, {
                        'symbol': symbol,
                        'history': hist,
                        'info': info,
                        'data_points': len(hist),
                        'date_range': f"{hist.index[0].date()} to {hist.index[-1].date()}",
                        'current_price': hist['Close'].iloc[-1],
                        'volume': hist['Volume'].iloc[-1],
                        'market_cap': info.get('marketCap', 0),
                        'sector': info.get('sector', 'Unknown'),
                        'industry': info.get('industry', 'Unknown')
                    }
                return symbol, None
                
            except Exception as e:
                logger.debug(f"Failed to download {symbol}: {e}")
                return symbol, None
        
        # Download in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(download_single_stock, symbol): symbol for symbol in symbols}
            
            completed = 0
            for future in as_completed(future_to_symbol):
                symbol, data = future.result()
                completed += 1
                
                if data:
                    successful_downloads[symbol] = data
                else:
                    failed_downloads.append(symbol)
                
                if completed % 50 == 0:
                    logger.info(f"Downloaded {completed}/{len(symbols)} - Success: {len(successful_downloads)}, Failed: {len(failed_downloads)}")
        
        logger.info(f"Download complete: {len(successful_downloads)} successful, {len(failed_downloads)} failed")
        
        self.market_data = successful_downloads
        self.valid_symbols = list(successful_downloads.keys())
        
        return successful_downloads
    
    def save_market_data(self, filename_prefix="entire_us_market"):
        """Save the downloaded market data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data directory
        os.makedirs('market_data', exist_ok=True)
        
        # Save symbols list
        symbols_file = f"market_data/{filename_prefix}_symbols_{timestamp}.json"
        with open(symbols_file, 'w') as f:
            json.dump({
                'timestamp': timestamp,
                'total_symbols': len(self.valid_symbols),
                'symbols': self.valid_symbols
            }, f, indent=2)
        
        # Save market summary
        summary = {
            'timestamp': timestamp,
            'total_stocks': len(self.valid_symbols),
            'data_period': '2 years',
            'sectors': {},
            'market_cap_ranges': {
                'large_cap_10B+': 0,
                'mid_cap_2B_10B': 0,
                'small_cap_300M_2B': 0,
                'micro_cap_under_300M': 0,
                'unknown': 0
            },
            'exchanges': {
                'NYSE': 0,
                'NASDAQ': 0,
                'ETF': 0,
                'Other': 0
            }
        }
        
        # Analyze the data
        for symbol, data in self.market_data.items():
            # Sector analysis
            sector = data.get('sector', 'Unknown')
            if sector not in summary['sectors']:
                summary['sectors'][sector] = 0
            summary['sectors'][sector] += 1
            
            # Market cap analysis
            market_cap = data.get('market_cap', 0)
            if market_cap > 10_000_000_000:
                summary['market_cap_ranges']['large_cap_10B+'] += 1
            elif market_cap > 2_000_000_000:
                summary['market_cap_ranges']['mid_cap_2B_10B'] += 1
            elif market_cap > 300_000_000:
                summary['market_cap_ranges']['small_cap_300M_2B'] += 1
            elif market_cap > 0:
                summary['market_cap_ranges']['micro_cap_under_300M'] += 1
            else:
                summary['market_cap_ranges']['unknown'] += 1
        
        summary_file = f"market_data/{filename_prefix}_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Market data saved:")
        logger.info(f"- Symbols: {symbols_file}")
        logger.info(f"- Summary: {summary_file}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"ENTIRE US STOCK MARKET DATA DOWNLOADED")
        print(f"{'='*60}")
        print(f"Total stocks with data: {len(self.valid_symbols)}")
        print(f"Data period: 2 years")
        print(f"\nTop sectors:")
        for sector, count in sorted(summary['sectors'].items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {sector}: {count} stocks")
        print(f"\nMarket cap distribution:")
        for range_name, count in summary['market_cap_ranges'].items():
            percentage = count / len(self.valid_symbols) * 100 if len(self.valid_symbols) > 0 else 0
            print(f"  {range_name}: {count} stocks ({percentage:.1f}%)")
        
        return symbols_file, summary_file

def main():
    logger.info("Starting entire US stock market data collection...")
    
    market_data = EntireUSMarketData()
    
    # Compile all symbols
    all_symbols = market_data.compile_entire_market()
    
    # Download historical data
    successful_data = market_data.download_historical_data(all_symbols)
    
    # Save the data
    symbols_file, summary_file = market_data.save_market_data()
    
    logger.info(f"Entire US market data collection complete!")
    logger.info(f"Successfully downloaded data for {len(successful_data)} stocks")
    
    return market_data.valid_symbols

if __name__ == "__main__":
    main()
