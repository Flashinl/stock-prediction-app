#!/usr/bin/env python3
"""
Test script to measure the loading speed of opportunities section
"""
import time
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options

def test_opportunities_loading_speed():
    """Test how fast the opportunities section loads"""
    
    # Setup Chrome options for headless testing
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    try:
        # Initialize the driver
        driver = webdriver.Chrome(options=chrome_options)
        
        print("🚀 Testing Today's Top Opportunities loading speed...")
        
        # Navigate to the homepage
        start_time = time.time()
        driver.get("http://127.0.0.1:5000")
        
        # Wait for the page to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "featuredGrid"))
        )
        
        # Wait for opportunities to load (look for actual opportunity cards)
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.CLASS_NAME, "opportunity-card"))
        )
        
        end_time = time.time()
        loading_time = end_time - start_time
        
        # Count the number of opportunities loaded
        opportunity_cards = driver.find_elements(By.CLASS_NAME, "opportunity-card")
        
        print(f"✅ Opportunities loaded successfully!")
        print(f"⏱️  Total loading time: {loading_time:.2f} seconds")
        print(f"📊 Number of opportunities: {len(opportunity_cards)}")
        
        # Check if loading time is acceptable (should be under 20 seconds now)
        if loading_time < 20:
            print(f"🎉 FAST! Loading time is excellent ({loading_time:.2f}s)")
        elif loading_time < 30:
            print(f"✅ GOOD! Loading time is acceptable ({loading_time:.2f}s)")
        else:
            print(f"⚠️  SLOW! Loading time needs improvement ({loading_time:.2f}s)")
        
        # Test caching by refreshing the page
        print("\n🔄 Testing cache performance...")
        cache_start_time = time.time()
        driver.refresh()
        
        # Wait for cached opportunities to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "opportunity-card"))
        )
        
        cache_end_time = time.time()
        cache_loading_time = cache_end_time - cache_start_time
        
        print(f"⚡ Cached loading time: {cache_loading_time:.2f} seconds")
        
        if cache_loading_time < 5:
            print(f"🚀 EXCELLENT! Cache is working perfectly ({cache_loading_time:.2f}s)")
        elif cache_loading_time < 10:
            print(f"✅ GOOD! Cache is working well ({cache_loading_time:.2f}s)")
        else:
            print(f"⚠️  Cache could be improved ({cache_loading_time:.2f}s)")
        
        return {
            "initial_load_time": loading_time,
            "cached_load_time": cache_loading_time,
            "opportunities_count": len(opportunity_cards),
            "success": True
        }
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return {
            "success": False,
            "error": str(e)
        }
    
    finally:
        try:
            driver.quit()
        except:
            pass

def test_api_speed():
    """Test the speed of individual API calls"""
    print("\n🔧 Testing individual API call speeds...")
    
    test_symbols = ['AAPL', 'MSFT', 'GOOGL']
    
    for symbol in test_symbols:
        start_time = time.time()
        try:
            response = requests.post(
                'http://127.0.0.1:5000/api/predict',
                json={'symbol': symbol, 'timeframe': 'auto'},
                timeout=15
            )
            end_time = time.time()
            
            if response.status_code == 200:
                api_time = end_time - start_time
                print(f"  {symbol}: {api_time:.2f}s")
            else:
                print(f"  {symbol}: API Error {response.status_code}")
                
        except Exception as e:
            print(f"  {symbol}: Failed - {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 STOCKTREK OPPORTUNITIES SPEED TEST")
    print("=" * 60)
    
    # Test API speeds first
    test_api_speed()
    
    # Test full page loading
    result = test_opportunities_loading_speed()
    
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    
    if result["success"]:
        print(f"✅ Test completed successfully")
        print(f"⏱️  Initial load: {result['initial_load_time']:.2f}s")
        print(f"⚡ Cached load: {result['cached_load_time']:.2f}s")
        print(f"📊 Opportunities: {result['opportunities_count']}")
        
        # Performance rating
        if result['initial_load_time'] < 15:
            print("🏆 PERFORMANCE: EXCELLENT")
        elif result['initial_load_time'] < 25:
            print("🥈 PERFORMANCE: GOOD")
        else:
            print("🥉 PERFORMANCE: NEEDS IMPROVEMENT")
    else:
        print(f"❌ Test failed: {result['error']}")
