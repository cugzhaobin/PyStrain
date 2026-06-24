"""Playwright-based integration tests for the PyStrain2 web app.

Tests the strain time-series workflow end-to-end using real data.

Run with:
    PYTHONPATH=/Users/zhao/PyStrain2/src /Users/zhao/anaconda3/bin/python3 tests/test_web_flows.py
"""

import os, sys, time
sys.path.insert(0, "/Users/zhao/PyStrain2/src")

from playwright.sync_api import sync_playwright

APP_URL = "http://127.0.0.1:8502"
DATA_DIR = "/Users/zhao/cmonoc/cmnc14/pytsfit"
GPS_FILE = "/Users/zhao/cmonoc/strain/timeseries/cmnc.llh"
POLY_FILE = "/Users/zhao/cmonoc/strain/timeseries/poly_list"
SCREENSHOT_DIR = "/tmp/pystrain2_test_screenshots"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

def test_landing_page(page):
    """Test the landing page loads and shows both workflow buttons."""
    page.goto(APP_URL)
    time.sleep(2)
    page.screenshot(path=f"{SCREENSHOT_DIR}/01_landing.png", full_page=True)
    assert "PyStrain2" in page.title()
    assert page.get_by_text("进入应变率场").is_visible()
    assert page.get_by_text("进入应变时间序列").is_visible()
    print("[PASS] Landing page loads correctly")

def test_timeseries_stations(page):
    """Test the time-series stations step."""
    page.goto(APP_URL)
    time.sleep(1)
    page.get_by_text("进入应变时间序列").click()
    time.sleep(2)
    page.screenshot(path=f"{SCREENSHOT_DIR}/02_stations_step.png", full_page=True)
    assert "测站坐标" in page.get_by_role("heading", level=1).first.inner_text()
    print("[PASS] Time-series stations step loads")

    # Upload GPS info file
    file_input = page.locator('input[type="file"]').first
    file_input.set_input_files(GPS_FILE)
    time.sleep(3)
    page.screenshot(path=f"{SCREENSHOT_DIR}/03_stations_uploaded.png", full_page=True)
    station_count_text = page.locator("text=已加载").first.inner_text()
    assert "已加载" in station_count_text
    print(f"[PASS] Stations uploaded: {station_count_text}")

def test_timeseries_config_with_polygon(page):
    """Test config step with polygon upload."""
    page.goto(APP_URL)
    time.sleep(1)
    page.get_by_text("进入应变时间序列").click()
    time.sleep(1)

    # Step 1: Upload stations
    file_input = page.locator('input[type="file"]').first
    file_input.set_input_files(GPS_FILE)
    time.sleep(3)

    # Navigate to config step
    next_btn = page.get_by_text("下一步 →")
    if next_btn.is_visible():
        next_btn.click()
    time.sleep(2)

    page.screenshot(path=f"{SCREENSHOT_DIR}/04_config_step.png", full_page=True)

    # Verify we're on the config step
    page_text = page.locator("body").inner_text()
    assert "时序配置" in page_text or "Step 2" in page_text
    print("[PASS] Config step loaded")

    # Upload polygon file (user method is default)
    # Find all file inputs (polygon upload is the second one)
    file_inputs = page.locator('input[type="file"]')
    count = file_inputs.count()
    print(f"  Found {count} file input(s)")

    if count >= 2:
        file_inputs.nth(1).set_input_files(POLY_FILE)
        time.sleep(3)
        page.screenshot(path=f"{SCREENSHOT_DIR}/05_polygon_uploaded.png", full_page=True)
        body = page.locator("body").inner_text()
        assert "已加载" in body and "多边形" in body
        print("[PASS] Polygon file uploaded successfully")

    # Set data directory path
    dir_input = page.locator('input[placeholder="/path/to/ts/data/"]')
    if dir_input.is_visible():
        dir_input.fill(DATA_DIR)
        time.sleep(1)
        print("[PASS] Data directory set")

    # Set epoch range
    sepoch_input = page.locator('input[aria-label="起始历元"]')
    if sepoch_input.is_visible():
        sepoch_input.fill("2020.0")
    eepoch_input = page.locator('input[aria-label="结束历元"]')
    if epoch_input.is_visible():
        eepoch_input.fill("2021.0")
    time.sleep(1)
    page.screenshot(path=f"{SCREENSHOT_DIR}/06_config_complete.png", full_page=True)
    print("[PASS] Epoch range set")

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        try:
            test_landing_page(page)
            test_timeseries_stations(page)
            test_timeseries_config_with_polygon(page)
            print(f"\nAll tests passed! Screenshots saved to {SCREENSHOT_DIR}/")
        except AssertionError as e:
            print(f"\nFAILED: {e}")
            page.screenshot(path=f"{SCREENSHOT_DIR}/FAILURE.png", full_page=True)
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            page.screenshot(path=f"{SCREENSHOT_DIR}/ERROR.png", full_page=True)
        finally:
            browser.close()

if __name__ == "__main__":
    main()
