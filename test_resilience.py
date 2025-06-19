#!/usr/bin/env python3
"""
Test script to verify OCR process resilience improvements
"""

import sys
import time

# Add the application path to Python path
app_path = (
    "/home/bruno/codigo-pacotes/bigocrpdf-bkp/bigocrpdf/usr/share/biglinux/bigocrpdf"
)
sys.path.insert(0, app_path)

from services.ocr_api import OcrProcess, OcrQueue


def test_ocr_process_resilience():
    """Test the resilience improvements to OCR process startup"""

    print("Testing OCR Process Resilience Improvements")
    print("=" * 50)

    # Test with a non-existent file to trigger retry logic
    print("\n1. Testing with non-existent file (should fail gracefully):")
    try:
        options = {"force_ocr": True}
        process = OcrProcess(
            input_file="/non/existent/file.pdf",
            output_file="/tmp/test_output.pdf",
            options=options,
            page_count=1,
        )
        process.start()
        print("   ❌ Should have failed but didn't")
    except Exception as e:
        print(f"   ✅ Failed as expected: {e}")

    # Test OCR Queue startup validation
    print("\n2. Testing OCR Queue startup:")
    try:
        queue = OcrQueue(max_concurrent=1)
        print("   ✅ OCR Queue created successfully")

        # Test starting empty queue
        queue.start()
        print("   ✅ Empty queue started without errors")

        queue.stop()
        print("   ✅ Queue stopped successfully")

    except Exception as e:
        print(f"   ❌ OCR Queue test failed: {e}")

    print("\n3. Testing process health monitoring:")
    try:
        queue = OcrQueue(max_concurrent=1)

        # Create a mock process to test health checking
        class MockProcess:
            def __init__(self):
                self.input_file = "test.pdf"
                self.start_time = time.time()
                self.progress_file = "/tmp/test_progress.txt"
                self.process = None

        mock_process = MockProcess()

        # Test health check
        health = queue._check_process_health(mock_process)
        print(f"   Health check result: {'✅ Healthy' if health else '❌ Unhealthy'}")

    except Exception as e:
        print(f"   ❌ Health monitoring test failed: {e}")

    print("\nTest completed!")


if __name__ == "__main__":
    test_ocr_process_resilience()
