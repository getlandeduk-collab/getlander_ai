"""
Test script for the job information extraction endpoint.
"""
import requests
import json

BASE_URL = "http://localhost:8000"


def test_extract_job_info(job_url: str):
    """Test extracting job information from a URL."""
    print(f"\n{'='*70}")
    print("  Test: Extract Job Information")
    print(f"{'='*70}")
    
    url = f"{BASE_URL}/api/extract-job-info"
    payload = {
        "job_url": job_url
    }
    
    print(f"POST {url}")
    print(f"Request Body: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("\n✓ Extraction successful!")
            print(f"\nResults:")
            print(f"  Job URL: {data['job_url']}")
            print(f"  Job Title: {data.get('job_title', 'N/A')}")
            print(f"  Company Name: {data.get('company_name', 'N/A')}")
            print(f"  Portal: {data.get('portal', 'N/A')}")
            print(f"  Success: {data['success']}")
            if data.get('error'):
                print(f"  Error: {data['error']}")
        else:
            print(f"✗ Error: {response.text}")
    except requests.exceptions.ConnectionError:
        print("✗ Connection Error: Is the server running?")
        print("  Start server with: uvicorn version2.app:app --reload")
    except Exception as e:
        print(f"✗ Exception: {str(e)}")


if __name__ == "__main__":
    print("="*70)
    print("  Job Information Extraction API - Test")
    print("="*70)
    print(f"\nBase URL: {BASE_URL}")
    
    # Example URLs for testing
    example_urls = [
        "https://internshala.com/job/detail/fresher-data-science-ai-ml-research-associate-job-in-qutubullapur-at-megaminds-it-services1760064997",
        "https://www.linkedin.com/jobs/view/1234567890",
        "https://www.indeed.com/viewjob?jk=abc123",
    ]
    
    # Get URL from user or use example
    print("\nExample URLs:")
    for i, url in enumerate(example_urls, 1):
        print(f"  {i}. {url}")
    
    user_input = input("\nEnter job URL (or press Enter to use first example): ").strip()
    
    if user_input:
        test_url = user_input
    else:
        test_url = example_urls[0]
    
    print("\nMake sure the server is running:")
    print("  uvicorn version2.app:app --reload")
    
    input("\nPress Enter to test...")
    
    test_extract_job_info(test_url)
    
    print("\n" + "="*70)
    print("  Testing Complete")
    print("="*70)
    print("\nTip: You can also test the endpoint at http://localhost:8000/docs")

