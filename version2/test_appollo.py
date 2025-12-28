import requests
import json
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

url = "https://api.apollo.io/api/v1/people/match"

# Get API key from .env file or environment variable
# Create a .env file in the project root with: APOLLO_API_KEY=your_key_here
# Get your API key from: https://app.apollo.io/#/settings/integrations/api-keys
API_KEY = os.getenv("APOLLO_API_KEY")

if not API_KEY:
    print("❌ Error: APOLLO_API_KEY not found!")
    print("Please create a .env file in the project root with:")
    print("APOLLO_API_KEY=your_api_key_here")
    print("\nGet your API key from: https://app.apollo.io/#/settings/integrations/api-keys")
    exit(1)

# Apollo API typically expects data in request body, not query params
data = {
    "email": "josh.garrison@apollo.io",
    "reveal_personal_emails": False,
    "reveal_phone_number": True,
    "webhook_url": "https://webhook-test.com/5b112b64ff0f4104d003444e843c161a"
}

headers = {
    "accept": "application/json",
    "Content-Type": "application/json",
    "x-api-key": API_KEY  # Use environment variable or replace with valid key
}

try:
    # Try with data in request body (most common for POST)
    response = requests.post(url, json=data, headers=headers)
    
    # Debug: Check status and content before parsing JSON
    print(f"Status Code: {response.status_code}")
    print(f"Response Headers: {dict(response.headers)}")
    print(f"\nResponse Text (first 1000 chars):")
    print(response.text[:1000])
    print(f"\nResponse Length: {len(response.text)} bytes")
    
    # Check status code first
    if response.status_code == 401:
        print("\n❌ Authentication Error (401): Invalid API key")
        print(f"The API key appears to be invalid or expired.")
        print("To fix this:")
        print("1. Get a valid API key from: https://app.apollo.io/#/settings/integrations/api-keys")
        print("2. Add it to your .env file: APOLLO_API_KEY=your_key_here")
        print("3. Make sure the .env file is in the project root directory")
        # Try to parse as JSON anyway (some APIs return JSON errors)
        try:
            error_data = response.json()
            print(f"Error details: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Error message: {response.text}")
    elif response.status_code == 200:
        # Try to parse JSON for successful responses
        if response.text.strip():
            try:
                data = response.json()
                print("\n=== ✅ JSON Response ===")
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError as e:
                print(f"\n❌ JSON Decode Error: {e}")
                print(f"Response is not valid JSON. Full response:")
                print(response.text)
        else:
            print("\n❌ Response is empty")
    else:
        # Handle other error status codes
        print(f"\n❌ Error: HTTP {response.status_code}")
        try:
            error_data = response.json()
            print(f"Error response: {json.dumps(error_data, indent=2)}")
        except:
            print(f"Error message: {response.text}")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Request failed: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()