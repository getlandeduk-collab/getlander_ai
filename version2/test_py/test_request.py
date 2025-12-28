import requests
import json

# File path
pdf_path = r"C:\Users\nrhar\OneDrive\Desktop\hari docs1\HARI_V RESUME.pdf"

# User ID for saving job applications to Firestore
user_id = "EW8g71N4zTUEBRvomoG04oMwpJA3"  # Replace with actual user ID

# Test the API
print("Sending request...")
files = {
    'pdf_file': ('resume.pdf', open(pdf_path, 'rb'), 'application/pdf')
}
data = {
    'json_body': json.dumps({
        'urls': [
            'https://internshala.com/job/detail/fresher-data-science-ai-ml-research-associate-job-in-qutubullapur-at-megaminds-it-services1760064997',
            'https://internshala.com/job/detail/billing-executive-job-in-ahmedabad-at-cfirst-background-checks-llp1760686613'
        ],
        'user_id': user_id  # Add user_id to save job applications
    })
}

try:
    response = requests.post('http://localhost:8000/api/match-jobs', files=files, data=data)
    print(f"\nStatus: {response.status_code}")
    result = response.json()
    print(f"Response:\n{json.dumps(result, indent=2)}")
    
    if response.status_code == 200:
        print(f"\n✅ Successfully processed {result.get('jobs_analyzed', 0)} jobs")
        print(f"✅ Found {len(result.get('matched_jobs', []))} matched jobs")
        if user_id:
            print(f"✅ Job applications saved to Firestore for user: {user_id}")
except Exception as e:
    print(f"Error: {e}")
    if hasattr(e, 'response'):
        print(f"Response: {e.response.text}")
