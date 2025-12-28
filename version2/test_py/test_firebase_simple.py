"""
Simple test script to verify Firebase document saving works.
Tests adding a document to users/{user_id}/job_applications/
"""
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import os

# Replace with your actual user ID
user_id = "EW8g71N4zTUEBRvomoG04oMwpJA3"

def test_save_document():
    """Test saving a document to Firestore."""
    try:
        print("="*70)
        print("Testing Firebase Document Save")
        print("="*70)
        
        # Step 1: Initialize Firebase (check if already initialized)
        try:
            app = firebase_admin.get_app()
            print("[OK] Firebase already initialized")
        except ValueError:
            # Not initialized yet
            print("Initializing Firebase...")
            
            # Load environment variables from .env if present
            from dotenv import load_dotenv
            load_dotenv()
            
            # Try to use service account key file
            service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            
            print(f"Checking GOOGLE_APPLICATION_CREDENTIALS: {service_account_path}")
            
            if service_account_path:
                # Handle Windows paths - normalize separators
                # Try different path formats
                paths_to_try = [
                    service_account_path,  # Original
                    service_account_path.replace('/', '\\'),  # Windows backslash
                    service_account_path.replace('\\', '/'),  # Forward slash
                    os.path.normpath(service_account_path),  # Normalized
                    os.path.abspath(service_account_path),  # Absolute
                ]
                
                path_to_use = None
                for path in paths_to_try:
                    if os.path.exists(path):
                        path_to_use = path
                        break
                
                if path_to_use:
                    print(f"Found service account file: {path_to_use}")
                    cred = credentials.Certificate(path_to_use)
                    firebase_admin.initialize_app(cred)
                else:
                    raise FileNotFoundError(
                        f"Service account file not found at any of these paths:\n" +
                        "\n".join(f"  - {p}" for p in paths_to_try)
                    )
            else:
                # Try with project ID from environment
                project_id = os.getenv("VITE_FIREBASE_PROJECT_ID") or os.getenv("FIREBASE_PROJECT_ID")
                if not project_id:
                    raise ValueError(
                        "GOOGLE_APPLICATION_CREDENTIALS or FIREBASE_PROJECT_ID not found.\n"
                        "Set GOOGLE_APPLICATION_CREDENTIALS=C:/Users/nrhar/Downloads/serviceAccountKey.json"
                    )
                
                print(f"Using project ID: {project_id}")
                # Try to initialize with project ID (uses Application Default Credentials)
                firebase_admin.initialize_app(options={'projectId': project_id})
            
            print("[OK] Firebase initialized successfully")
        
        # Step 2: Create Firestore client
        db = firestore.client()
        print("[OK] Firestore client created")
        
        # Step 3: Reference to job_applications collection
        collection_ref = db.collection("users").document(user_id).collection("job_applications")
        print(f"[OK] Collection reference created: users/{user_id}/job_applications")
        
        # Step 4: Data to add
        job_data = {
            "appliedDate": datetime.now(),
            "company": "Hiringlabs Business Solutions (HBS)",
            "createdAt": datetime.now(),
            "interviewDate": "",
            "jobDescription": "Test job description",
            "link": "https://internshala.com/internships/computer-science-internship/",
            "notes": "This is a test document",
            "portal": "LinkedIn",
            "role": "Food QC & Inventory Management",
            "status": "Applied",
            "visaRequired": "No"
        }
        
        print(f"\n[INFO] Document data to save:")
        print(f"  Company: {job_data['company']}")
        print(f"  Role: {job_data['role']}")
        print(f"  Portal: {job_data['portal']}")
        print(f"  Link: {job_data['link']}")
        
        # Step 5: Add document (auto ID)
        print(f"\n[SAVE] Saving document to Firestore...")
        result = collection_ref.add(job_data)
        
        # Handle return value - add() returns (timestamp, document_reference)
        if isinstance(result, tuple):
            update_time, doc_ref = result
            doc_id = doc_ref.id
        else:
            doc_ref = result
            doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
        
        print(f"\n[SUCCESS] Document added with ID: {doc_id}")
        print(f"\n[PATH] users/{user_id}/job_applications/{doc_id}")
        print(f"\n[INFO] Verify in Firebase Console:")
        print(f"   Firestore -> users -> {user_id} -> job_applications -> {doc_id}")
        
        # Step 6: Verify by reading it back
        print(f"\n[VERIFY] Verifying document was saved...")
        doc = collection_ref.document(doc_id).get()
        
        if doc.exists:
            print(f"[OK] Verification successful! Document exists.")
            doc_dict = doc.to_dict()
            print(f"\n[DOCUMENT] Contents:")
            for key, value in doc_dict.items():
                print(f"  {key}: {value}")
        else:
            print(f"[WARNING] Document not found after saving")
        
        return doc_id
        
    except Exception as e:
        print(f"\n[ERROR]: {str(e)}")
        import traceback
        print(f"\n[TRACEBACK]:")
        print(traceback.format_exc())
        
        print("\n" + "="*70)
        print("Troubleshooting:")
        print("="*70)
        print("1. Make sure firebase-admin is installed: pip install firebase-admin")
        print("2. Set GOOGLE_APPLICATION_CREDENTIALS environment variable:")
        print("   set GOOGLE_APPLICATION_CREDENTIALS=C:/Users/nrhar/Downloads/serviceAccountKey.json")
        print("   OR add it to your .env file")
        print("3. Verify the service account JSON file exists at the specified path")
        print("4. Verify your service account has Firestore write permissions")
        print("5. Check Firestore security rules allow writes")
        return None


if __name__ == "__main__":
    print("Firebase Document Save Test")
    print("="*70)
    print(f"User ID: {user_id}")
    print()
    
    doc_id = test_save_document()
    
    if doc_id:
        print("\n" + "="*70)
        print("[PASSED] TEST PASSED - Document saved successfully!")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("[FAILED] TEST FAILED - Could not save document")
        print("="*70)

