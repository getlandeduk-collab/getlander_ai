"""
Quick diagnostic script to check Firestore permissions and security rules.
Run this to identify permission issues.
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

import os
import json
from dotenv import load_dotenv

def check_firestore_access():
    """Check if we can access Firestore with current credentials."""
    
    print("=" * 80)
    print("FIRESTORE PERMISSIONS DIAGNOSTIC")
    print("=" * 80)
    
    # Load environment
    load_dotenv()
    creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    
    if not creds_path:
        print("\n[ERROR] GOOGLE_APPLICATION_CREDENTIALS not set")
        print("\nFix: Set the environment variable in .env file")
        return False
    
    print(f"\n[1] Credentials Path: {creds_path}")
    
    # Load credentials to check service account
    try:
        with open(creds_path, 'r') as f:
            creds_data = json.load(f)
        
        project_id = creds_data.get('project_id', 'N/A')
        client_email = creds_data.get('client_email', 'N/A')
        
        print(f"[2] Project ID: {project_id}")
        print(f"[3] Service Account: {client_email}")
        
        # Check if it's a valid service account
        if 'iam.gserviceaccount.com' not in client_email:
            print("\n[WARNING] This doesn't look like a valid service account email")
            print("Expected format: xxxxx@xxxxx.iam.gserviceaccount.com")
        
        print("\n[4] Testing Firebase initialization...")
        
        try:
            import firebase_admin
            from firebase_admin import credentials, firestore
            
            # Check if already initialized
            try:
                app = firebase_admin.get_app()
                print("    [OK] Firebase already initialized")
            except ValueError:
                # Initialize
                cred = credentials.Certificate(creds_path)
                app = firebase_admin.initialize_app(cred)
                print("    [OK] Firebase initialized successfully")
            
            # Get Firestore client
            print("\n[5] Testing Firestore client...")
            db = firestore.client()
            print("    [OK] Firestore client created")
            
            # Try a test read operation
            print("\n[6] Testing Firestore READ access...")
            try:
                # Try to list collections (doesn't require specific data)
                collections = db.collections()
                collection_list = [col.id for col in collections]
                print(f"    [OK] Can read Firestore (found {len(collection_list)} collections)")
                if collection_list:
                    print(f"    Collections: {', '.join(collection_list)}")
            except Exception as read_error:
                print(f"    [ERROR] Cannot read from Firestore: {read_error}")
                return False
            
            # Try a test write operation
            print("\n[7] Testing Firestore WRITE access...")
            test_user_id = "TEST_USER_DIAGNOSTIC"
            
            try:
                # Try to write a test document
                test_ref = db.collection("users").document(test_user_id).collection("test")
                doc_ref = test_ref.document("diagnostic_test")
                
                doc_ref.set({
                    "test": True,
                    "message": "Diagnostic test document",
                    "timestamp": firestore.SERVER_TIMESTAMP
                })
                
                print("    [OK] Successfully wrote test document")
                
                # Try to read it back
                doc = doc_ref.get()
                if doc.exists:
                    print("    [OK] Successfully read back test document")
                    
                    # Clean up - delete test document
                    doc_ref.delete()
                    print("    [OK] Cleaned up test document")
                else:
                    print("    [WARNING] Could not read back test document")
                
                print("\n" + "=" * 80)
                print("SUCCESS - FIRESTORE PERMISSIONS ARE WORKING!")
                print("=" * 80)
                print("\nYour service account has:")
                print("  - READ access to Firestore")
                print("  - WRITE access to Firestore")
                print("  - Proper authentication")
                print("\nThe 'Invalid JWT Signature' error is likely due to:")
                print("  1. The service account key being revoked")
                print("  2. System clock being out of sync")
                print("  3. Network/connectivity issues")
                print("\nRecommended fix:")
                print("  1. Generate a NEW service account key")
                print("  2. Replace the old key file")
                print("  3. Run this diagnostic again")
                print("\nSee: FIX_FIREBASE_CREDENTIALS.md")
                
                return True
                
            except Exception as write_error:
                error_msg = str(write_error)
                print(f"    [ERROR] Cannot write to Firestore: {write_error}")
                
                if "PERMISSION_DENIED" in error_msg:
                    print("\n" + "=" * 80)
                    print("ISSUE: PERMISSION DENIED")
                    print("=" * 80)
                    print("\nYour service account lacks write permissions.")
                    print("\nFix options:")
                    print("\n1. UPDATE FIRESTORE SECURITY RULES (Easiest):")
                    print(f"   https://console.firebase.google.com/project/{project_id}/firestore/rules")
                    print("\n   Use this rule for testing:")
                    print("   ```")
                    print("   rules_version = '2';")
                    print("   service cloud.firestore {")
                    print("     match /databases/{database}/documents {")
                    print("       match /{document=**} {")
                    print("         allow read, write: if true;")
                    print("       }")
                    print("     }")
                    print("   }")
                    print("   ```")
                    print("\n2. UPDATE SERVICE ACCOUNT PERMISSIONS:")
                    print(f"   https://console.cloud.google.com/iam-admin/iam?project={project_id}")
                    print("\n   Add role: 'Cloud Datastore User' to service account:")
                    print(f"   {client_email}")
                    print("\n3. GENERATE NEW SERVICE ACCOUNT KEY:")
                    print(f"   https://console.firebase.google.com/project/{project_id}/settings/serviceaccounts/adminsdk")
                    
                elif "invalid_grant" in error_msg or "Invalid JWT" in error_msg:
                    print("\n" + "=" * 80)
                    print("ISSUE: INVALID SERVICE ACCOUNT KEY")
                    print("=" * 80)
                    print("\nThe service account key is invalid, revoked, or expired.")
                    print("\nFix: Generate a NEW service account key:")
                    print(f"1. Go to: https://console.firebase.google.com/project/{project_id}/settings/serviceaccounts/adminsdk")
                    print("2. Click 'Generate new private key'")
                    print("3. Save the downloaded JSON file")
                    print("4. Update GOOGLE_APPLICATION_CREDENTIALS in .env")
                    print("\nSee: FIX_FIREBASE_CREDENTIALS.md for detailed steps")
                    
                return False
                
        except ImportError:
            print("\n[ERROR] firebase-admin not installed")
            print("\nFix: pip install firebase-admin")
            return False
        except Exception as init_error:
            print(f"    [ERROR] Firebase initialization failed: {init_error}")
            print("\nThis usually means:")
            print("  1. Invalid service account key")
            print("  2. Key has been revoked")
            print("  3. Wrong credentials file")
            return False
            
    except FileNotFoundError:
        print(f"\n[ERROR] Credentials file not found: {creds_path}")
        return False
    except json.JSONDecodeError:
        print(f"\n[ERROR] Invalid JSON in credentials file: {creds_path}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    success = check_firestore_access()
    
    if success:
        print("\n[NEXT STEP] Run your actual tests:")
        print("  python test_firebase_simple.py")
        print("  python test_your_urls.py")
    else:
        print("\n[ACTION REQUIRED] Fix the issues above before proceeding.")
        print("\nSee:")
        print("  - CHECK_FIRESTORE_PERMISSIONS.md")
        print("  - FIX_FIREBASE_CREDENTIALS.md")
    
    exit(0 if success else 1)


