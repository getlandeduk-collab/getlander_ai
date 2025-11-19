"""
Firebase service for fetching resumes from Firestore.
"""
from __future__ import annotations

import os
import base64
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

# Load environment variables
load_dotenv()
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)

# CRITICAL: Also explicitly load from system environment
# This ensures GOOGLE_APPLICATION_CREDENTIALS_JSON or GOOGLE_APPLICATION_CREDENTIALS is available
import os as _os

# Check for JSON string in environment variable (preferred for production)
if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in _os.environ:
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS_JSON", _os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])
    json_value = _os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"]
    # Show first/last 50 chars for security (don't print full JSON)
    if len(json_value) > 100:
        preview = json_value[:50] + "..." + json_value[-50:]
    else:
        preview = json_value[:50] + "..."
    print(f"[Firebase] GOOGLE_APPLICATION_CREDENTIALS_JSON found in system environment (length: {len(json_value)})")
    print(f"[Firebase] Preview: {preview}")
elif "GOOGLE_APPLICATION_CREDENTIALS" in _os.environ:
    # Fallback to file path (for local development)
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", _os.environ["GOOGLE_APPLICATION_CREDENTIALS"])
    print(f"[Firebase] GOOGLE_APPLICATION_CREDENTIALS found in system environment: {_os.environ['GOOGLE_APPLICATION_CREDENTIALS']}")
    print(f"[Firebase] GOOGLE_APPLICATION_CREDENTIALS in os.environ: {os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')}")
else:
    print(f"[Firebase] [WARNING] Neither GOOGLE_APPLICATION_CREDENTIALS_JSON nor GOOGLE_APPLICATION_CREDENTIALS found in system environment")


class FirebaseService:
    """Service for interacting with Firebase Firestore to fetch resumes."""
    
    _app = None
    _db = None
    
    def __init__(self):
        """Initialize Firebase Admin SDK."""
        if not FIREBASE_AVAILABLE:
            raise ImportError(
                "firebase-admin is not installed. Install it with: pip install firebase-admin"
            )
        
        print(f"[Firebase] [INIT] Initializing FirebaseService instance...")
        
        # Initialize Firebase Admin SDK if not already initialized
        if FirebaseService._app is None:
            print("[Firebase] [INIT] Firebase app is None, initializing...")
            self._initialize_firebase()
        else:
            print("[Firebase] [INIT] Firebase app already exists")
        
        # Always get a fresh client instance to ensure it's properly initialized
        # EXACT same pattern as test_firebase_simple.py line 79: db = firestore.client()
        if FirebaseService._db is None:
            print("[Firebase] [INIT] Creating new Firestore client...")
            FirebaseService._db = firestore.client()
            print("[Firebase] [INIT] [OK] Firestore client created")
        else:
            print("[Firebase] [INIT] Using existing Firestore client")
            
        print(f"[Firebase] [INIT] FirebaseService initialized successfully")
        print(f"[Firebase] [INIT] _app is None: {FirebaseService._app is None}")
        print(f"[Firebase] [INIT] _db is None: {FirebaseService._db is None}")
    
    def _initialize_firebase(self):
        """
        Initialize Firebase Admin SDK with credentials from environment variables.
        
        Priority:
        1. GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string directly in env var)
        2. GOOGLE_APPLICATION_CREDENTIALS (file path to JSON file)
        3. FIREBASE_PROJECT_ID (for Application Default Credentials)
        """
        try:
            # Try to initialize Firebase Admin SDK
            try:
                # First, check if Firebase is already initialized
                FirebaseService._app = firebase_admin.get_app()
                print("[Firebase] Firebase already initialized")
            except ValueError:
                # Not initialized yet
                print("[Firebase] Initializing Firebase...")
                
                # Load environment variables from .env if present
                from dotenv import load_dotenv
                load_dotenv()
                
                # METHOD 1: Try GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string in env var)
                # This is preferred for production/deployment (e.g., Render, Heroku, etc.)
                firebase_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON") or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_JSON")
                
                if firebase_json:
                    print("[Firebase] Found GOOGLE_APPLICATION_CREDENTIALS_JSON (using JSON from environment variable)")
                    try:
                        # Convert string to dict
                        cred_dict = json.loads(firebase_json)
                        # Initialize Firebase Admin SDK with JSON dict
                        cred = credentials.Certificate(cred_dict)
                        FirebaseService._app = firebase_admin.initialize_app(cred)
                        print("[Firebase] [OK] Firebase initialized successfully from JSON string")
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON in GOOGLE_APPLICATION_CREDENTIALS_JSON: {str(e)}")
                    except Exception as e:
                        raise RuntimeError(f"Failed to initialize Firebase from JSON: {str(e)}")
                
                # METHOD 2: Try GOOGLE_APPLICATION_CREDENTIALS (file path)
                # Fallback for local development
                elif not FirebaseService._app:
                    print("[Firebase] GOOGLE_APPLICATION_CREDENTIALS_JSON not found, trying file path...")
                    
                    # Try multiple methods to get GOOGLE_APPLICATION_CREDENTIALS
                    # 1. Try os.getenv (from dotenv)
                    service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
                    
                    # 2. Try os.environ directly
                    if not service_account_path:
                        service_account_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
                    
                    # 3. Try system environment (Windows)
                    if not service_account_path:
                        try:
                            import subprocess
                            result = subprocess.run(
                                ['powershell', '-Command', '[Environment]::GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS", "User")'],
                                capture_output=True,
                                text=True,
                                timeout=2
                            )
                            if result.returncode == 0 and result.stdout.strip():
                                service_account_path = result.stdout.strip()
                                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_path
                        except:
                            pass
                    
                    print(f"[Firebase] Checking GOOGLE_APPLICATION_CREDENTIALS: {service_account_path}")
                    
                    if service_account_path:
                        # Handle Windows paths - normalize separators
                        # Try different path formats (same as test_firebase_simple.py)
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
                            print(f"[Firebase] Found service account file: {path_to_use}")
                            cred = credentials.Certificate(path_to_use)
                            FirebaseService._app = firebase_admin.initialize_app(cred)
                            print("[Firebase] [OK] Firebase initialized successfully from file")
                        else:
                            raise FileNotFoundError(
                                f"Service account file not found at any of these paths:\n" +
                                "\n".join(f"  - {p}" for p in paths_to_try)
                            )
                
                # METHOD 3: Try with project ID from environment (Application Default Credentials)
                if not FirebaseService._app:
                    project_id = os.getenv("VITE_FIREBASE_PROJECT_ID") or os.getenv("FIREBASE_PROJECT_ID")
                    if not project_id:
                        raise ValueError(
                            "No Firebase credentials found. Please set one of:\n"
                            "  - GOOGLE_APPLICATION_CREDENTIALS_JSON (JSON string)\n"
                            "  - GOOGLE_APPLICATION_CREDENTIALS (file path)\n"
                            "  - FIREBASE_PROJECT_ID (for Application Default Credentials)"
                        )
                    
                    print(f"[Firebase] Using project ID: {project_id}")
                    # Try to initialize with project ID (uses Application Default Credentials)
                    FirebaseService._app = firebase_admin.initialize_app(options={'projectId': project_id})
                    print("[Firebase] [OK] Firebase initialized successfully with project ID")
                
        except Exception as e:
            if isinstance(e, (RuntimeError, ValueError, FileNotFoundError)):
                raise
            raise RuntimeError(f"Failed to initialize Firebase: {str(e)}")
    
    def get_user_resumes(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Fetch all resumes for a specific user from Firestore.
        
        Args:
            user_id: The user ID to fetch resumes for
            
        Returns:
            List of resume dictionaries containing resume data
        """
        try:
            # Reference to the resumes collection for this user
            resumes_ref = FirebaseService._db.collection("users").document(user_id).collection("resumes")
            
            # Fetch all resumes
            resumes_docs = resumes_ref.stream()
            
            resumes = []
            for doc in resumes_docs:
                resume_data = doc.to_dict()
                resume_data["id"] = doc.id  # Add document ID
                resumes.append(resume_data)
            
            return resumes
            
        except Exception as e:
            raise RuntimeError(f"Failed to fetch resumes for user {user_id}: {str(e)}")
    
    def get_resume_by_id(self, user_id: str, resume_id: str) -> Optional[Dict[str, Any]]:
        """
        Fetch a specific resume by ID for a user.
        
        Args:
            user_id: The user ID
            resume_id: The resume document ID
            
        Returns:
            Resume dictionary or None if not found
        """
        try:
            resume_ref = (
                FirebaseService._db
                .collection("users")
                .document(user_id)
                .collection("resumes")
                .document(resume_id)
            )
            
            resume_doc = resume_ref.get()
            
            if resume_doc.exists:
                resume_data = resume_doc.to_dict()
                resume_data["id"] = resume_doc.id
                return resume_data
            else:
                return None
                
        except Exception as e:
            raise RuntimeError(
                f"Failed to fetch resume {resume_id} for user {user_id}: {str(e)}"
            )
    
    def extract_pdf_base64(self, resume_data: Dict[str, Any]) -> Optional[str]:
        """
        Extract base64 PDF content from resume data.
        
        Handles the PDF_BASE64: prefix format mentioned in the requirements.
        
        Args:
            resume_data: Dictionary containing resume data with content field
            
        Returns:
            Base64 string (without prefix) or None if not found
        """
        content = resume_data.get("content")
        
        if not content:
            return None
        
        # Handle content that starts with "PDF_BASE64:" prefix
        if isinstance(content, str):
            if content.startswith("PDF_BASE64:"):
                return content[len("PDF_BASE64:"):]
            # If already plain base64, return as is
            return content
        
        return None
    
    def get_resume_pdf_bytes(self, user_id: str, resume_id: str) -> Optional[bytes]:
        """
        Get resume PDF as bytes.
        
        Args:
            user_id: The user ID
            resume_id: The resume document ID
            
        Returns:
            PDF bytes or None if not found
        """
        resume_data = self.get_resume_by_id(user_id, resume_id)
        
        if not resume_data:
            return None
        
        base64_content = self.extract_pdf_base64(resume_data)
        
        if not base64_content:
            return None
        
        try:
            # Decode base64 to bytes
            return base64.b64decode(base64_content)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 PDF content: {str(e)}")
    
    def get_user_saved_cvs(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Get saved CVs array for a user (as mentioned in the requirements).
        This might be stored at the user document level.
        
        Args:
            user_id: The user ID
            
        Returns:
            List of saved CV references/data
        """
        try:
            user_ref = FirebaseService._db.collection("users").document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                user_data = user_doc.to_dict()
                saved_cvs = user_data.get("savedCVs", [])
                return saved_cvs if isinstance(saved_cvs, list) else []
            else:
                return []
                
        except Exception as e:
            raise RuntimeError(f"Failed to fetch savedCVs for user {user_id}: {str(e)}")

    def _check_job_application_duplicate(self, user_id: str, job_data: Dict[str, Any]) -> Optional[str]:
        """
        Check if a job application already exists for this user.
        
        Args:
            user_id: The user ID
            job_data: Dictionary containing job application data
            
        Returns:
            Document ID of existing duplicate if found, None otherwise
        """
        try:
            db = FirebaseService._db
            if db is None:
                return None
            
            collection_ref = db.collection("users").document(user_id).collection("job_applications")
            
            # Extract fields to check for duplicates
            job_link = job_data.get("link", "").strip()
            company = job_data.get("company", "").strip()
            role = job_data.get("role", "").strip()
            
            # Strategy 1: Check by job URL (most reliable)
            if job_link:
                query = collection_ref.where("link", "==", job_link).limit(1)
                docs = query.stream()
                for doc in docs:
                    print(f"[Firebase] [DUPLICATE] Found duplicate job application by URL: {doc.id}")
                    return doc.id
            
            # Strategy 2: Check by company + role (fallback if no URL)
            if company and role:
                query = collection_ref.where("company", "==", company).where("role", "==", role).limit(1)
                docs = query.stream()
                for doc in docs:
                    print(f"[Firebase] [DUPLICATE] Found duplicate job application by company+role: {doc.id}")
                    return doc.id
            
            return None
            
        except Exception as e:
            print(f"[Firebase] [DUPLICATE] Error checking for duplicate: {e}")
            # Don't fail on duplicate check errors, just log and continue
            return None
    
    def save_job_application(self, user_id: str, job_data: Dict[str, Any]) -> str:
        """
        Save a job application to Firestore.
        Uses the EXACT same approach as test_firebase_simple.py
        Checks for duplicates before saving.
        
        Args:
            user_id: The user ID
            job_data: Dictionary containing job application data
            
        Returns:
            The document ID of the saved application (or existing duplicate)
        """
        try:
            print(f"\n{'='*70}")
            print(f"[Firebase] [SAVE] Starting save_job_application")
            print(f"{'='*70}")
            
            # CRITICAL: Ensure Firebase app is initialized
            if FirebaseService._app is None:
                print("[Firebase] [WARNING] Firebase app is None, re-initializing...")
                self._initialize_firebase()
            
            # CRITICAL: Ensure Firestore client is initialized and get a fresh reference
            # Use the EXACT same pattern as test_firebase_simple.py line 79
            print(f"[Firebase] [DEBUG] Checking Firestore client...")
            print(f"[Firebase] [DEBUG] FirebaseService._db is None: {FirebaseService._db is None}")
            
            if FirebaseService._db is None:
                print("[Firebase] [WARNING] Firestore client is None, creating new client...")
                FirebaseService._db = firestore.client()
                print("[Firebase] [OK] Firestore client created")
            else:
                print("[Firebase] [OK] Using existing Firestore client")
                # Verify client is actually usable by creating a fresh reference
                try:
                    test_ref = FirebaseService._db.collection("users")
                    print(f"[Firebase] [OK] Client is usable (type: {type(FirebaseService._db)})")
                except Exception as test_error:
                    print(f"[Firebase] [ERROR] Client not usable: {test_error}")
                    print("[Firebase] [WARNING] Recreating client...")
                    FirebaseService._db = firestore.client()
                    print("[Firebase] [OK] New Firestore client created")
            
            # Get a direct reference to the db client (same as test script)
            db = FirebaseService._db
            print(f"[Firebase] [DEBUG] Using db client: {type(db)}")
            
            # Check for duplicates before saving
            print(f"[Firebase] [DUPLICATE] Checking for duplicate job application...")
            existing_doc_id = self._check_job_application_duplicate(user_id, job_data)
            if existing_doc_id:
                print(f"[Firebase] [DUPLICATE] ✓ Duplicate found, skipping save. Existing document ID: {existing_doc_id}")
                print(f"[Firebase] [DUPLICATE] Path: users/{user_id}/job_applications/{existing_doc_id}")
                print(f"{'='*70}\n")
                return existing_doc_id
            
            print(f"[Firebase] [DUPLICATE] No duplicate found, proceeding with save...")
            print(f"[Firebase] [INFO] Saving job application for user: {user_id}")
            
            # Prepare document data with defaults
            # Use datetime.now() directly EXACTLY like test_firebase_simple.py (lines 88-98)
            document_data = {
                "appliedDate": job_data.get("appliedDate", datetime.now()),
                "company": job_data.get("company", ""),
                "createdAt": datetime.now(),  # EXACT same as test script line 90
                "interviewDate": job_data.get("interviewDate", ""),
                "jobDescription": job_data.get("jobDescription", ""),
                "link": job_data.get("link", ""),
                "notes": job_data.get("notes", ""),
                "portal": job_data.get("portal", "Unknown"),
                "role": job_data.get("role", ""),
                "status": job_data.get("status", "Applied"),
                "visaRequired": job_data.get("visaRequired", "No")
            }
            
            print(f"[Firebase] [DEBUG] Document data prepared:")
            print(f"  Company: {document_data['company']}")
            print(f"  Role: {document_data['role']}")
            print(f"  Portal: {document_data['portal']}")
            print(f"  appliedDate type: {type(document_data['appliedDate'])}")
            print(f"  createdAt type: {type(document_data['createdAt'])}")
            print(f"[Firebase] [DEBUG] Full document_data: {json.dumps({k: str(v) if isinstance(v, datetime) else v for k, v in document_data.items()}, indent=2)}")
            
            # Reference to job_applications subcollection
            # EXACT same approach as test_firebase_simple.py line 83
            print(f"[Firebase] [DEBUG] Creating collection reference...")
            collection_ref = db.collection("users").document(user_id).collection("job_applications")
            print(f"[Firebase] [OK] Collection reference created: users/{user_id}/job_applications")
            print(f"[Firebase] [DEBUG] Collection ref type: {type(collection_ref)}")
            
            # Use add() method which returns (timestamp, document_reference)
            # EXACT same approach as test_firebase_simple.py line 109
            print(f"[Firebase] [SAVE] Calling collection_ref.add(document_data)...")
            print(f"[Firebase] [DEBUG] About to save document...")
            
            result = collection_ref.add(document_data)
            
            print(f"[Firebase] [DEBUG] add() returned: {result}")
            print(f"[Firebase] [DEBUG] Result type: {type(result)}")
            
            # Handle return value - EXACT same logic as test_firebase_simple.py (lines 112-117)
            if isinstance(result, tuple):
                update_time, doc_ref = result
                doc_id = doc_ref.id
                print(f"[Firebase] [DEBUG] Result is tuple: update_time={update_time}, doc_ref.id={doc_id}")
            else:
                doc_ref = result
                doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
                print(f"[Firebase] [DEBUG] Result is single object: doc_id={doc_id}")
            
            print(f"[Firebase] [SUCCESS] Document added with ID: {doc_id}")
            print(f"[Firebase] [PATH] users/{user_id}/job_applications/{doc_id}")
            
            # Verify the document was saved - EXACT same as test_firebase_simple.py (lines 125-135)
            print(f"[Firebase] [VERIFY] Verifying document was saved...")
            doc = collection_ref.document(doc_id).get()
            
            if doc.exists:
                print(f"[Firebase] [OK] Verification successful! Document exists.")
                doc_dict = doc.to_dict()
                print(f"[Firebase] [DEBUG] Document contents:")
                for key, value in doc_dict.items():
                    print(f"  {key}: {value} (type: {type(value)})")
            else:
                print(f"[Firebase] [WARNING] Document not found after saving")
                print(f"[Firebase] [ERROR] This indicates the save operation failed silently!")
            
            print(f"{'='*70}\n")
            return doc_id
                
        except Exception as e:
            import traceback
            error_msg = f"Failed to save job application for user {user_id}: {str(e)}"
            print(f"[Firebase] [ERROR] {error_msg}")
            print(f"[Firebase] [TRACEBACK]:")
            print(traceback.format_exc())
            print(f"{'='*70}\n")
            raise RuntimeError(error_msg)

    def save_job_applications_batch(self, user_id: str, jobs_data: List[Dict[str, Any]]) -> List[str]:
        """
        Save multiple job applications to Firestore in a batch.
        Duplicates are automatically skipped (existing document IDs are returned).
        
        Args:
            user_id: The user ID
            jobs_data: List of dictionaries containing job application data
            
        Returns:
            List of document IDs of saved applications (includes existing duplicates)
        """
        try:
            print(f"[Firebase] [BATCH] Starting batch save for {len(jobs_data)} job applications...")
            document_ids = []
            duplicates_count = 0
            new_count = 0
            
            for idx, job_data in enumerate(jobs_data, 1):
                print(f"\n[Firebase] [BATCH] Processing job {idx}/{len(jobs_data)}...")
                try:
                    # Check if this is a duplicate before attempting save
                    existing_doc_id = self._check_job_application_duplicate(user_id, job_data)
                    if existing_doc_id:
                        document_ids.append(existing_doc_id)
                        duplicates_count += 1
                        print(f"[Firebase] [BATCH] ⚠️  Duplicate found for job {idx}, skipping save. Existing ID: {existing_doc_id}")
                    else:
                        doc_id = self.save_job_application(user_id, job_data)
                        document_ids.append(doc_id)
                        new_count += 1
                        print(f"[Firebase] [BATCH] ✓ Successfully saved job {idx} with ID: {doc_id}")
                except Exception as job_error:
                    print(f"[Firebase] [BATCH] [ERROR] Failed to save job {idx}: {str(job_error)}")
                    import traceback
                    print(f"[Firebase] [BATCH] Traceback: {traceback.format_exc()}")
                    # Continue with other jobs even if one fails
                    continue
            
            print(f"\n[Firebase] [BATCH] Completed: {len(document_ids)}/{len(jobs_data)} jobs processed")
            print(f"[Firebase] [BATCH] - New: {new_count}, Duplicates: {duplicates_count}, Errors: {len(jobs_data) - len(document_ids)}")
            return document_ids
            
        except Exception as e:
            print(f"[Firebase] [BATCH] [ERROR] Batch save failed: {str(e)}")
            import traceback
            print(f"[Firebase] [BATCH] Traceback: {traceback.format_exc()}")
            raise RuntimeError(f"Failed to save job applications batch for user {user_id}: {str(e)}")

    def _check_sponsorship_duplicate(self, user_id: str, company_name: str, request_id: Optional[str] = None) -> Optional[str]:
        """
        Check if a sponsorship check already exists for this user and company.
        
        Args:
            user_id: The user ID
            company_name: The company name
            request_id: Optional request ID to check (if provided, checks for exact match)
            
        Returns:
            Document ID of existing duplicate if found, None otherwise
        """
        try:
            db = FirebaseService._db
            if db is None:
                return None
            
            collection_ref = db.collection("sponsorship_checks").document(user_id).collection("checks")
            
            # Strategy 1: Check by request_id (most specific - same match-jobs request)
            if request_id:
                query = collection_ref.where("requestId", "==", request_id).limit(1)
                docs = query.stream()
                for doc in docs:
                    print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Found duplicate sponsorship by request_id: {doc.id}")
                    return doc.id
            
            # Strategy 2: Check by company name (same company checked before)
            if company_name and company_name.strip():
                company_name_clean = company_name.strip()
                query = collection_ref.where("companyName", "==", company_name_clean).limit(1)
                docs = query.stream()
                for doc in docs:
                    print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Found duplicate sponsorship by company: {doc.id}")
                    return doc.id
            
            return None
            
        except Exception as e:
            print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Error checking for duplicate: {e}")
            # Don't fail on duplicate check errors, just log and continue
            return None
    
    def save_sponsorship_info(self, user_id: str, request_id: str, sponsorship_data: Dict[str, Any], job_info: Optional[Dict[str, Any]] = None) -> str:
        """
        Save sponsorship information to Firestore.
        Checks for duplicates before saving.
        
        Args:
            user_id: The user ID
            request_id: The request ID (unique per match-jobs call)
            sponsorship_data: Dictionary containing sponsorship data
            job_info: Optional job information dictionary
            
        Returns:
            The document ID of the saved sponsorship (or existing duplicate)
        """
        try:
            print(f"\n{'='*70}")
            print(f"[Firebase] [SPONSORSHIP] Starting save_sponsorship_info")
            print(f"{'='*70}")
            
            # CRITICAL: Ensure Firebase app is initialized - EXACT same as save_job_application
            if FirebaseService._app is None:
                print("[Firebase] [SPONSORSHIP] [WARNING] Firebase app is None, re-initializing...")
                self._initialize_firebase()
            
            # CRITICAL: Ensure Firestore client is initialized - EXACT same as save_job_application
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Checking Firestore client...")
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] FirebaseService._db is None: {FirebaseService._db is None}")
            
            if FirebaseService._db is None:
                print("[Firebase] [SPONSORSHIP] [WARNING] Firestore client is None, creating new client...")
                FirebaseService._db = firestore.client()
                print("[Firebase] [SPONSORSHIP] [OK] Firestore client created")
            else:
                print("[Firebase] [SPONSORSHIP] [OK] Using existing Firestore client")
                try:
                    test_ref = FirebaseService._db.collection("users")
                    print(f"[Firebase] [SPONSORSHIP] [OK] Client is usable (type: {type(FirebaseService._db)})")
                except Exception as test_error:
                    print(f"[Firebase] [SPONSORSHIP] [ERROR] Client not usable: {test_error}")
                    print("[Firebase] [SPONSORSHIP] [WARNING] Recreating client...")
                    FirebaseService._db = firestore.client()
                    print("[Firebase] [SPONSORSHIP] [OK] New Firestore client created")
            
            # Get a direct reference to the db client - EXACT same as save_job_application
            db = FirebaseService._db
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Using db client: {type(db)}")
            
            # CRITICAL: Verify Firebase app credentials are working by testing a read operation
            # This ensures authentication is properly configured
            try:
                print(f"[Firebase] [SPONSORSHIP] [AUTH] Verifying Firebase authentication...")
                test_collection = db.collection("users")
                # Try to get a document (this will fail if auth is wrong)
                test_collection.limit(1).get()
                print(f"[Firebase] [SPONSORSHIP] [AUTH] ✓ Authentication verified successfully")
            except Exception as auth_error:
                print(f"[Firebase] [SPONSORSHIP] [AUTH] [ERROR] Authentication verification failed: {auth_error}")
                print(f"[Firebase] [SPONSORSHIP] [AUTH] This may indicate credential/permission issues")
                # Don't fail here, but log the warning - the actual save will show the real error
            
            # Extract company name for duplicate check and document preparation
            company_name = job_info.get("company") if job_info else sponsorship_data.get("company_name", "")
            
            # Check for duplicates before saving
            print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Checking for duplicate sponsorship check...")
            existing_doc_id = self._check_sponsorship_duplicate(user_id, company_name, request_id)
            if existing_doc_id:
                print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] ✓ Duplicate found, skipping save. Existing document ID: {existing_doc_id}")
                print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] Path: sponsorship_checks/{user_id}/checks/{existing_doc_id}")
                print(f"{'='*70}\n")
                return existing_doc_id
            
            print(f"[Firebase] [SPONSORSHIP] [DUPLICATE] No duplicate found, proceeding with save...")
            print(f"[Firebase] [SPONSORSHIP] [INFO] Saving sponsorship info for user: {user_id}")
            portal = job_info.get("portal", "") if job_info else ""
            job_url = job_info.get("job_url", "") if job_info else ""
            
            # Prepare document data with FLAT structure (same as job_applications)
            # This ensures consistency and avoids any potential nested object serialization issues
            document_data = {
                "requestId": request_id,
                "companyName": company_name or "",
                "portal": portal if portal else "",
                "website": job_url if job_url else "",
                "sponsorsWorkers": bool(sponsorship_data.get("sponsors_workers", False)),
                "visaTypes": sponsorship_data.get("visa_types", "") or "",
                "summary": sponsorship_data.get("summary", "") or "",
                "createdAt": datetime.now(),  # EXACT same as job_applications
            }
            
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Document data prepared:")
            print(f"  Request ID: {document_data['requestId']}")
            print(f"  Company Name: {document_data['companyName']}")
            print(f"  Sponsors Workers: {document_data['sponsorsWorkers']}")
            print(f"  User ID: {user_id}")
            print(f"  createdAt type: {type(document_data['createdAt'])}")
            
            # Helper function to serialize datetime for JSON
            def serialize_for_json(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                elif isinstance(obj, dict):
                    return {k: serialize_for_json(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [serialize_for_json(item) for item in obj]
                return obj
            
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Full document_data: {json.dumps(serialize_for_json(document_data), indent=2)}")
            
            # Reference to sponsorship_checks/{user_id}/{doc_id} structure
            # Since Firestore doesn't allow collections directly under collections,
            # we'll use a subcollection structure where user_id appears as a collection-like grouping
            # Structure: sponsorship_checks (top-level collection) > {user_id} (document) > checks (subcollection) > {doc_id} (document)
            # This allows us to group documents by user_id while maintaining Firestore's data model
            
            # First, ensure the user_id document exists in sponsorship_checks (this acts as a "collection" grouping)
            user_doc_ref = db.collection("sponsorship_checks").document(user_id)
            user_doc = user_doc_ref.get()
            if not user_doc.exists:
                print(f"[Firebase] [SPONSORSHIP] [INFO] Creating user grouping document: sponsorship_checks/{user_id}")
                user_doc_ref.set({"userId": user_id, "createdAt": datetime.now()}, merge=True)
            
            # Create subcollection under user_id document
            # Using "checks" as the subcollection name (can be changed to any name)
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Creating collection reference...")
            collection_ref = db.collection("sponsorship_checks").document(user_id).collection("checks")
            print(f"[Firebase] [SPONSORSHIP] [OK] Collection reference created: sponsorship_checks/{user_id}/checks")
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Collection ref type: {type(collection_ref)}")
            print(f"[Firebase] [SPONSORSHIP] [NOTE] Path structure: sponsorship_checks/{user_id}/_/<doc_id>")
            print(f"[Firebase] [SPONSORSHIP] [NOTE] In console: sponsorship_checks > {user_id} > _ > <doc_id>")
            
            # Use add() method which returns (timestamp, document_reference)
            print(f"[Firebase] [SPONSORSHIP] [SAVE] Calling collection_ref.add(document_data)...")
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] About to save document...")
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Document structure: FLAT (same as job_applications)")
            print(f"[Firebase] [SPONSORSHIP] [DEBUG] Collection path: sponsorship_checks/{user_id}/checks")
            
            # CRITICAL: Save the document and get the result
            try:
                result = collection_ref.add(document_data)
                print(f"[Firebase] [SPONSORSHIP] [DEBUG] add() returned: {result}")
                print(f"[Firebase] [SPONSORSHIP] [DEBUG] Result type: {type(result)}")
            except Exception as save_error:
                # Catch permission errors specifically
                error_str = str(save_error).lower()
                if "permission" in error_str or "denied" in error_str or "403" in error_str:
                    print(f"[Firebase] [SPONSORSHIP] [ERROR] Permission denied! Check Firestore security rules.")
                    print(f"[Firebase] [SPONSORSHIP] [ERROR] Service account may not have write permissions.")
                    print(f"[Firebase] [SPONSORSHIP] [ERROR] Error details: {save_error}")
                print(f"[Firebase] [SPONSORSHIP] [ERROR] Save operation failed with exception: {save_error}")
                import traceback
                print(f"[Firebase] [SPONSORSHIP] [TRACEBACK]: {traceback.format_exc()}")
                raise RuntimeError(f"Failed to save sponsorship document: {save_error}")
            
            # Handle return value - EXACT same logic as save_job_application
            if isinstance(result, tuple):
                update_time, doc_ref = result
                doc_id = doc_ref.id
                print(f"[Firebase] [SPONSORSHIP] [DEBUG] Result is tuple: update_time={update_time}, doc_ref.id={doc_id}")
            else:
                doc_ref = result
                doc_id = doc_ref.id if hasattr(doc_ref, 'id') else str(doc_ref)
                print(f"[Firebase] [SPONSORSHIP] [DEBUG] Result is single object: doc_id={doc_id}")
            
            if not doc_id:
                raise RuntimeError("Document ID is None or empty after save operation")
            
            print(f"[Firebase] [SPONSORSHIP] [SUCCESS] Document added with ID: {doc_id}")
            print(f"[Firebase] [SPONSORSHIP] [PATH] sponsorship_checks/{user_id}/checks/{doc_id}")
            print(f"[Firebase] [SPONSORSHIP] [NOTE] In Firebase Console, navigate to: sponsorship_checks > {user_id} > checks > {doc_id}")
            
            # Verify the document was saved with retry logic
            print(f"[Firebase] [SPONSORSHIP] [VERIFY] Verifying document was saved...")
            import time
            doc = None
            for attempt in range(3):
                try:
                    doc = collection_ref.document(doc_id).get()
                    if doc.exists:
                        break
                    if attempt < 2:
                        print(f"[Firebase] [SPONSORSHIP] [VERIFY] Attempt {attempt + 1} failed, retrying in 0.5s...")
                        time.sleep(0.5)
                except Exception as verify_error:
                    print(f"[Firebase] [SPONSORSHIP] [VERIFY] Error on attempt {attempt + 1}: {verify_error}")
                    if attempt < 2:
                        time.sleep(0.5)
            
            if doc and doc.exists:
                print(f"[Firebase] [SPONSORSHIP] [OK] Verification successful! Document exists.")
                doc_dict = doc.to_dict()
                print(f"[Firebase] [SPONSORSHIP] [DEBUG] Document contents:")
                for key, value in doc_dict.items():
                    print(f"  {key}: {value} (type: {type(value)})")
                print(f"[Firebase] [SPONSORSHIP] [FINAL] Document successfully saved and verified at: sponsorship_checks/{user_id}/checks/{doc_id}")
            else:
                error_msg = f"Document not found after saving (doc_id: {doc_id})"
                print(f"[Firebase] [SPONSORSHIP] [WARNING] {error_msg}")
                print(f"[Firebase] [SPONSORSHIP] [ERROR] This indicates the save operation failed silently!")
                # Don't raise here - let the caller handle it, but log clearly
                print(f"[Firebase] [SPONSORSHIP] [INFO] Check Firestore Console at: sponsorship_checks/{user_id}/checks/{doc_id}")
            
            print(f"{'='*70}\n")
            return doc_id
                
        except Exception as e:
            import traceback
            error_msg = f"Failed to save sponsorship info for user {user_id}: {str(e)}"
            print(f"[Firebase] [SPONSORSHIP] [ERROR] {error_msg}")
            print(f"[Firebase] [SPONSORSHIP] [TRACEBACK]:")
            print(traceback.format_exc())
            print(f"{'='*70}\n")
            raise RuntimeError(error_msg)


# Singleton instance
_firebase_service: Optional[FirebaseService] = None


def get_firebase_service() -> FirebaseService:
    """Get or create the Firebase service instance."""
    global _firebase_service
    if _firebase_service is None:
        _firebase_service = FirebaseService()
    return _firebase_service