"""
╔══════════════════════════════════════════════════════════════╗
║   FraudShield AI — Firebase Auth & Firestore Layer           ║
║   firebase_admin SDK · ID Token verification · Sessions      ║
╚══════════════════════════════════════════════════════════════╝

ENV VARS REQUIRED:
    FIREBASE_PROJECT_ID       → from Firebase console
    FIREBASE_PRIVATE_KEY_ID   → from service account JSON
    FIREBASE_PRIVATE_KEY      → from service account JSON (with literal \\n)
    FIREBASE_CLIENT_EMAIL     → from service account JSON
    FIREBASE_CLIENT_ID        → from service account JSON
    FIREBASE_CLIENT_CERT_URL  → from service account JSON

HOW TO GET SERVICE ACCOUNT JSON:
    Firebase Console → Project Settings → Service Accounts
    → Generate new private key → download JSON
    → Copy values into Render env vars (see render.yaml)
"""

import os
import json
import functools
from datetime import datetime, timezone

import firebase_admin
from firebase_admin import credentials, auth, firestore

# ─── Init Firebase Admin (runs once at import) ────────────────────────────────

def _build_credentials():
    """
    Build credentials from individual env vars (Render-friendly).
    Avoids storing a JSON file on disk — all secrets stay in env vars.
    """
    private_key = os.environ.get('FIREBASE_PRIVATE_KEY', '')
    # Render stores \\n literally — convert to real newlines
    private_key = private_key.replace('\\n', '\n')

    service_account = {
        "type": "service_account",
        "project_id":                  os.environ['FIREBASE_PROJECT_ID'],
        "private_key_id":              os.environ['FIREBASE_PRIVATE_KEY_ID'],
        "private_key":                 private_key,
        "client_email":                os.environ['FIREBASE_CLIENT_EMAIL'],
        "client_id":                   os.environ['FIREBASE_CLIENT_ID'],
        "auth_uri":                    "https://accounts.google.com/o/oauth2/auth",
        "token_uri":                   "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url":        os.environ.get('FIREBASE_CLIENT_CERT_URL', ''),
    }
    return credentials.Certificate(service_account)


def init_firebase():
    """
    Call once in app.py:  init_firebase()
    Safe to call multiple times — skips if already initialised.
    Returns the Firestore client.
    """
    if not firebase_admin._apps:
        try:
            cred = _build_credentials()
            firebase_admin.initialize_app(cred)
            print("✓  Firebase Admin initialised")
        except KeyError as e:
            print(f"⚠  Firebase env var missing: {e} — running in DEMO mode (no auth)")
            # Initialise with no credentials so the app still starts
            firebase_admin.initialize_app()
        except Exception as e:
            print(f"⚠  Firebase init failed: {e} — running in DEMO mode")
            firebase_admin.initialize_app()

    try:
        return firestore.client()
    except Exception:
        return None


# ─── Token verification ───────────────────────────────────────────────────────

def verify_id_token(id_token: str) -> dict | None:
    """
    Verify a Firebase ID token sent from the frontend.
    Returns decoded token dict on success, None on failure.

    Frontend sends:  Authorization: Bearer <idToken>
    """
    try:
        decoded = auth.verify_id_token(id_token)
        return decoded
    except Exception as e:
        print(f"Token verification failed: {e}")
        return None


def require_auth(f):
    """
    Flask route decorator — rejects requests without a valid Firebase ID token.

    Usage:
        @app.route('/api/protected')
        @require_auth
        def protected(firebase_user):
            return jsonify({'uid': firebase_user['uid']})
    """
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        from flask import request, jsonify

        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing Authorization header'}), 401

        id_token = auth_header.split('Bearer ', 1)[1].strip()
        decoded  = verify_id_token(id_token)
        if not decoded:
            return jsonify({'error': 'Invalid or expired token'}), 401

        return f(*args, firebase_user=decoded, **kwargs)
    return wrapper


# ─── Firestore helpers ────────────────────────────────────────────────────────

class UserStore:
    """
    CRUD helpers for the `users` collection in Firestore.

    Schema (each doc keyed by Firebase UID):
    {
        uid:          string,
        email:        string,
        display_name: string,
        role:         "analyst" | "admin",
        created_at:   timestamp,
        last_login:   timestamp,
        run_count:    number      ← incremented on each /api/detect call
    }
    """

    COLLECTION = 'users'

    def __init__(self, db_client):
        self._db = db_client

    def _col(self):
        return self._db.collection(self.COLLECTION)

    def get(self, uid: str) -> dict | None:
        """Fetch user profile by UID."""
        try:
            doc = self._col().document(uid).get()
            return doc.to_dict() if doc.exists else None
        except Exception as e:
            print(f"UserStore.get error: {e}")
            return None

    def upsert(self, uid: str, data: dict) -> bool:
        """Create or merge-update a user document."""
        try:
            data['updated_at'] = datetime.now(timezone.utc)
            self._col().document(uid).set(data, merge=True)
            return True
        except Exception as e:
            print(f"UserStore.upsert error: {e}")
            return False

    def on_login(self, uid: str, email: str, display_name: str = '') -> dict:
        """
        Called after token verification succeeds.
        Creates profile on first login, updates last_login on subsequent logins.
        Returns the user profile dict.
        """
        existing = self.get(uid)
        if not existing:
            # First login — create profile
            profile = {
                'uid':          uid,
                'email':        email,
                'display_name': display_name or email.split('@')[0].title(),
                'role':         'analyst',
                'run_count':    0,
                'created_at':   datetime.now(timezone.utc),
                'last_login':   datetime.now(timezone.utc),
            }
            self.upsert(uid, profile)
            print(f"✓  New user profile created: {email}")
            return profile
        else:
            # Existing user — update last_login
            self.upsert(uid, {'last_login': datetime.now(timezone.utc), 'email': email})
            existing['last_login'] = datetime.now(timezone.utc).isoformat()
            return existing

    def increment_run_count(self, uid: str):
        """Increment run_count when user triggers /api/detect."""
        try:
            from google.cloud.firestore import Increment
            self._col().document(uid).update({'run_count': Increment(1)})
        except Exception as e:
            print(f"UserStore.increment_run_count error: {e}")

    def list_all(self, limit: int = 100) -> list:
        """List all user profiles (admin use)."""
        try:
            docs = self._col().limit(limit).stream()
            return [d.to_dict() for d in docs]
        except Exception as e:
            print(f"UserStore.list_all error: {e}")
            return []


class SessionStore:
    """
    Lightweight session log in Firestore.
    Each sign-in gets a record — useful for audit trails.

    Collection: `sessions`
    Doc ID: auto-generated
    """

    COLLECTION = 'sessions'

    def __init__(self, db_client):
        self._db = db_client

    def record(self, uid: str, email: str, ip: str = '', user_agent: str = ''):
        """Write a session record when a user authenticates."""
        try:
            self._db.collection(self.COLLECTION).add({
                'uid':        uid,
                'email':      email,
                'ip':         ip,
                'user_agent': user_agent[:200],
                'created_at': datetime.now(timezone.utc),
            })
        except Exception as e:
            print(f"SessionStore.record error: {e}")

    def get_user_sessions(self, uid: str, limit: int = 10) -> list:
        """Get recent sessions for a specific user."""
        try:
            docs = (self._db.collection(self.COLLECTION)
                    .where('uid', '==', uid)
                    .order_by('created_at', direction='DESCENDING')
                    .limit(limit)
                    .stream())
            return [d.to_dict() for d in docs]
        except Exception as e:
            print(f"SessionStore.get_user_sessions error: {e}")
            return []
