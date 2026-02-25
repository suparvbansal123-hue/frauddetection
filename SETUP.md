# FraudShield AI — Firebase + Render Setup Guide

## What's included

| File | Purpose |
|------|---------|
| `firebase_db.py` | Firebase Admin SDK init, token verification, `UserStore`, `SessionStore` |
| `app.py` | Flask backend with 3 new auth endpoints wired to Firestore |
| `index.html` | Frontend with full Firebase Auth (email/password + Google SSO) |
| `requirements.txt` | Python deps incl. `firebase-admin` and `gunicorn` |
| `render.yaml` | Render.com deploy config |

---

## Firestore Data Model

```
users/{uid}
  uid:          string
  email:        string
  display_name: string
  role:         "analyst" | "admin"
  run_count:    number       ← auto-incremented on each detection run
  created_at:   timestamp
  last_login:   timestamp

sessions/{auto-id}
  uid:          string
  email:        string
  ip:           string
  user_agent:   string
  created_at:   timestamp
```

---

## Step 1 — Enable Firebase Auth providers

1. Go to **Firebase Console → Authentication → Sign-in method**
2. Enable **Email/Password**
3. Enable **Google** (set your support email)

---

## Step 2 — Get Service Account credentials

1. Firebase Console → **Project Settings** (gear icon) → **Service Accounts**
2. Click **"Generate new private key"** → download JSON
3. Open the JSON — you need these fields:
   - `project_id`
   - `private_key_id`
   - `private_key`  ← the whole `-----BEGIN RSA PRIVATE KEY-----...-----END RSA PRIVATE KEY-----` block
   - `client_email`
   - `client_id`
   - `client_x509_cert_url`

---

## Step 3 — Enable Firestore

1. Firebase Console → **Firestore Database** → **Create database**
2. Choose **"Start in production mode"** → pick a region
3. Set security rules (paste these):

```
rules_version = '2';
service cloud.firestore {
  match /databases/{database}/documents {
    // Users can only read/write their own profile
    match /users/{uid} {
      allow read, write: if request.auth != null && request.auth.uid == uid;
    }
    // Sessions: backend only (Firebase Admin bypasses rules)
    match /sessions/{sessionId} {
      allow read: if request.auth != null;
      allow write: if false;  // only backend (Admin SDK) can write
    }
  }
}
```

---

## Step 4 — Local development

```bash
# Clone & install
pip install -r requirements.txt

# Set env vars (create a .env file)
export FIREBASE_PROJECT_ID="frauddetectorog"
export FIREBASE_PRIVATE_KEY_ID="your-key-id"
export FIREBASE_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----\n"
export FIREBASE_CLIENT_EMAIL="firebase-adminsdk-xxx@frauddetectorog.iam.gserviceaccount.com"
export FIREBASE_CLIENT_ID="your-client-id"
export FIREBASE_CLIENT_CERT_URL="https://www.googleapis.com/robot/v1/metadata/x509/..."

python app.py
# → http://localhost:5000
```

> **Tip:** Use `python-dotenv` and a `.env` file locally — just don't commit it.

---

## Step 5 — Deploy to Render.com

1. Push your code to **GitHub** (make sure `.env` is in `.gitignore`)
2. Go to [render.com](https://render.com) → **New → Web Service**
3. Connect your GitHub repo
4. Render detects `render.yaml` and sets up the service
5. In **Environment → Environment Variables**, add each of the 6 Firebase vars:

| Key | Value |
|-----|-------|
| `FIREBASE_PROJECT_ID` | `frauddetectorog` |
| `FIREBASE_PRIVATE_KEY_ID` | from service account JSON |
| `FIREBASE_PRIVATE_KEY` | the full key block (with literal `\n` — Render stores it correctly) |
| `FIREBASE_CLIENT_EMAIL` | `firebase-adminsdk-xxx@...iam.gserviceaccount.com` |
| `FIREBASE_CLIENT_ID` | from service account JSON |
| `FIREBASE_CLIENT_CERT_URL` | from service account JSON |

6. Click **Deploy** — Render builds and starts your app
7. Note your URL: `https://fraudshield-api.onrender.com`

---

## Step 6 — Update frontend to point at your Render URL

In `index.html`, find this line near the top of the non-module `<script>`:

```js
const API = window.BACKEND_URL || 'http://localhost:5000';
```

Either:
- Set `window.BACKEND_URL = 'https://fraudshield-api.onrender.com'` before the script loads, OR
- Replace `'http://localhost:5000'` with your Render URL directly

Also add your Render domain to Firebase Auth's **Authorised domains**:
Firebase Console → Authentication → Settings → Authorised domains → **Add domain**

---

## New API Endpoints

```
POST /api/auth/session    Verify Firebase ID token, upsert Firestore profile, log session
                          Body: { "idToken": "..." }

GET  /api/auth/me         Return current user's Firestore profile
                          Header: Authorization: Bearer <idToken>

GET  /api/auth/sessions   Return last 10 sign-in sessions for current user
                          Header: Authorization: Bearer <idToken>
```

All original endpoints (`/api/detect`, `/api/monitor`, `/api/sample`, `/health`) are unchanged.
`/api/detect` optionally accepts a Bearer token to increment `run_count` in Firestore.

---

## Auth flow diagram

```
Frontend                    Firebase Auth              Backend (Render)          Firestore
   │                              │                          │                      │
   │─── signInWithEmailAndPassword ──▶│                      │                      │
   │◀────────── idToken ─────────────│                      │                      │
   │                                 │                      │                      │
   │──── POST /api/auth/session ──────────────────────────▶│                      │
   │            { idToken }          │                      │──verify_id_token()──▶│ (Firebase Admin)
   │                                 │                      │◀─── decoded token ───│
   │                                 │                      │──── upsert user ────▶│ users/{uid}
   │                                 │                      │──── log session ────▶│ sessions/{}
   │◀────────── { user profile } ────────────────────────────                      │
```
