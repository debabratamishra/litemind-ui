# User Authentication & Per-User Session Isolation Design

**Date**: 2026-07-22
**Status**: Approved
**Author**: Claude Code (via superpowers:brainstorming)

## Overview

Add a user login/registration flow to LiteMindUI using self-hosted Supabase Auth (GoTrue). Each user's session and conversation data must be fully isolated — users cannot access other users' profiles or contents. The system must support multiple clients: web (Next.js), CLI, and future desktop apps.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Auth backend | Supabase Auth (GoTrue), self-hosted | Open-source, JWT-based, scales well, self-hostable |
| Login methods | Email + Password | Simple, universal. Architecture supports adding magic links/social login later |
| Database | Supabase Postgres for users/conversations, ChromaDB for RAG vectors | Unified data layer, proper foreign keys, fixes SQLite wipe-on-restart bug |
| Session management | Hybrid — JWT in response body + HTTP-only cookie for web | Secure for browsers, flexible for CLI/desktop clients |
| Data migration | Clean break to Supabase Postgres | Existing SQLite data is dev-only; clean migration is simpler |

## Architecture

```
┌─────────────────┐     ┌──────────────────────┐
│   Next.js Web   │     │   litemind-cli /     │
│   (port 3000)   │     │   litemind-desktop   │
└────────┬────────┘     └──────────┬───────────┘
         │                         │
         │  HTTP (Bearer Token)    │  HTTP (Bearer Token)
         ▼                         ▼
┌──────────────────────────────────────────────┐
│              FastAPI Backend (port 8000)      │
│                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌───────┐ │
│  │  Auth Router │  │ Chat Router │  │ RAG   │ │
│  │  (GoTrue)    │  │  (w/ auth)  │  │ Router│ │
│  └──────┬──────┘  └──────┬──────┘  └───┬───┘ │
│         │                │             │     │
│         ▼                ▼             ▼     │
│  ┌─────────────┐  ┌─────────────┐  ┌───────┐ │
│  │ Supabase    │  │ Conversation│  │ Chroma│ │
│  │ Postgres    │  │ Memory      │  │ DB    │ │
│  │ (users,     │  │ (per-user)  │  │ (RAG) │ │
│  │  convos)    │  │             │  │       │ │
│  └─────────────┘  └─────────────┘  └───────┘ │
└──────────────────────────────────────────────┘
```

**Key principle**: The backend is the single source of truth for auth. All clients authenticate via the same API.

## Backend Changes

### New Dependencies (`pyproject.toml`)

```toml
gotrue >= 3.0.0                          # Supabase GoTrue client (auth + JWT verification)
python-jose[cryptography] >= 3.3.0       # JWT handling (fallback if GoTrue verification is insufficient)
passlib[bcrypt] >= 2.0.0                 # Password hashing (fallback)
asyncpg >= 0.29.0                        # Async PostgreSQL adapter (matches backend's async patterns)
```

### New Auth Router (`app/backend/api/auth.py`)

**Endpoints**:
- `POST /api/auth/register` — Register with email + password
- `POST /api/auth/login` — Login, returns JWT + sets HTTP-only cookie
- `POST /api/auth/logout` — Clears cookie, invalidates session
- `GET /api/auth/me` — Returns current user (from JWT)
- `POST /api/auth/refresh` — Refreshes access token (future)

**Login flow**:
1. Frontend POSTs email/password to `/api/auth/login`
2. Backend calls GoTrue to authenticate
3. GoTrue returns JWT
4. Backend sets HTTP-only cookie (for web browsers)
5. Backend returns JWT in response body (for CLI/desktop)
6. Subsequent requests use either cookie (web) or `Authorization: Bearer <token>` (CLI/desktop)

### Auth Dependency (`app/backend/api/auth_deps.py`)

FastAPI dependency that extracts and validates JWT from:
1. `Authorization: Bearer <token>` header (CLI/desktop)
2. `access_token` HTTP-only cookie (web)

```python
def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    access_token: str = Cookie(None),
) -> User:
    token = credentials.credentials if credentials else access_token
    if not token:
        raise HTTPException(401, "Not authenticated")
    user_data = verify_jwt(token)
    return User(**user_data)
```

### Apply Auth to Existing Endpoints

Add `Depends(get_current_user)` to:
- `POST /api/chat/stream` — Chat endpoint
- `POST /api/rag/query` — RAG query endpoint
- `POST /api/voice/offer` — Voice pipeline endpoint

The user ID from JWT namespaces conversations:
```python
# Before (no user context):
session = memory_service.get_or_create_session(session_id)

# After (user-scoped):
user_session_id = f"{user.id}:{session_id}"
session = memory_service.get_or_create_session(user_session_id)
```

### Database Migration to Supabase Postgres

Replace `conversation_db.py` with a Postgres-backed implementation.

**Schema**:
```sql
-- Users table (synced from GoTrue)
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Conversations table (user-scoped)
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    conversation_type TEXT DEFAULT 'chat',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    summary TEXT
);

-- Messages table (user-scoped via conversation FK)
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
```

**Key changes**:
- Remove `clear_database_file()` call (fixes the wipe-on-restart bug)
- Add `user_id` foreign key to `conversations` table
- All queries filter by `user_id` for isolation
- Use `psycopg2` for Postgres connectivity

### Conversation Memory Isolation

Modify `ConversationMemoryService` to accept `user_id`:

```python
class ConversationMemoryService:
    def get_or_create_session(self, user_id: str, session_id: str) -> ConversationContext:
        scoped_session_id = f"{user_id}:{session_id}"
        return self._sessions.get(scoped_session_id) or self._create_session(scoped_session_id)
```

This ensures User A's sessions are completely isolated from User B's sessions.

## Frontend Changes

### Auth Store (Zustand)

Add a new auth slice to `src/lib/store.ts`:

```typescript
interface AuthState {
  user: User | null;
  accessToken: string | null;
  isLoading: boolean;
  isAuthenticated: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  fetchCurrentUser: () => Promise<void>;
}
```

The store persists the token in memory (not localStorage for security). On page refresh, it calls `/api/auth/me` to rehydrate the session.

### Auth Context Provider

Create `src/app/auth-provider.tsx` that wraps the root layout. On mount, calls `fetchCurrentUser()` to check if the session is valid.

### Login & Register Pages

New routes:
- `src/app/login/page.tsx` — Login form with email/password
- `src/app/register/page.tsx` — Registration form

**Login page design**:
- Email + password fields
- "Remember me" checkbox (extends cookie max-age)
- "Sign up" link → redirects to register
- Social login buttons (Google, GitHub) — disabled initially, enabled later

### Protected Route Wrapper

Create `src/components/auth/protected-route.tsx` that redirects to `/login` if not authenticated. Apply to `/chat` and `/rag` pages.

### API Client Updates

Modify `src/lib/api.ts` to inject the JWT:

```typescript
const { accessToken } = useAuthStore.getState();
const response = await fetch(`${API_URL}/chat/stream`, {
  method: 'POST',
  headers: {
    'Authorization': `Bearer ${accessToken}`,
    'Content-Type': 'application/json',
  },
  body: JSON.stringify(request),
});
```

### Sidebar Auth Integration

Add to the sidebar:
- If authenticated: User avatar/name + "Sign out" button
- If not authenticated: "Sign in" button → redirects to `/login`

### Session Management

- **Web**: HTTP-only cookie is set by the backend. The frontend doesn't manage it directly.
- **Token refresh**: Frontend calls `/api/auth/me` on load to check session validity. If expired, redirects to login.
- **CLI/Desktop**: Token stored in response body, managed by client application.

## Self-Hosting Setup

### Docker Compose Integration

Add GoTrue and PostgreSQL to `docker-compose.yml`:

```yaml
services:
  gotrue:
    image: supabase/gotrue:v100.0.0
    environment:
      GOTRUE_DB_DRIVER: postgres
      GOTRUE_DB_DATABASE_URL: postgresql://postgres:${POSTGRES_PASSWORD}@db:5432/postgres
      GOTRUE_API_EXTERNAL_URL: http://localhost:9999
      GOTRUE_JWT_SECRET: ${GOTRUE_JWT_SECRET}
      GOTRUE_SITE_URL: http://localhost:3000
      GOTRUE_URI_ALLOW_LIST: http://localhost:3000,http://localhost:8501
      GOTRUE_DISABLE_SIGNUP: false
      GOTRUE_SMTP_HOST: ${SMTP_HOST}
      GOTRUE_SMTP_PORT: ${SMTP_PORT}
      GOTRUE_SMTP_USER: ${SMTP_USER}
      GOTRUE_SMTP_PASS: ${SMTP_PASS}
      GOTRUE_SMTP_ADMIN_EMAIL: ${SMTP_ADMIN_EMAIL}
    ports:
      - "9999:9999"
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-postgres}
      POSTGRES_DB: postgres
    volumes:
      - pgdata:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

volumes:
  pgdata:
```

### Standalone Mode

For standalone (non-Docker) mode, provide a `Makefile` target that runs GoTrue binary directly. The backend detects the mode via `AUTH_MODE=docker|standalone` env var and adjusts the GoTrue URL accordingly.

### Environment Variables (`.env.example` updates)

```env
# Supabase Auth / GoTrue
GOTRUE_API_URL=http://localhost:9999
GOTRUE_JWT_SECRET=your-super-secret-jwt-key-change-this
AUTH_MODE=docker  # or "standalone"

# PostgreSQL
POSTGRES_PASSWORD=postgres
POSTGRES_USER=postgres
POSTGRES_DB=postgres

# SMTP (for password reset, email verification)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASS=your-app-password
SMTP_ADMIN_EMAIL=admin@litemind.local
```

## Data Flow

### Login Flow (Web)

```
1. User enters email/password on /login
2. Frontend POSTs to /api/auth/login
3. Backend calls GoTrue to authenticate
4. GoTrue returns JWT
5. Backend sets HTTP-only cookie + returns user info
6. Frontend redirects to /chat
7. Subsequent requests: browser auto-sends cookie
8. Backend validates cookie JWT via get_current_user dependency
```

### Login Flow (CLI/Desktop)

```
1. CLI prompts for email/password
2. CLI POSTs to /api/auth/login
3. Backend calls GoTrue, returns JWT in response body
4. CLI stores JWT in ~/.litemind/auth_token (chmod 600)
5. CLI sends Authorization: Bearer <token> on all requests
```

### Conversation Isolation

```
1. User A sends a chat message
2. Backend extracts user_id from JWT
3. Backend creates scoped session: "userA:session123"
4. ConversationMemoryService uses scoped session
5. ConversationDatabase queries filter by user_id
6. User B cannot access User A's conversations (different user_id)
```

## Error Handling

- **Invalid credentials**: 401 — "Invalid email or password"
- **Email already registered**: 409 — "Email already in use"
- **Expired token**: 401 — "Token expired" — frontend redirects to login
- **Unauthorized access**: 403 — "Access denied"
- **GoTrue unavailable**: 503 — "Auth service unavailable"

## Testing Strategy

### Backend Tests (`tests/`)
- `test_auth.py` — Register, login, logout, token validation
- `test_auth_deps.py` — Auth dependency with valid/invalid/missing tokens
- `test_conversation_isolation.py` — Verify User A can't see User B's conversations

### Frontend Tests (`nextjs-frontend/`)
- `auth.test.tsx` — Login/register form validation, auth store
- `protected-route.test.tsx` — Redirect behavior for unauthenticated users

### Integration Tests
- Full login → chat → logout flow
- Multi-user isolation test (two users, separate conversations)
- Token expiry and refresh

## Migration Plan

1. **Phase 1**: Add GoTrue + Postgres to Docker Compose, create auth router
2. **Phase 2**: Migrate conversation_db.py to Postgres, add user_id columns
3. **Phase 3**: Add auth dependency to chat/rag/voice endpoints
4. **Phase 4**: Build frontend auth pages, context, protected routes
5. **Phase 5**: Update CLI to use JWT auth
6. **Phase 6**: Testing and documentation

## Future Extensibility

The architecture supports adding these later without breaking changes:
- **Magic Link (Email OTP)**: Add `POST /api/auth/magic-link` endpoint, GoTrue handles OTP delivery
- **Social Login (Google, GitHub)**: Add OAuth redirect endpoints, configure providers in GoTrue
- **Refresh Tokens**: Add `POST /api/auth/refresh` endpoint
- **Email Verification**: Enable GoTrue's built-in email confirmation flow
- **Password Reset**: Enable GoTrue's built-in password recovery flow
