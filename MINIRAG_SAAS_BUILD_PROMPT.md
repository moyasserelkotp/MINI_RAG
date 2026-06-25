# 🚀 MINI-RAG → SaaS Business: Complete AI Prompt Guide
> A step-by-step prompt library to build your RAG engine into a full SaaS product.
> Use each prompt with Claude, GPT-4, or any capable AI assistant.
> Follow the phases in order. Each prompt is self-contained and production-focused.

---

## 📋 HOW TO USE THIS FILE

1. Open your AI assistant (Claude recommended)
2. Go to each phase in order
3. Copy the full prompt block
4. Paste it into the AI
5. The AI will generate production-ready code, architecture, or plans
6. Move to the next step only after completing the current one

> ⚠️ Always give the AI context about your stack:
> **Python + FastAPI + MongoDB + Qdrant + Flutter Web**

---

---

# PHASE 0 — MARKET & PRODUCT DECISIONS

---

## Prompt 0.1 — Choose Your Niche and Position

```
You are a SaaS product strategist with 10 years of experience launching B2B AI products.

I have built a production-ready RAG (Retrieval-Augmented Generation) backend called MINI-RAG with the following capabilities:
- FastAPI backend
- MongoDB for metadata and chat history
- Qdrant vector database
- 5-layer memory system (semantic cache, window, summary, entity, vector)
- Multi-provider LLM support: OpenAI, Cohere, Gemini, Llama
- Multi-provider embeddings: Cohere, OpenAI, Gemini, HuggingFace
- Cohere cross-encoder reranking
- Arabic + English multilingual support with RTL-aware prompts
- Docker Compose production deployment
- Prometheus + Grafana observability
- 6 chunking strategies

I want to turn this into a SaaS business targeting business owners and team leads, with a strong focus on the MENA (Middle East and North Africa) Arabic-speaking market.

Please help me:
1. Define the top 3 most viable niche markets I should target first (ranked by ease of selling + revenue potential)
2. For each niche, write:
   - The exact pain point this solves
   - The buyer persona (who pays)
   - The value proposition in one sentence
   - What they compare it to (competitors)
   - Estimated willingness to pay per month
3. Recommend ONE niche to start with and explain why
4. Write a positioning statement for that niche (the "we help X do Y so they can Z" formula)
5. Suggest a product name that works in Arabic and English markets
6. Define 3 pricing tiers with feature limits for that niche

Be specific. Give real examples. Do not be generic.
```

---

## Prompt 0.2 — Define MVP Feature Set

```
You are a senior product manager specializing in AI SaaS products.

I am building an AI knowledge platform called [PRODUCT NAME] based on a RAG engine with:
- Document ingestion (PDF, DOCX, TXT, CSV, MD, HTML)
- Semantic search with reranking
- Multi-turn AI chat with memory
- Arabic + English support
- FastAPI + MongoDB + Qdrant backend

My target market is: [INSERT YOUR CHOSEN NICHE FROM PROMPT 0.1]

Define my MVP (Minimum Viable Product) with:

1. The exact 10 features that must exist before I charge the first customer
2. For each feature, write:
   - User story: "As a [role], I want to [action] so that [outcome]"
   - Acceptance criteria (what done looks like)
   - Estimated build time in days
   - Priority: MUST / SHOULD / NICE TO HAVE
3. List 10 features I should NOT build yet (and why)
4. Define what "version 1.0 launched" means in one paragraph
5. Create a 12-week build roadmap with weekly milestones

Format as a table where appropriate. Be very specific.
```

---

---

# PHASE 1 — MULTI-TENANT SAAS ARCHITECTURE

---

## Prompt 1.1 — Multi-Tenant Database Design

```
You are a senior backend engineer specializing in multi-tenant SaaS systems with MongoDB.

I have an existing single-tenant RAG application with these MongoDB collections:
- projects (project_id, created_at)
- assets (asset_project_id, asset_name, asset_type, asset_size, asset_config)
- data_chunks (chunk_text, chunk_metadata, chunk_order, chunk_project_id, chunk_asset_id)
- chat_sessions (session_id, project_id, summary, message_count)
- chat_messages (session_id, role, text, created_at)

I need to convert this into a full multi-tenant SaaS architecture where:
- Multiple organizations (companies) use the same platform
- Each organization has multiple users with different roles
- Each organization has isolated data (no data leakage between orgs)
- Each organization has usage limits based on their subscription plan
- One user can belong to multiple organizations

Please provide:

1. Complete new MongoDB schema with all collections:
   - organizations
   - users
   - memberships (user ↔ organization relationship)
   - workspaces
   - subscriptions
   - api_keys
   - usage_logs
   - (keep existing collections but updated)

2. For each collection write the full Pydantic schema in Python with:
   - All fields with types
   - Field validators
   - Default values
   - Indexes to create
   - Example document in JSON

3. Show the data isolation strategy:
   - How organization_id is enforced on every query
   - How to prevent cross-tenant data access
   - Vector collection naming convention per organization

4. Write the MongoDB index creation script in Python using Motor (async)

5. Show the migration plan to convert existing single-tenant data to multi-tenant

Use Python, Pydantic v2, Motor async MongoDB driver. Production quality code only.
```

---

## Prompt 1.2 — Organization and Workspace API

```
You are a senior FastAPI engineer building multi-tenant SaaS APIs.

I need to build the organization and workspace management layer for my AI SaaS platform.

Tech stack:
- Python 3.11
- FastAPI
- MongoDB with Motor (async)
- Pydantic v2
- JWT authentication (to be added in next step)

Build complete production-ready code for:

1. Organization CRUD API:
   POST /api/v1/organizations/ — create org
   GET /api/v1/organizations/{org_id} — get org details
   PUT /api/v1/organizations/{org_id} — update org
   DELETE /api/v1/organizations/{org_id} — delete org (soft delete)
   GET /api/v1/organizations/{org_id}/members — list members
   POST /api/v1/organizations/{org_id}/invite — invite user by email
   DELETE /api/v1/organizations/{org_id}/members/{user_id} — remove member

2. Workspace CRUD API:
   POST /api/v1/organizations/{org_id}/workspaces/ — create workspace
   GET /api/v1/organizations/{org_id}/workspaces/ — list workspaces
   GET /api/v1/organizations/{org_id}/workspaces/{workspace_id} — get workspace
   PUT /api/v1/organizations/{org_id}/workspaces/{workspace_id} — update
   DELETE /api/v1/organizations/{org_id}/workspaces/{workspace_id} — delete

3. For each endpoint provide:
   - Full FastAPI route code
   - Pydantic request/response schemas
   - MongoDB query with organization isolation
   - Error handling with proper HTTP status codes
   - Response enums for signal codes

4. Role-based access control (RBAC):
   - Define OWNER, ADMIN, EDITOR, VIEWER roles
   - Write a permission checker dependency
   - Show how to apply it to each endpoint

5. Write unit tests for each endpoint using pytest and httpx

Make the code production-ready with proper logging, error handling, and docstrings.
```

---

## Prompt 1.3 — Vector Database Multi-Tenancy

```
You are a senior ML infrastructure engineer specializing in Qdrant vector databases.

I need to implement multi-tenant isolation in Qdrant for my RAG SaaS platform.

Current setup:
- Qdrant vector database
- One collection per project: "collection_{project_id}"
- FastAPI backend
- Python qdrant-client

I need to convert this to a multi-tenant system where:
- Each organization has completely isolated vector storage
- Collection naming: "{org_id}_{workspace_id}_{project_id}"
- No cross-tenant vector searches possible
- Usage tracking per organization (vector count, storage size)

Please provide:

1. Updated QdrantDBProvider class with:
   - Multi-tenant collection naming
   - Organization-scoped search (can never search across orgs)
   - Batch upsert with organization metadata in payload
   - Delete all vectors for an organization
   - Get vector count and storage stats per organization

2. Collection management:
   - Create collection with proper HNSW config for production
   - Auto-create collection if not exists
   - Collection health check
   - Snapshot and backup per organization

3. Migration script to rename existing collections to new naming convention

4. Usage tracking:
   - Count vectors per organization
   - Storage size estimation per organization
   - Write usage to MongoDB usage_logs collection

5. Performance optimization:
   - Optimal HNSW parameters for different dataset sizes (10K, 100K, 1M vectors)
   - Batch size recommendations
   - Index warmup strategy

Write complete Python code using qdrant-client async. Production quality only.
```

---

---

# PHASE 2 — AUTHENTICATION & AUTHORIZATION

---

## Prompt 2.1 — JWT Authentication System

```
You are a senior security engineer building authentication for a FastAPI SaaS application.

I need to build a complete authentication system for my AI SaaS platform.

Tech stack:
- Python 3.11
- FastAPI
- MongoDB with Motor
- Pydantic v2
- python-jose for JWT
- passlib for password hashing
- sendgrid or resend for emails

Build a complete production authentication system:

1. User registration:
   - Email + password registration
   - Email verification flow (send verification link)
   - Check for duplicate emails
   - Password strength validation
   - Auto-create personal organization on signup

2. Login system:
   - Email + password login
   - Return access token (15 min expiry) + refresh token (30 days)
   - Store refresh tokens in MongoDB (for revocation)
   - Failed login attempt tracking (lock after 5 attempts)

3. Token management:
   - POST /auth/refresh — get new access token using refresh token
   - POST /auth/logout — revoke refresh token
   - POST /auth/logout-all — revoke all refresh tokens for user

4. Password management:
   - POST /auth/forgot-password — send reset email
   - POST /auth/reset-password — reset with token
   - POST /auth/change-password — change while logged in

5. FastAPI dependencies:
   - get_current_user — decode JWT, return user object
   - get_current_active_user — check user is not banned/deleted
   - require_org_member — check user belongs to org
   - require_role — check user has required role in org

6. Security features:
   - Rate limiting on auth endpoints (use slowapi)
   - CORS configuration
   - Security headers middleware
   - JWT secret rotation strategy

Write every route, schema, model, and dependency. Full production code.
Include example .env variables needed.
```

---

## Prompt 2.2 — OAuth2 Social Login

```
You are a senior FastAPI engineer implementing OAuth2 social authentication.

I need to add Google, GitHub, and Microsoft login to my FastAPI SaaS application.

Existing system:
- JWT authentication already built
- MongoDB user collection with email, hashed_password, etc.
- Users can already register with email/password

Add OAuth2 social login:

1. Google OAuth2:
   - GET /auth/google — redirect to Google
   - GET /auth/google/callback — handle callback
   - Extract user email, name, profile picture from Google
   - If user exists: log them in (return JWT)
   - If user doesn't exist: create account, then log in
   - Link Google account to existing email account if same email

2. GitHub OAuth2:
   - Same flow as Google
   - GET /auth/github and GET /auth/github/callback

3. Microsoft OAuth2:
   - Same flow
   - GET /auth/microsoft and GET /auth/microsoft/callback

4. Account linking:
   - Users can connect multiple social accounts to one profile
   - Store in user document: linked_accounts: [{provider, provider_user_id, email}]
   - API to list linked accounts
   - API to unlink a social account

5. Frontend redirect handling:
   - After successful OAuth, redirect to frontend with token
   - Handle errors gracefully
   - PKCE flow for security

Provide complete FastAPI code, all schemas, environment variables needed, and a test strategy.
```

---

## Prompt 2.3 — RBAC Permission System

```
You are a senior backend engineer building a role-based access control system.

I need a complete RBAC system for my multi-tenant AI SaaS platform.

Roles per organization:
- OWNER: full control, billing, delete org
- ADMIN: manage members, all workspace actions
- EDITOR: upload docs, create projects, chat
- VIEWER: read-only, search and chat only

Build:

1. Permission matrix — define exactly what each role can do for:
   - Organization management
   - Workspace management
   - Document upload and deletion
   - Project management
   - AI chat and search
   - Analytics viewing
   - API key management
   - Billing access

2. Python implementation:
   - Permissions enum with all possible actions
   - Role-to-permissions mapping dictionary
   - has_permission(user, org_id, permission) async function
   - FastAPI dependency: require_permission(permission)
   - Decorator approach for route protection

3. Dynamic permissions:
   - Workspace-level overrides (user is VIEWER in org but EDITOR in specific workspace)
   - API key permissions (an API key can have limited permissions)
   - Guest access (time-limited access to specific workspace)

4. Audit logging:
   - Log every permission check that results in denial
   - Log sensitive actions: delete, admin changes, billing
   - Store in MongoDB audit_logs collection
   - API to query audit logs (admin only)

5. Tests:
   - Unit tests for every role/permission combination
   - Integration tests for protected routes

Full Python code, FastAPI dependencies, MongoDB schemas. Production ready.
```

---

---

# PHASE 3 — BILLING & SUBSCRIPTIONS

---

## Prompt 3.1 — Stripe Billing Integration

```
You are a senior engineer specializing in SaaS billing with Stripe.

I need to build a complete billing system for my AI SaaS platform using Stripe.

My pricing plans:
- FREE: 100 AI queries/month, 3 documents, 1 workspace, 1 user
- PRO: $29/month — 5,000 queries, 100 documents, 5 workspaces, 5 users
- TEAM: $99/month — 20,000 queries, 500 documents, 20 workspaces, 20 users
- ENTERPRISE: Custom pricing — unlimited

Tech stack: Python, FastAPI, MongoDB, Stripe Python SDK

Build complete billing system:

1. Stripe product and price setup:
   - Script to create products and prices in Stripe
   - Monthly and annual billing options (annual = 20% discount)
   - Map Stripe price IDs to internal plan names

2. Subscription management API:
   POST /billing/checkout — create Stripe checkout session
   POST /billing/portal — create Stripe customer portal session
   GET /billing/subscription — get current subscription status
   POST /billing/cancel — cancel subscription

3. Stripe webhook handler:
   POST /billing/webhook — handle all Stripe events:
   - checkout.session.completed → activate subscription
   - invoice.payment_succeeded → renew subscription
   - invoice.payment_failed → send warning email, grace period
   - customer.subscription.deleted → downgrade to free
   - customer.subscription.updated → change plan

4. Usage metering:
   - Track queries used per organization per billing period
   - Track documents stored
   - Track storage in MB
   - Enforce limits (return 429 when limit reached)
   - Reset counters on billing period renewal

5. Usage limit enforcement middleware:
   - FastAPI middleware that checks limits before every AI query
   - Return informative error when limit reached
   - Grace period handling (allow 10% overage before hard block)

6. Billing dashboard data API:
   - Current plan details
   - Usage this billing period
   - Usage history (last 6 months)
   - Invoice history
   - Upgrade/downgrade options

Write complete Python code, Stripe webhook verification, all schemas. Production ready.
```

---

## Prompt 3.2 — Usage Tracking System

```
You are a senior backend engineer building a usage analytics and metering system.

I need to track usage across my AI SaaS platform for billing and analytics.

Things to track:
- AI queries (chat messages sent)
- Documents uploaded (count and MB)
- Vectors stored (count)
- API calls (by endpoint)
- Embedding API calls (to track external API costs)
- LLM API calls (to track external API costs)

Build:

1. Usage tracking middleware for FastAPI:
   - Intercept every API request
   - Record: org_id, user_id, endpoint, method, status_code, response_time_ms, timestamp
   - Non-blocking (async, don't slow down requests)
   - Batch writes to MongoDB every 10 seconds

2. Specific usage counters:
   - increment_query_count(org_id) — called after each AI response
   - increment_document_count(org_id, size_bytes) — called after upload
   - decrement_document_count(org_id, size_bytes) — called after delete
   - track_embedding_call(org_id, token_count, provider) — for cost tracking
   - track_llm_call(org_id, input_tokens, output_tokens, provider) — for cost tracking

3. Usage aggregation:
   - Daily aggregation job (runs at midnight)
   - Monthly rollup for billing
   - Store in usage_summary collection

4. Usage API endpoints:
   GET /analytics/usage/current — current period usage vs limits
   GET /analytics/usage/history — last 12 months monthly usage
   GET /analytics/usage/daily — daily breakdown for current month
   GET /analytics/costs — estimated API cost breakdown

5. Cost estimation:
   - Calculate estimated cost of Cohere API calls
   - Calculate estimated cost of OpenAI API calls
   - Show margin (what you charge vs what you pay)

MongoDB schemas, Python code, FastAPI routes. Full production implementation.
```

---

---

# PHASE 4 — FRONTEND DASHBOARD

---

## Prompt 4.1 — Flutter Web Dashboard Architecture

```
You are a senior Flutter engineer specializing in Flutter Web SaaS dashboards.

I need to build a complete Flutter Web dashboard for my AI SaaS platform called [PRODUCT NAME].

Backend: FastAPI REST API with JWT authentication
Design: Modern, clean, dark mode support, Arabic RTL support

Plan the complete frontend architecture:

1. Project structure:
   - Full folder structure for a Flutter Web SaaS app
   - State management: Riverpod (recommended) or Bloc
   - Routing: GoRouter
   - HTTP client: Dio with interceptors
   - Local storage: flutter_secure_storage

2. Pages to build (list all with route paths):
   - Landing page (/)
   - Auth pages (/login, /register, /forgot-password, /reset-password)
   - Dashboard (/dashboard)
   - Workspaces (/workspaces)
   - Document manager (/workspaces/:id/documents)
   - AI Chat (/workspaces/:id/chat)
   - Search (/workspaces/:id/search)
   - Analytics (/analytics)
   - Team management (/settings/team)
   - Billing (/settings/billing)
   - API keys (/settings/api-keys)
   - Organization settings (/settings/organization)

3. For each page provide:
   - What data it fetches
   - What API endpoints it calls
   - Key UI components needed
   - State it manages

4. Shared components to build:
   - Sidebar navigation
   - Top app bar
   - Document upload widget (drag and drop)
   - AI chat widget (embeddable)
   - Usage meter widget
   - Data table with pagination
   - Empty state components
   - Loading skeletons

5. API service layer:
   - AuthService (login, register, logout, refresh token)
   - OrganizationService
   - WorkspaceService
   - DocumentService
   - ChatService
   - AnalyticsService
   - BillingService

6. Arabic RTL support:
   - How to implement RTL toggle
   - Font choices for Arabic (Cairo font)
   - Direction-aware layout widgets

Give me the complete architecture plan, folder structure, and the code for 3 most important pages: Dashboard, Document Manager, and AI Chat.
```

---

## Prompt 4.2 — AI Chat Widget (Embeddable)

```
You are a senior Flutter and JavaScript engineer building embeddable chat widgets.

I need to build TWO versions of an AI chat widget for my SaaS platform:

VERSION 1 — Flutter Web Chat Page (full page inside dashboard)
VERSION 2 — Embeddable JavaScript widget (customers paste on their website)

For VERSION 1 (Flutter):
Build a complete Flutter chat UI with:
1. Message list with:
   - User messages (right aligned, blue bubble)
   - AI messages (left aligned, dark bubble)
   - Source citations shown below AI message (clickable chips)
   - Typing indicator animation
   - Timestamps
   - Copy message button
   - Thumbs up/down feedback buttons

2. Input area:
   - Text field with send button
   - Voice input button (optional)
   - Attach file button
   - Character counter

3. Chat history sidebar:
   - List of past sessions
   - Session title (auto-generated from first message)
   - Delete session option
   - New chat button

4. State management with Riverpod:
   - ChatNotifier class
   - Message streaming support (SSE)
   - Optimistic UI updates

For VERSION 2 (JavaScript embeddable widget):
Build a standalone JavaScript chat widget that:
1. Is loaded with a single script tag:
   <script src="https://yourplatform.com/widget.js" 
           data-workspace-id="abc123" 
           data-api-key="key_xxx">
   </script>

2. Features:
   - Floating chat button (bottom right)
   - Opens chat popup on click
   - Streams AI responses
   - Shows source citations
   - Persists conversation in localStorage
   - Mobile responsive
   - Customizable colors via data attributes
   - Works on any website (no framework required)

3. Build as:
   - Single JS file (no dependencies)
   - Minified production build
   - CORS-compatible with your FastAPI backend
   - CSP-compatible

Provide complete Flutter code for VERSION 1 and complete vanilla JavaScript for VERSION 2.
```

---

---

# PHASE 5 — INTEGRATIONS

---

## Prompt 5.1 — Google Drive Integration

```
You are a senior integration engineer building Google Drive sync for a SaaS platform.

I need to build a Google Drive integration for my AI knowledge platform.
When users connect Google Drive, their selected folders are automatically synced into the RAG system.

Tech stack: Python, FastAPI, MongoDB, Google Drive API v3

Build complete integration:

1. OAuth2 connection flow:
   - GET /integrations/google-drive/connect — start OAuth flow
   - GET /integrations/google-drive/callback — handle OAuth callback
   - Store encrypted refresh token in MongoDB
   - GET /integrations/google-drive/status — check connection status
   - DELETE /integrations/google-drive/disconnect — revoke access

2. Folder browser API:
   - GET /integrations/google-drive/folders — list user's Drive folders
   - POST /integrations/google-drive/select-folders — select folders to sync

3. Sync engine:
   - Initial sync: download all supported files from selected folders
   - Incremental sync: detect new/modified/deleted files using Drive webhooks
   - Support file types: PDF, DOCX, TXT, MD, Google Docs (export to text)
   - Track sync state per file in MongoDB (synced_files collection)

4. Change detection:
   - Register Google Drive push notifications (webhooks)
   - POST /integrations/google-drive/webhook — receive change notifications
   - Queue changed files for re-processing

5. Background sync worker:
   - Celery task for initial sync
   - Celery beat for periodic re-sync (every 6 hours)
   - Progress tracking (show sync status in dashboard)
   - Error handling and retry logic

6. File processing pipeline:
   - Download file from Drive
   - Extract text based on file type
   - Chunk and embed (using existing RAG engine)
   - Store in Qdrant with source metadata (drive_file_id, folder_path)
   - Update sync status in MongoDB

MongoDB schemas, complete Python code, error handling. Production quality.
```

---

## Prompt 5.2 — Slack Integration

```
You are a senior integration engineer building Slack integrations for a SaaS platform.

I need to build a Slack integration for my AI knowledge platform.
Goal: employees can ask AI questions directly from Slack without opening the dashboard.

Tech stack: Python, FastAPI, Slack Bolt SDK

Build complete Slack integration:

1. Slack app setup guide:
   - What OAuth scopes to request
   - Event subscriptions to enable
   - Slash commands to register
   - App manifest YAML

2. OAuth installation flow:
   - GET /integrations/slack/install — redirect to Slack OAuth
   - GET /integrations/slack/callback — handle callback, store bot token
   - Store workspace_id, bot_token, team_name in MongoDB

3. Slash command: /ask
   User types: /ask What is our refund policy?
   Bot responds with AI answer + source citations in thread

4. App mention handler:
   User types: @YourBot how do I request a vacation?
   Bot responds in same channel/thread

5. Interactive components:
   - "Show sources" button → opens modal with full source text
   - "👍 Helpful" / "👎 Not helpful" buttons → store feedback
   - "Ask follow-up" button → opens modal for follow-up question

6. Workspace-to-workspace mapping:
   - When installing Slack app, user selects which workspace to query
   - Multiple Slack workspaces can connect to different AI workspaces

7. Slack home tab:
   - Show recent questions
   - Show quick action buttons
   - Show usage stats

Complete Python code using Slack Bolt async. All event handlers, schemas, MongoDB storage.
```

---

## Prompt 5.3 — Notion Integration

```
You are a senior integration engineer building Notion sync for a SaaS AI platform.

I need to build a Notion integration that syncs Notion pages into my RAG knowledge base.

Tech stack: Python, FastAPI, Notion API

Build complete integration:

1. OAuth connection:
   - GET /integrations/notion/connect
   - GET /integrations/notion/callback
   - Store access_token in MongoDB (encrypted)

2. Content browser:
   - GET /integrations/notion/pages — list accessible pages and databases
   - POST /integrations/notion/select — select pages/databases to sync

3. Content extraction:
   - Extract text from Notion pages (handle all block types)
   - Extract rows from Notion databases
   - Handle nested pages recursively
   - Preserve page hierarchy in metadata

4. Sync engine:
   - Initial full sync of selected pages
   - Incremental sync using Notion's last_edited_time
   - Detect deleted pages and remove from vector DB
   - Run sync every 4 hours via Celery beat

5. Rich content handling:
   - Convert Notion blocks to clean text
   - Handle: paragraph, heading, bulleted_list, numbered_list, toggle, code, quote, callout, table
   - Preserve headers as section metadata for better retrieval

6. Source display:
   - When AI cites a Notion page, show: page title + Notion page URL
   - Deep link back to exact section if possible

Complete Python code, Notion API calls, text extraction logic, sync worker.
```

---

---

# PHASE 6 — AI AGENT LAYER

---

## Prompt 6.1 — AI Agent Framework

```
You are a senior AI engineer building an agentic AI system on top of a RAG platform.

I need to upgrade my RAG system from "question answering" to "AI agents that take actions."

Current system: FastAPI + RAG pipeline (search + LLM generation)
Goal: AI that can use tools and complete multi-step tasks

Build complete AI agent framework:

1. Tool definition system:
   - Base Tool class with: name, description, parameters schema, execute() method
   - Tool registry: register and discover available tools
   - Tool result handling

2. Build these initial tools:
   a. SearchKnowledgeBase(query, top_k) — search RAG documents
   b. CreateCalendarEvent(title, date, time, description) — Google Calendar
   c. SendSlackMessage(channel, message) — send to Slack
   d. CreateJiraTicket(title, description, priority) — create issue
   e. SendEmail(to, subject, body) — send email via Resend
   f. SummarizeDocument(document_id) — summarize a specific doc
   g. GetWeather(city) — example external API tool

3. Agent loop (ReAct pattern):
   - Thought: AI reasons about what to do
   - Action: AI calls a tool
   - Observation: AI sees tool result
   - Repeat until final answer
   - Maximum 10 iterations (prevent infinite loops)

4. FastAPI integration:
   POST /nlp/agent/{project_id} — run agent task
   Request: {"task": "Summarize this week's support tickets and send to Slack #team"}
   Response: streaming SSE with agent thoughts + final result

5. Agent memory:
   - Agents remember previous tool calls in current session
   - Can reference previous results in current task

6. Safety guardrails:
   - Confirm before destructive actions (delete, send email)
   - Rate limit tool calls per user
   - Audit log all agent actions

Complete Python code, tool implementations, agent loop, FastAPI routes. Production ready.
```

---

## Prompt 6.2 — Workflow Automation Engine

```
You are a senior backend engineer building a workflow automation system.

I need to build a workflow automation engine for my AI SaaS platform.
Users should be able to create automated workflows triggered by events.

Example workflows:
- "When new document uploaded → summarize → extract FAQs → notify Slack"
- "When support ticket received → classify → find solution → draft reply"
- "Every Monday → summarize last week's documents → email team"

Build complete workflow system:

1. Workflow data model:
   - Workflow: id, name, trigger, steps, enabled, org_id
   - Trigger types: document_uploaded, schedule (cron), webhook, manual
   - Step types: ai_summarize, ai_extract_entities, ai_generate_faq, send_slack, send_email, create_ticket, http_request

2. Workflow builder API:
   POST /workflows/ — create workflow
   GET /workflows/ — list workflows
   PUT /workflows/{id} — update workflow
   DELETE /workflows/{id} — delete workflow
   POST /workflows/{id}/run — trigger manually
   GET /workflows/{id}/runs — execution history

3. Workflow executor (Celery):
   - Execute steps in sequence
   - Pass output of one step as input to next
   - Handle step failures (retry, skip, abort)
   - Store execution log with step results

4. Built-in workflow steps:
   Each step as a Python class with execute(input, config) → output:
   - SummarizeStep: call LLM to summarize input text
   - ExtractEntitiesStep: extract key entities from text
   - GenerateFAQStep: generate FAQ from document
   - SlackNotifyStep: send message to Slack channel
   - EmailStep: send email via Resend
   - WebhookStep: POST to external URL
   - FilterStep: conditionally continue workflow

5. Event system:
   - Emit events when things happen (document uploaded, chat completed)
   - Workflow engine listens for events and triggers matching workflows
   - Use Redis pub/sub for event distribution

6. Workflow templates:
   - Pre-built workflow templates users can install in one click
   - "New employee onboarding assistant"
   - "Customer support auto-reply"
   - "Weekly knowledge digest"

MongoDB schemas, Celery tasks, FastAPI routes, complete Python code.
```

---

---

# PHASE 7 — ANALYTICS & OBSERVABILITY

---

## Prompt 7.1 — Business Analytics Dashboard Data

```
You are a senior data engineer building analytics for a SaaS AI platform.

I need to build the analytics data layer for my AI SaaS dashboard.

Data I need to show to organization admins:

1. Knowledge Base Analytics:
   - Total documents, total chunks, total vectors
   - Documents by type (PDF, DOCX, etc.)
   - Storage used vs limit
   - Recently added documents
   - Most searched documents

2. AI Usage Analytics:
   - Total queries this month (with trend vs last month)
   - Queries per day (last 30 days) — line chart data
   - Queries by user — bar chart
   - Average response time trend
   - Cache hit rate (queries answered from cache vs LLM)

3. Question Analytics:
   - Top 20 most asked questions
   - Questions with low confidence answers (potential knowledge gaps)
   - Questions that returned no results (knowledge gaps)
   - User satisfaction score (from thumbs up/down feedback)

4. Team Activity:
   - Active users this week
   - Queries per user
   - Last active timestamp per user

Build:
1. MongoDB aggregation pipelines for each metric
2. FastAPI endpoints returning each dataset
3. Caching strategy (cache analytics results for 1 hour in Redis)
4. Data format optimized for chart rendering (labels[], datasets[])

GET /analytics/overview — all KPIs in one request
GET /analytics/queries/daily — daily query counts
GET /analytics/questions/top — top questions
GET /analytics/questions/gaps — unanswered questions
GET /analytics/team — team activity

Full Python code, MongoDB aggregations, FastAPI routes. Production ready.
```

---

---

# PHASE 8 — ENTERPRISE FEATURES

---

## Prompt 8.1 — API Key Management

```
You are a senior backend engineer building API key management for a SaaS platform.

I need to build a developer API key system so customers can build on top of my platform.

Features needed:

1. API Key model:
   - key_id, key_prefix (first 8 chars, shown in UI), key_hash (stored), 
   - name, description, org_id, created_by, created_at
   - permissions (list of allowed actions)
   - rate_limit (requests per minute)
   - expires_at (optional)
   - last_used_at, total_uses
   - is_active

2. API:
   POST /api-keys/ — generate new key (return full key ONCE, never again)
   GET /api-keys/ — list keys (show prefix, never full key)
   DELETE /api-keys/{key_id} — revoke key
   PUT /api-keys/{key_id} — update name/permissions/rate limit
   GET /api-keys/{key_id}/usage — usage stats for this key

3. Authentication middleware:
   - Accept API key in header: X-API-Key: key_xxx...
   - Hash the received key, look up in MongoDB
   - Check key is active, not expired
   - Check rate limit (use Redis sliding window)
   - Inject org context from key's organization
   - Update last_used_at (async, non-blocking)

4. Key generation:
   - Format: prefix_randomsecure64chars
   - Use secrets.token_urlsafe(48) for the secret part
   - Store SHA256 hash in MongoDB
   - Never store or log plain key after creation response

5. Rate limiting per key:
   - Redis sliding window rate limiter
   - Configurable per key: 10/min, 100/min, 1000/min
   - Return 429 with Retry-After header when exceeded

Complete Python code, Redis integration, FastAPI middleware. Security-focused.
```

---

## Prompt 8.2 — Production Deployment & Security Hardening

```
You are a senior DevOps engineer hardening a FastAPI SaaS application for production.

I need to harden my AI SaaS platform for production deployment.

Current stack:
- FastAPI application
- MongoDB
- Qdrant
- Redis (new)
- Celery workers (new)
- Nginx reverse proxy
- Docker Compose

Provide complete production hardening guide:

1. Updated docker-compose.yml with:
   - All services: nginx, fastapi, mongodb, qdrant, redis, celery_worker, celery_beat, prometheus, grafana
   - Health checks for every service
   - Resource limits (CPU and memory)
   - Restart policies
   - Non-root users for all containers
   - Secrets via Docker secrets (not environment variables)
   - Internal network (services not exposed directly)

2. Nginx configuration:
   - SSL/TLS with Let's Encrypt (certbot auto-renewal)
   - HTTP/2 support
   - Rate limiting: 100 req/min per IP
   - Security headers: HSTS, CSP, X-Frame-Options, etc.
   - Gzip compression
   - WebSocket proxy for streaming

3. FastAPI security middleware:
   - CORS with explicit allowed origins
   - Request ID middleware (for tracing)
   - Security headers middleware
   - Request size limits (reject > 50MB)
   - Trusted proxy IPs only

4. MongoDB security:
   - Enable authentication
   - Create least-privilege application user
   - Enable audit logging
   - Disable direct external access
   - Automated backup script (mongodump to S3)

5. Secrets management:
   - All secrets in environment variables loaded from .env
   - .env.example with all variables documented
   - Secret rotation procedure
   - No secrets in code or Docker images

6. Monitoring alerts:
   - Prometheus alert rules for:
     - High error rate (> 1%)
     - High latency (P95 > 3s)
     - Low disk space (< 20%)
     - MongoDB connection failures
     - High memory usage (> 85%)

Complete docker-compose.yml, nginx.conf, Python middleware code, backup scripts.
```

---

---

# PHASE 9 — LAUNCH

---

## Prompt 9.1 — Landing Page Copy

```
You are a world-class SaaS copywriter specializing in AI products.

I am launching an AI knowledge platform called [PRODUCT NAME].

Target audience: Business owners and team leads at SMEs in the MENA region (Arabic + English speakers)

Core value: employees stop wasting time searching for information. They ask the AI, get instant answers from company documents, in Arabic or English.

Write complete landing page copy:

1. Hero section:
   - Headline (max 8 words, outcome-focused)
   - Subheadline (max 20 words)
   - CTA button text
   - Social proof line (e.g., "Join 200+ teams")

2. Problem section:
   - 3 pain points with emotional headlines
   - Each with 2-line description

3. Solution section:
   - 5 key features with benefit-focused headlines (not feature names)
   - Each with icon suggestion and 2-line description

4. How it works (3 steps):
   - Step 1, 2, 3 with action verbs

5. Social proof:
   - 3 customer testimonials (write example ones for the target market)
   - 2 company logos section placeholder

6. Pricing section:
   - FREE, PRO ($29), TEAM ($99), ENTERPRISE
   - For each: name, price, 5 bullet points, CTA

7. FAQ section:
   - 8 most common questions and answers

8. Footer CTA:
   - Final conversion headline + button

Write in English first, then provide Arabic translation.
Tone: professional but approachable. No jargon.
```

---

## Prompt 9.2 — Go-To-Market Strategy

```
You are a B2B SaaS growth strategist with experience launching in the MENA market.

I am launching [PRODUCT NAME], an AI knowledge platform for SMEs, with strong Arabic language support.

I have: a working product, no users yet, no marketing budget yet (bootstrap).

Build a complete 90-day go-to-market plan:

DAY 1-30 (Validation):
1. Where to find first 10 customers (specific communities, platforms, LinkedIn groups)
2. Exact outreach message template (LinkedIn DM, Arabic and English)
3. What to offer first customers (free extended trial, hands-on setup)
4. What questions to ask in discovery calls
5. How to measure if you have product-market fit

DAY 31-60 (First revenue):
1. Conversion strategy: free → paid
2. Content to create (LinkedIn posts, topics, posting schedule)
3. Communities to participate in (Arabic tech communities, startup groups)
4. Partnerships to approach (business consultants, IT companies, universities)
5. First 3 case studies to create

DAY 61-90 (Scale):
1. Referral program structure
2. First paid marketing channel to test (with $500 budget)
3. SEO content strategy (keywords to target, Arabic vs English)
4. Partnership pipeline
5. Metrics and KPIs to track weekly

Also provide:
- 10 LinkedIn post ideas for the first 30 days
- Email sequence for trial users (7 emails over 14 days)
- 5 objections customers will raise and how to handle them

Be specific to the MENA market. Give real platform names, community names, and tactics.
```

---

---

# QUICK REFERENCE — TECH STACK SUMMARY

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | FastAPI + Python 3.11 | API server |
| Database | MongoDB + Motor | Metadata, chat, users |
| Vector DB | Qdrant | Semantic search |
| Cache | Redis | Rate limiting, queues, cache |
| Auth | JWT + OAuth2 | Authentication |
| Background Jobs | Celery + RabbitMQ | Async processing |
| Billing | Stripe | Subscriptions |
| Email | Resend | Transactional email |
| Frontend | Flutter Web | Dashboard |
| Monitoring | Prometheus + Grafana | Observability |
| Deployment | Docker Compose + Nginx | Infrastructure |
| Object Storage | MinIO / S3 | File storage |

---

# PHASE ORDER (RECOMMENDED BUILD SEQUENCE)

```
Phase 0 → Choose niche + define MVP          (Week 1)
Phase 1 → Multi-tenant architecture          (Week 2-3)
Phase 2 → Authentication                     (Week 3-4)
Phase 3 → Billing (basic Stripe)             (Week 4-5)
Phase 4 → Frontend dashboard                 (Week 5-7)
Phase 5 → First integration (Google Drive)   (Week 7-8)
Phase 6 → AI agents (basic tools)            (Week 8-9)
Phase 7 → Analytics                          (Week 9-10)
Phase 8 → Hardening + API keys               (Week 10-11)
Phase 9 → Launch                             (Week 12)
```

---

> Built with MINI-RAG engine — Production-Ready RAG Infrastructure
> This prompt library is your complete blueprint from engine to SaaS business.
