# Chat — End-to-end workflow

This document describes how chat, threads, and history work from UI to backend and back. All endpoints use the API prefix (e.g. `/api/v1`). JWT is required unless noted.

---

## 1. Startup

- On API startup, `VectorQueries.init_chat_history_table()` runs and creates the **`chat_history`** table if it does not exist.
- Columns: `id`, `session_id`, `user_id`, `role` (`user` | `assistant`), `content`, `created_at`.
- **`session_id`** in the DB is the same as **`thread_id`** in the API.

---

## 2. List user’s conversations (thread list)

**Goal:** Show the user a list of past conversations so they can pick one or start a new one.

| Step | Who | Action |
|------|-----|--------|
| 1 | UI | Call **GET** `{API_PREFIX}/chat/threads` with JWT (optional query: `?limit=50`). |
| 2 | API | Resolves `user_id` from JWT, calls `VectorQueries.list_chat_threads(user_id, limit)`. |
| 3 | DB | Returns rows: one per thread (`session_id`), with `updated_at` and first user message as `preview` (truncated). |
| 4 | API | Returns **`{ "threads": [ { "thread_id", "updated_at", "preview" }, ... ] }`**. |
| 5 | UI | Renders list (e.g. sidebar). User can click a thread or choose “New chat”. |

---

## 3. Open a thread (load history for one conversation)

**Goal:** When the user selects a thread, load its messages and show them in the chat UI.

| Step | Who | Action |
|------|-----|--------|
| 1 | UI | User selects a thread; UI has a **thread_id** (from list or from last response). |
| 2 | UI | Call **GET** `{API_PREFIX}/chat/history?thread_id=<thread_id>` with JWT. |
| 3 | API | Resolves `user_id` from JWT, calls `VectorQueries.get_chat_history(thread_id, user_id, 50)`. |
| 4 | DB | Returns messages for that `session_id` and user, ordered by `created_at` ASC. |
| 5 | API | Returns **`{ "thread_id", "history": [ { "role", "text" }, ... ] }`**. |
| 6 | UI | Renders messages (user vs assistant) in order. |

---

## 4. Send a message (REST)

**Goal:** User sends a message in the current thread; backend uses history, runs the orchestrator, persists the new turn, and returns the assistant reply.

| Step | Who | Action |
|------|-----|--------|
| 1 | UI | User types a message. UI has current **thread_id** (from list, from “Open thread”, or a new UUID for “New chat”). |
| 2 | UI | **POST** `{API_PREFIX}/query` with JWT and body: **`{ "query", "thread_id", "person_id?", "history?", "cot?" }`**. |
| 3 | API | Resolves `user_id` from JWT. Uses **thread_id** from body or generates new UUID. |
| 4 | API | If **history** not in body: loads **history** = `get_chat_history(thread_id, user_id)`. Builds **context** = `{ person_id, thread_id, history, cot, current_date }`. |
| 5 | API | (Optional) Checks response cache by (query, cot, user, thread). On cache hit: persists user + assistant messages to **chat_history**, returns cached response. |
| 6 | API | On cache miss: calls **orchestrator.run_async(query, context)**. Orchestrator puts **context["history"]** into state as **chat_history** and runs the graph (classifier → retrieval → synthesizer). |
| 7 | API | After run: **insert_chat_message(thread_id, user_id, "user", query)** and **insert_chat_message(thread_id, user_id, "assistant", response_text)**. |
| 8 | API | Returns **QueryResponse** (response, documents, agent_path, etc.). Response body includes **thread_id** if the UI needs it. |
| 9 | UI | Appends user message and assistant reply to the current thread view. Keeps using same **thread_id** for the next message. |

---

## 5. Send a message (WebSocket / streaming)

**Goal:** Same as REST, but the UI connects over WebSocket and gets streaming events; history and persistence behave the same.

| Step | Who | Action |
|------|-----|--------|
| 1 | UI | Connect to **WebSocket** `/ws/chat` with JWT (e.g. query param or first message). |
| 2 | UI | Send JSON: **`{ "query", "thread_id?", "person_id?", "history?", "cot?" }`**. |
| 3 | API | Resolves **user_id** from token. **thread_id** = body or `person_id` or `user_id` or `"default_thread"`. **history** = body or `get_chat_history(thread_id, user_id)`. **context** = same shape as REST. |
| 4 | API | **orchestrator.run(query, context)** streams events; API forwards each event to the client. |
| 5 | API | On stream done: **insert_chat_message(thread_id, user_id, "user", query)** and **insert_chat_message(thread_id, user_id, "assistant", response_text)** (response_text from last complete event). |
| 6 | UI | Renders streamed chunks; on “done”, treats reply as complete and keeps **thread_id** for next message. |

---

## 6. New chat

**Goal:** Start a new conversation without loading an old thread.

| Step | Who | Action |
|------|-----|--------|
| 1 | UI | User clicks “New chat”. UI generates a new **thread_id** (e.g. UUID) and clears the current thread’s messages from the view. |
| 2 | UI | Next **POST /query** or WebSocket message is sent **without** `thread_id`, or with this new **thread_id**. |
| 3 | API | If no **thread_id** in request: API generates a new UUID and uses it for this request. That UUID is the new thread; all subsequent messages in that conversation should use the same **thread_id** (returned in response or sent by client). |
| 4 | API | **get_chat_history(new_thread_id, user_id)** returns `[]`. Orchestrator runs with empty history. After response, both messages are persisted under the new **thread_id**. |
| 5 | UI | From then on, UI uses this **thread_id** for every message in this conversation and can later load it via **GET /chat/history?thread_id=...** or show it in **GET /chat/threads**. |

---

## 7. Data flow summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ UI                                                                           │
│   List threads → GET /chat/threads                                          │
│   Open thread  → GET /chat/history?thread_id=...                            │
│   Send message → POST /query { query, thread_id }  OR  WS /ws/chat            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ API (main.py)                                                                │
│   • Resolve user_id from JWT                                                 │
│   • thread_id from request or new UUID                                       │
│   • history = request.history ?? get_chat_history(thread_id, user_id)        │
│   • context = { person_id, thread_id, history, cot, current_date }           │
│   • orchestrator.run_async(query, context) or orchestrator.run(...)          │
│   • insert_chat_message(thread_id, user_id, "user", query)                 │
│   • insert_chat_message(thread_id, user_id, "assistant", response)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    ▼                 ▼                 ▼
┌───────────────────┐  ┌──────────────────────┐  ┌─────────────────────────┐
│ VectorQueries     │  │ Orchestrator          │  │ chat_history table      │
│ list_chat_threads │  │ state["chat_history"] │  │ session_id, user_id,    │
│ get_chat_history  │  │ = context["history"]  │  │ role, content,          │
│ insert_chat_message│  │ (used by agents)      │  │ created_at              │
└───────────────────┘  └──────────────────────┘  └─────────────────────────┘
```

---

## 8. Response shapes (for UI)

- **GET /chat/threads**  
  `{ "threads": [ { "thread_id": string, "updated_at": string (ISO), "preview": string | null }, ... ] }`

- **GET /chat/history**  
  `{ "thread_id": string, "history": [ { "role": "user" | "assistant", "text": string }, ... ] }`

- **POST /query**  
  `QueryResponse`: `response`, `documents`, `agent_path`, `regulation_types`, `user_id`, `query_id`, etc. Use the same **thread_id** you sent for the next request.

- **WebSocket**  
  Stream of events; final event type `"complete"` carries the full result (including `response`). Persistence uses the **thread_id** you sent in the message.
