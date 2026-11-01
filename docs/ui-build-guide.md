# Il Mercato Chat UI — Build Guide for an LLM

Oct 27, 2026 · @Hasaan

## 1. Brief

Build one self-contained `index.html` (HTML + CSS + vanilla JS, no build step) that recreates the **Il Mercato** design and talks to Eataly's internal RAG chatbot over HTTP. Hand this whole doc to the LLM; section 9 is a ready-made prompt that points at the rest.

**Deliverable**

- One file, `index.html`, under \~600 lines, opening directly in a browser.
- Works against a real endpoint set in one config object, and falls back to a built-in mock when `USE_MOCK = true`.
- No frameworks, no CDN scripts. Google Fonts is the only external request.

**Non-negotiables**

- Match the tokens in section 2 exactly: colors, fonts, radii, spacing.
- Every bot answer shows its source documents as chips (this is a RAG tool; staff must see where answers come from).
- Render model output as text, never as raw HTML (section 8).
- Keyboard and screen-reader usable; 44 px minimum touch targets.
- Layout collapses cleanly to phone width (sidebar stacks above the chat).

**Open questions to answer before building** (fill these in, or the LLM should leave them as clearly marked constants):

- [ ] Chatbot endpoint URL and auth method (bearer token, cookie session, or SSO proxy)
- [ ] Does the backend stream (SSE / chunked) or return one JSON response?
- [ ] Exact field names the backend returns for answer text and sources
- [ ] Where the signed-in employee's name and role come from

## 2. Design tokens

Define these as CSS custom properties on `:root` and use only the variables in the rest of the stylesheet. The greens and red are approximations of Eataly's palette; swap in official brand values if you have the guide.

**Colors**

| Token | Hex | Used for |
| --- | --- | --- |
| `--green` | #0F4C36 | Sidebar background, user bubble, bot avatar letter, typing dots, empty-state heading |
| `--green-dark` | #0A3626 | Link hover |
| `--red` | #C8102E | Send button (white icon on it) |
| `--cream` | #F6F1E7 | Page background; text on green |
| `--paper` | #FFFDF8 | Bot bubble, composer, header pill, suggestion cards |
| `--chip` | #F1EADB | Source chips |
| `--chip-hover` | #EFE7D6 | Source chip hover |
| `--line` | #E5DCC8 | Borders and dividers |
| `--line-strong` | #D9CDB3 | Composer border |
| `--ink` | #1B1B1B | Body text |
| `--ink-2` | #3F3B34 | Chip and pill text |
| `--muted` | #5E5A52 | Secondary text, labels, icon buttons (passes 4.5:1 on cream) |
| `--side-muted` | #CFE0D4 | Secondary text on green |
| `--side-label` | #A9C7B4 | Small uppercase labels on green |
| `--ok` | #2E8B57 | Knowledge-base status dot |

Topic dots in the sidebar: Menu #E0473F, Wine #E6B85C, HR #9FD1B3, SOPs #F6F1E7. The tricolore stripe under the wordmark uses #3FA36B / #F6F1E7 / #E0473F.

**Typography** (Google Fonts: `Fraunces` opsz 9..144, weights 400 and 600; `Inter` 400, 500, 600)

| Role | Font | Size / weight | Notes |
| --- | --- | --- | --- |
| Wordmark "eataly" | Fraunces | 30 px / 600 | letter-spacing 0.02em, line-height 1 |
| Page greeting (h1) | Fraunces | 24 px / 600 |  |
| Empty-state heading | Fraunces | 32 px / 600 | color `--green` |
| Bot avatar letter "e" | Fraunces | 18 px / 600 |  |
| Body, messages | Inter | 15 px / 400 | bot text line-height 1.6, user 1.5 |
| Buttons, nav | Inter | 14 px / 500 |  |
| Small labels | Inter | 11–12 px / 400 | uppercase, letter-spacing 0.06–0.12em |

**Shape and depth**

- Radii: buttons 12 px; nav items 10 px; chips 8 px; cards 14 px; composer 16 px; pills 999 px.
- Bubbles: user `18px 18px 4px 18px` (tail bottom-right); bot `4px 18px 18px 18px` (tail top-left).
- One shadow only, on the composer: `0 6px 24px rgba(60,45,20,.08)`. No gradients.
- Spacing on a 4/8 px rhythm: message gap 24 px, main padding 32×40 px, sidebar padding 28×20 px.

## 3. Layout anatomy

The page is a two-column flex row: a fixed-feel green sidebar on the left and the chat on the right. The row uses `flex-wrap: wrap` so the sidebar drops above the chat on phones without media queries.

**Root**: `display:flex; flex-wrap:wrap; min-height:100vh; background:var(--cream)`.

**Sidebar** (`<aside>`): `flex: 1 1 280px; max-width:100%`, background `--green`, padding 28×20 px, a vertical stack with 28 px gaps. Top to bottom:

1. Brand block: wordmark "eataly", then a row with a 36×3 px tricolore stripe and the label "STAFF ASSISTANT".
2. "New conversation" button (outlined, full width).
3. "Ask about" nav: four topic buttons, each with a coloured 8 px dot.
4. "Recent" list: the last few conversation titles; the active one has a faint highlight.
5. User footer, pinned to the bottom with `margin-top:auto`: initials avatar, name, role.

**Main** (`<main>`): `flex: 999 1 560px; min-width:0; height:100vh; display:flex; flex-direction:column`.

1. Header: greeting h1 + one-line subtitle on the left; a knowledge-base pill on the right (green dot + "Menu · HR Handbook · Kitchen SOPs"). Bottom border `--line`; wraps on narrow screens.
2. Message scroller: `flex:1; min-height:0; overflow-y:auto; display:flex; flex-direction:column-reverse`. Inside it, one inner column (`max-width:820px; margin:0 auto; padding:32px 40px; gap:24px`). The `column-reverse` wrapper keeps the view pinned to the newest message without scroll JavaScript.
3. Composer area: same 820 px column, padding `8px 40px 24px`; the input form, then a one-line disclaimer centred under it.

&#91;embedded content: Il Mercato layout · sidebar + main column\]

The sidebar stacks five parts with the user footer pinned last; the main column is header, scrolling messages and composer, top to bottom.

**Phone width (< \~860 px)**: the sidebar wraps to full width on top. Collapse it to the brand row plus a horizontal, scrollable row of topic buttons; hide "Recent" and the user footer. Reduce main padding to 16 px. On phones the main column can use `height:100dvh` so the composer sits above the keyboard.

## 4. Component specs

Each component below lists its markup element, its style, and its states. Icons are inline stroke SVGs (24×24 viewBox, `stroke="currentColor"`, stroke-width 2, round caps), never emoji or icon fonts.

| Component | Element | Style | States and behaviour |
| --- | --- | --- | --- |
| New conversation | `<button>` | min-height 44, 1px border rgba(246,241,231,.35), transparent bg, cream text, radius 12, plus icon 16 px | Clears messages, shows empty state, focuses input |
| Topic button | `<button>` in `<nav aria-label="Topics">` | min-height 44, padding 0 12, radius 10, 8 px colour dot + label, 14/500 | Hover bg rgba(246,241,231,.10). Click sends that topic's starter question |
| Recent item | `<button>` | padding 10×12, 14 px, `--side-muted` | Active: bg rgba(246,241,231,.08), text #E4ECE6 |
| User footer | `<div>` | top border rgba(246,241,231,.18); 36 px cream circle with green initials | Static |
| KB pill | `<div>` | 999 radius, `--paper` bg, `--line` border, 13 px, 8 px `--ok` dot | Turn dot `--red` and text "Knowledge base offline" if health check fails |
| User bubble | `<div>` in right-aligned row | max-width 75%, `--green` bg, cream text, padding 12×16, radius 18 18 4 18 | Plain text |
| Bot message | row: avatar + column | Avatar 36 px circle, `--paper` bg, `--line` border, Fraunces "e". Bubble `--paper`, `--line` border, padding 16×18, radius 4 18 18 18, `white-space: pre-line` | Gap 14 between avatar and bubble; 12 between bubble and sources row |
| Sources row | flex-wrap row under bot bubble | "SOURCES" label 12 px uppercase `--muted`; chips | Hidden when the answer has no sources |
| Source chip | `<a>` (if URL) else `<span>` | padding 6×10, radius 8, `--chip` bg, 13 px `--ink-2`, 13 px file icon | Hover `--chip-hover`. Opens document in new tab with `rel="noopener"` |
| Feedback buttons | two `<button aria-label="Helpful" / "Not helpful">` | 44×36, transparent, `--muted` icon, pushed right with a flex spacer | Toggle `aria-pressed`; selected uses `--green`. Sends feedback (section 5) |
| Typing indicator | bot row | Avatar + pill with three 7 px `--green` dots blinking (1.2 s, 0.2 s stagger) + "Searching the knowledge base…" 13 px `--muted` | Shown from send until first token or response |
| Empty state | in message column | Fraunces 32/600 "Come posso aiutarti?" in `--green`, subtitle, 2-column grid of suggestion cards | Shown when there are no messages |
| Suggestion card | `<button>` | min-height 72, padding 16, radius 14, `--paper` bg, `--line` border, 14/500 | Hover border `--green`. Click sends its text |
| Composer | `<form>` | `--paper` bg, 1px `--line-strong` border, radius 16, padding 8 8 8 20, the one shadow | Focus-within: border `--green` + 3px ring rgba(15,76,54,.15) |
| Input | `<textarea rows="1">` + visually hidden `<label>` | no border, 15 px Inter, min-height 44, auto-grows to 160 px | Enter sends; Shift+Enter newline; disabled while waiting |
| Send button | `<button type="submit" aria-label="Send">` | 44×44, radius 12, `--red` bg, white arrow icon | Disabled (opacity .5) when input empty or a request is in flight. Becomes a Stop button (square icon) while streaming |
| Disclaimer | `<p>` | 12 px `--muted`, centred | "Answers come from Eataly's internal documents. Always confirm allergen questions with the kitchen lead." |
| Error message | bot row | Bubble with 1px `--red` border, text "Something went wrong. Try again." + Retry button | Retry re-sends the last user message |

## 5. Backend integration

The page talks to the RAG service only through one small adapter, so the UI never changes when the backend does. Keep all endpoint details in a single `CONFIG` object at the top of the script.

```javascript
const CONFIG = {
  API_BASE: "/api",            // same-origin proxy; see Security
  CHAT_PATH: "/chat",
  FEEDBACK_PATH: "/feedback",
  HEALTH_PATH: "/health",
  STREAMING: false,            // true if the backend sends SSE
  USE_MOCK: true,              // false once the endpoint is live
  TIMEOUT_MS: 30000,
  HISTORY_TURNS: 6             // past turns sent for context
};
```

**Chat request** — `POST {API_BASE}{CHAT_PATH}`, `Content-Type: application/json`, credentials `include` if auth is cookie-based:

```json
{
  "message": "What allergens are in the Tagliatelle al Ragù?",
  "conversation_id": "c_8f2a",
  "history": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "topic": "menu"
}
```

**Chat response** (non-streaming) — the shape the UI expects after normalising:

```json
{
  "answer": "Tagliatelle al Ragù contains gluten, egg, dairy and celery...",
  "sources": [
    {"title": "Allergen Matrix", "url": "https://intranet/.../allergens.pdf", "snippet": "Contains: wheat, egg, milk, celery"}
  ],
  "conversation_id": "c_8f2a",
  "message_id": "m_31c9"
}
```

If the backend uses other field names, map them in one function and nowhere else:

```javascript
function normalize(raw) {
  return {
    answer: raw.answer ?? raw.response ?? raw.output ?? "",
    sources: (raw.sources ?? raw.documents ?? raw.citations ?? []).map(s => ({
      title: s.title ?? s.name ?? s.metadata?.source ?? "Document",
      url: s.url ?? s.metadata?.url ?? null,
      snippet: s.snippet ?? s.page_content?.slice(0, 200) ?? ""
    })),
    conversationId: raw.conversation_id ?? raw.session_id ?? null,
    messageId: raw.message_id ?? raw.id ?? null
  };
}
```

**Streaming** (when `STREAMING: true`) — the backend responds with `text/event-stream`. Read it with `fetch` + `response.body.getReader()` (not `EventSource`, which cannot POST). Expected events:

| Event | Data | UI action |
| --- | --- | --- |
| `token` | `{"text": "partial words"}` | Hide typing dots on first token; append text to the open bot bubble |
| `sources` | `{"sources": [...]}` | Render the chips under the bubble |
| `done` | `{"message_id": "...", "conversation_id": "..."}` | Enable input, store ids |
| `error` | `{"message": "..."}` | Replace bubble with the error state |

**Feedback** — `POST {API_BASE}{FEEDBACK_PATH}` with `{"message_id": "m_31c9", "rating": "up" | "down"}`. Fire-and-forget; never block the UI on it.

**Health** — `GET {API_BASE}{HEALTH_PATH}` on load; any non-2xx turns the header pill to its offline state.

**Errors and timeouts**

- Wrap every call in an `AbortController` with `TIMEOUT_MS`; the Stop button calls `abort()` too.
- 401/403: show "Your session has expired. Please sign in again." with a link to the sign-in page.
- 429: "Too many questions at once. Try again in a moment."
- Network error, 5xx, timeout: the generic error bubble with Retry.

**Mock mode** — when `USE_MOCK` is true, `askBackend()` resolves after \~900 ms from a small keyword table (allergens, time off, wine pairing, closing checklist, uniform) and returns the normalised shape above, including sources. This lets designers and testers use the page before the API exists.

## 6. Front-end logic

Keep one plain state object and one `render()` path per region; no framework needed.

```javascript
const state = {
  conversationId: null,
  messages: [],        // {id, role: "user"|"bot"|"error", text, sources: [], rating: null}
  pending: false,      // a request is in flight
  controller: null,    // AbortController for the open request
  user: { name: "Marco R.", role: "Pasta Station · Front of House" }
};
```

**Send flow**

1. Trim the input; ignore if empty or `state.pending`.
2. Push `{role:"user", text}`, clear and reset the textarea height, set `pending = true`, disable Send, show the typing indicator.
3. Build the request from the message, `conversationId` and the last `HISTORY_TURNS` messages (user and bot only, no errors).
4. Call `askBackend()` (real or mock). Non-streaming: on resolve, `normalize()`, push the bot message with its sources. Streaming: push an empty bot message on the first `token`, append text as it arrives, attach sources on `sources`.
5. On `done` or resolve: store `conversationId` and `messageId`, set `pending = false`, re-enable input, return focus to the textarea.
6. On error or abort: remove the typing indicator, push `{role:"error"}` (skip for a user-initiated Stop; keep partial text instead).

**Rendering rules**

- Build DOM with `document.createElement` and set text with `textContent`. Bot text keeps line breaks through `white-space: pre-line`.
- Optional light formatting: if the backend returns Markdown, convert only `**bold**`, numbered lists and line breaks with a tiny escaper-first formatter, or use a sanitiser (see section 8). Never assign model output to `innerHTML` unsanitised.
- Append new messages instead of re-rendering the whole list, so streaming does not flicker.
- Announce new bot answers to screen readers: the message list has `role="log" aria-live="polite"`.

**Other behaviour**

- Topic buttons and suggestion cards call `send(starterQuestion)` and pass `topic` in the request.
- New conversation: abort any open request, clear `messages`, set `conversationId = null`, show the empty state.
- Recent list: optional. Store titles (first user message, 40 characters) per conversation in `localStorage`, wrapped in try/catch; it is a convenience, not a record.
- Feedback: set `rating`, toggle `aria-pressed`, POST once; a second click on the same thumb clears it.
- Composer keys: Enter sends; Shift+Enter inserts a newline; Escape while streaming stops.
- Textarea auto-grow: on `input`, set `height = "auto"` then `height = min(scrollHeight, 160) + "px"`.
- Greeting by time of day in Riyadh local time is optional: Buongiorno before 17:00, Buonasera after.

## 7. Reference HTML skeleton

Start from this structure. It fixes the class names, landmarks and token wiring; the LLM fills in the remaining CSS from section 4 and the logic from sections 5–6.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Eataly Staff Assistant</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
<style>
  :root{
    --green:#0F4C36; --green-dark:#0A3626; --red:#C8102E;
    --cream:#F6F1E7; --paper:#FFFDF8; --chip:#F1EADB; --chip-hover:#EFE7D6;
    --line:#E5DCC8; --line-strong:#D9CDB3;
    --ink:#1B1B1B; --ink-2:#3F3B34; --muted:#5E5A52;
    --side-muted:#CFE0D4; --side-label:#A9C7B4; --ok:#2E8B57;
    --serif:"Fraunces", Georgia, serif; --sans:"Inter", system-ui, sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--cream);color:var(--ink);font:400 15px/1.5 var(--sans)}
  .app{display:flex;flex-wrap:wrap;min-height:100vh}
  .sidebar{flex:1 1 280px;max-width:100%;background:var(--green);color:var(--cream);
           padding:28px 20px;display:flex;flex-direction:column;gap:28px}
  .main{flex:999 1 560px;min-width:0;height:100vh;display:flex;flex-direction:column}
  .scroller{flex:1;min-height:0;overflow-y:auto;display:flex;flex-direction:column-reverse}
  .column{width:100%;max-width:820px;margin:0 auto;padding:32px 40px;display:flex;flex-direction:column;gap:24px}
  .composer{display:flex;align-items:flex-end;gap:10px;padding:8px 8px 8px 20px;background:var(--paper);
            border:1px solid var(--line-strong);border-radius:16px;box-shadow:0 6px 24px rgba(60,45,20,.08)}
  .composer:focus-within{border-color:var(--green);box-shadow:0 0 0 3px rgba(15,76,54,.15)}
  .sr-only{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
  @keyframes blink{0%,80%,100%{opacity:.25}40%{opacity:1}}
  @media (prefers-reduced-motion: reduce){.dot{animation:none}}
  /* ...bubbles, chips, nav, cards, buttons per section 4... */
</style>
</head>
<body>
<div class="app">
  <aside class="sidebar">
    <div class="brand">
      <div class="wordmark">eataly</div>
      <div class="brand-sub"><span class="tricolore" aria-hidden="true"></span>Staff Assistant</div>
    </div>
    <button class="btn-new" id="newChat">New conversation</button>
    <nav aria-label="Topics" id="topics"></nav>
    <section aria-label="Recent conversations" id="recent"></section>
    <div class="user" id="user"></div>
  </aside>

  <main class="main">
    <header class="header">
      <div>
        <h1 id="greeting">Buongiorno, Marco</h1>
        <p class="subtitle">Ask anything about the menu, policies or kitchen procedures.</p>
      </div>
      <div class="kb-pill" id="kbPill"><span class="kb-dot" aria-hidden="true"></span>Menu · HR Handbook · Kitchen SOPs</div>
    </header>

    <div class="scroller">
      <div class="column" id="messages" role="log" aria-live="polite" aria-label="Conversation"></div>
    </div>

    <div class="composer-wrap">
      <form class="composer" id="composer">
        <label for="input" class="sr-only">Your question</label>
        <textarea id="input" rows="1" placeholder="Ask about a dish, a policy, a procedure…"></textarea>
        <button type="submit" class="btn-send" id="send" aria-label="Send"><!-- arrow svg --></button>
      </form>
      <p class="disclaimer">Answers come from Eataly's internal documents. Always confirm allergen questions with the kitchen lead.</p>
    </div>
  </main>
</div>

<script>
  const CONFIG = { /* section 5 */ };
  const TOPICS = [
    { label: "Menu & allergens", color: "#E0473F", topic: "menu", q: "What allergens are in the Tagliatelle al Ragù?" },
    { label: "Wine & pairings",  color: "#E6B85C", topic: "wine", q: "Wine pairing for Cacio e Pepe?" },
    { label: "HR & policies",    color: "#9FD1B3", topic: "hr",   q: "How do I request time off?" },
    { label: "Kitchen SOPs",     color: "#F6F1E7", topic: "sops", q: "Closing checklist for the pasta station" }
  ];
  const state = { /* section 6 */ };

  function normalize(raw) { /* section 5 */ }
  async function askBackend(payload, onToken, onSources, signal) { /* real or mock */ }
  async function send(text, topic) { /* section 6 send flow */ }
  function renderUserMessage(msg) {}
  function renderBotMessage(msg) {}      // bubble + sources row + feedback
  function renderTyping(show) {}
  function renderEmptyState() {}
  function renderError(retryText) {}
  function sendFeedback(messageId, rating) {}
  function newConversation() {}
  function checkHealth() {}

  // wire events: submit, Enter/Shift+Enter, auto-grow, topic clicks, new chat
  checkHealth();
  renderEmptyState();
</script>
</body>
</html>
```

## 8. Accessibility, security and QA

**Security**

- Never put API keys or model-provider secrets in the HTML. The page calls a same-origin backend (`/api/...`) that holds credentials and enforces employee sign-in.
- Treat all model output and source titles as untrusted: `textContent` only, or sanitise with DOMPurify before any `innerHTML`.
- Source links: allow only `https:` (and your intranet scheme); open with `target="_blank" rel="noopener noreferrer"`.
- If the page is served from a different origin than the API, configure CORS on the backend for that one origin; do not use `*` with credentials.
- Do not store full conversations in `localStorage` on shared store terminals; keep only titles, or nothing.

**Accessibility**

- Landmarks: `<aside>`, `<nav aria-label="Topics">`, `<main>`, `<header>`; one `<h1>`.
- Every icon-only button has `aria-label`; decorative SVGs have `aria-hidden="true"`.
- Visible focus: `:focus-visible { outline: 2px solid var(--green); outline-offset: 2px }` (cream outline inside the sidebar).
- Message list is `role="log" aria-live="polite"`; the typing indicator has visually hidden text "Assistant is typing".
- Contrast: all listed text tokens pass 4.5:1 on their backgrounds; do not lighten `--muted`.
- Respect `prefers-reduced-motion` for the typing dots.

**QA checklist** (run before shipping)

- [ ] Side-by-side with the Mercato mockup at 1440×900: sidebar, header, bubbles, chips and composer match
- [ ] At 390 px wide: no horizontal scroll, sidebar stacked, composer visible above the keyboard
- [ ] Mock mode: each topic button returns an answer with source chips
- [ ] Real endpoint: answer, sources and ids map correctly through `normalize()`
- [ ] Streaming (if used): text appears progressively; Stop button aborts cleanly
- [ ] Errors: kill the API and confirm the error bubble and Retry; expired session shows sign-in message
- [ ] A bot answer containing `<img src=x onerror=alert(1)>` renders as plain text
- [ ] Keyboard only: Tab reaches every control in order; Enter sends; Shift+Enter adds a line
- [ ] Screen reader (VoiceOver) announces new answers
- [ ] Feedback POST fires once per click with the right `message_id`

## 9. Copy-paste prompt

Paste this into the other LLM, followed by sections 1–8 of this doc (export the doc as Markdown to paste it in one go). Replace the bracketed values first.

```markdown
You are a senior front-end engineer. Build a single self-contained file, index.html,
for "Eataly Staff Assistant", an internal RAG chatbot that Eataly employees use to ask
about the menu, wine, HR policies and kitchen SOPs.

Recreate the "Il Mercato" design exactly as specified in the guide below:
- Use the CSS tokens, fonts, radii and component specs verbatim.
- Start from the reference HTML skeleton (section 7) and keep its class names and landmarks.
- Vanilla HTML/CSS/JS only. Google Fonts is the only external request.

Backend:
- Endpoint base: [https://your-host/api]  (chat: [/chat], feedback: [/feedback], health: [/health])
- Auth: [cookie session | bearer token injected by proxy | none in dev]
- Streaming: [yes, SSE events token/sources/done/error | no, single JSON]
- Response fields: answer in [answer], sources in [sources] with [title/url/snippet]
Put all of this in one CONFIG object, route every response through normalize(),
and include a mock mode (USE_MOCK) with sample answers and sources.

Rules:
- Never render model output with innerHTML unless sanitised; use textContent.
- Every bot answer shows its sources as chips; hide the row when there are none.
- Keyboard and screen-reader accessible; 44 px minimum targets; works at 390 px wide.
- No placeholder lorem ipsum; use the sample questions in the guide.

Return the complete index.html in one code block, then a short list of anything
you assumed. Before answering, check your output against the QA checklist in section 8.

--- GUIDE ---
[paste sections 1-8 here]
```

Tip: also attach a screenshot of the Mercato artboard from the design canvas. Models match a design far more closely when they can see it.
