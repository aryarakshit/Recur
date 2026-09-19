# RECURSIVE 2026 — Official Hackathon Discord AI Support Bot

A production-grade, authoritative Discord AI support bot engineered for the **RECURSIVE 2026 Hackathon** (organized by the GNIT Kolkata ACM Student Chapter). Built with **Python 3.11+**, **discord.py**, **FAISS vector search**, and **Gemini / Groq LLMs**.

The bot acts as an official, hallucination-free hackathon assistant that reads messages across authorized channels, automatically identifies legitimate participant inquiries while ignoring casual chatter, and delivers answers strictly grounded in official documentation (Markdown & PDF). Whenever information is not covered in the knowledge base, it reliably triggers a safe fallback directing participants to tag **`@Core Member`** or **`@Volunteer`**.

---

## Key Features

- **Authoritative & Hallucination-Free RAG**: Answers participant questions strictly using facts from `knowledge/` (`.md` and `.pdf` files). Never invents rules, deadlines, prizes, or campus guidelines.
- **Safe Maintainer Fallback**: If an inquiry cannot be answered with official context, the bot states:
  > *"I couldn't find this information in the official hackathon knowledge base. Please tag @Core Member or @Volunteer for clarification."*
  Dynamic role tagging automatically resolves the server's actual `@Core Member` and `@Volunteer` roles and logs the unanswered question into SQLite for organizers to review.
- **Two-Stage Reply Decision Layer**:
  1. *Direct Mentions & Replies*: Always answered.
  2. *Fast Heuristic & Noise Filter*: Keyword recognition for hackathon inquiries, immediate silence on memes, greetings, bot commands, and casual chat.
  3. *Cheap LLM Evaluation*: Fast single-token classification (`YES` or `NO`) for ambiguous messages.
- **Multi-Provider LLM Abstraction**:
  - Primary: **Google Gemini API** (`gemini-2.5-flash` / `text-embedding-004`).
  - Alternative: **Groq API** (e.g. `qwen/qwen3.8-27b`, `llama-3.3-70b-versatile`) with automatic rate-limit token bounding.
  - Offline Mock Mode: Runs locally when keys are blank.
- **Vector Search (FAISS)**: Fast, normalized cosine similarity retrieval over chunked documents with calibrated similarity filtering. Supports both Markdown and PDF documents.
- **Lightweight Sliding Memory**: Retains the last 6 conversation turns per user/channel in SQLite so follow-up inquiries maintain context without polluting authoritative answers.
- **Organizer Slash Commands**:
  - `/status` — View bot uptime, active LLM provider, chunk count, and database stats.
  - `/reloadkb` — Re-parse and re-index `knowledge/` dynamically without restarting the bot.
  - `/ask <question>` — Ask a question directly via slash command.
  - `/asktest <question>` — Privately test RAG responses with retrieved chunk sources and similarity scores.
  - `/clearcache` — Clear conversation history cache for the channel or entire server.
  - `/unanswered [limit]` — View recent questions that triggered safe fallback to update FAQ documentation.

---

## Project Structure

```text
Recur/
├── bot.py                     # Main Discord bot entrypoint & slash command sync
├── config.py                  # Dataclass configuration & .env loader
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Container image specification
├── docker-compose.yml         # Container orchestration
├── hackbot.service            # Systemd service unit template
├── .env.example               # Environment variable template
├── .env                       # Local secrets (gitignored)
├── .gitignore
├── README.md
│
├── ai/                        # AI & LLM integration layer
│   ├── classifier.py          # Heuristics & LLM message classifier (YES/NO)
│   ├── embeddings.py          # Gemini & Local sub-word embedding providers
│   ├── generator.py           # Grounded prompt synthesis & safe fallback policy
│   └── provider.py            # LLMProvider abstraction (Gemini, Groq, Mock)
│
├── discord_bot/               # Discord UI & interaction layer
│   ├── commands.py            # Slash commands (/status, /reloadkb, /ask, etc.)
│   ├── message_handler.py     # Channel message listener & role-based dispatcher
│   └── permissions.py         # Admin, Core Member & Volunteer role validators
│
├── knowledge/                 # Official hackathon knowledge base (.md & .pdf)
│   ├── rules.md               # Team limits, eligibility, code conduct
│   ├── schedule.md            # Timeline, round deadlines, demo day
│   ├── submission.md          # Idea submission process & Devfolio instructions
│   ├── prizes.md              # Cash prizes, track awards, sponsor perks
│   ├── judging.md             # Evaluation criteria and weights
│   ├── faq.md                 # 10 official Q&As from recursiveacm.in
│   ├── venue.md               # GNIT Kolkata address, transit from Sodepur station
│   ├── chair.md               # Hackathon chair lore and origins
│   ├── problem-statements.md  # 6 hackathon tracks & problem statements
│   ├── technology-rules.md    # Allowed tech stacks, libraries, templates
│   ├── sponsors.md            # Partners & sponsors
│   ├── contacts.md            # Contact info & social links
│   └── links.md               # Official Devfolio, Google Slides & GitHub links
│
├── rag/                       # RAG indexing & retrieval pipeline
│   ├── indexer.py             # Markdown + PDF chunker & FAISS builder
│   ├── retriever.py           # Vector similarity search & context formatting
│   └── models.py              # DocumentChunk & RetrievalResult data models
│
├── database/                  # SQLite persistence engine
│   └── db.py                  # Unanswered questions, conversation history, metrics
│
├── storage/                   # Storage & memory wrappers
│   ├── database.py            # Database client alias
│   └── memory.py              # Sliding-window conversation manager
│
├── scripts/                   # CLI maintenance tools
│   ├── index_knowledge.py     # Build or rebuild FAISS index from knowledge/
│   ├── rebuild_index.py       # Re-indexer alias
│   └── ask.py                 # Test Q&A pipeline directly in terminal
│
└── tests/                     # Automated test suite (17 unit tests)
    ├── test_decision_layer.py # Heuristics & classifier tests
    ├── test_provider.py       # Provider fallback & generation tests
    ├── test_rag.py            # Indexer & retrieval threshold tests
    └── test_storage.py        # SQLite history & logging tests
```

---

## Setup & Installation Guide

### 1. Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git
- Discord Bot Token

### 2. Clone & Install Dependencies
```bash
git clone https://github.com/your-org/Recur.git
cd Recur

# Create and activate virtual environment
python -m venv venv
# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Discord Developer Portal Configuration

1. Visit the [Discord Developer Portal](https://discord.com/developers/applications).
2. Click **New Application** and name it (e.g. `Recursive AI Support`).
3. In the left navigation menu, click **Bot**:
   - Click **Reset Token** and copy your token.
   - Under **Privileged Gateway Intents**, turn **ON**:
     - **MESSAGE CONTENT INTENT** *(CRITICAL: The bot cannot read message content without this enabled!)*
     - **SERVER MEMBERS INTENT** *(Recommended for resolving roles dynamically)*
4. In the left navigation menu, click **OAuth2** -> **URL Generator**:
   - Under **Scopes**, select:
     - `bot`
     - `applications.commands`
   - Under **Bot Permissions**, select:
     - `Send Messages`
     - `Send Messages in Threads`
     - `Read Messages/View Channels`
     - `Read Message History`
     - `Embed Links`
     - `Mention Everyone` *(Optional: needed only if the bot pings roles without permissions)*
5. Copy the generated URL at the bottom, paste it into your browser, and authorize it for your server.

---

## Environment Configuration

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

### `.env` File Reference

```env
# ==============================================================================
# DISCORD BOT CONFIGURATION
# ==============================================================================
# Paste your Discord bot token here:
DISCORD_TOKEN=your_real_discord_bot_token_here

# Channel where participants can reach organizers
ORGANIZER_CHANNEL_NAME=#help
ORGANIZER_CHANNEL_ID=

# Role names or IDs for permissions and safe fallback pings
ADMIN_ROLE_ID=
CORE_MEMBER_ROLE_ID=
VOLUNTEER_ROLE_ID=
MAINTAINER_ROLE_ID=

# Display string used when mentioning maintainers in fallback answers
# Default: "@Core Member or @Volunteer"
MAINTAINER_MENTION=@Core Member or @Volunteer

# Optional comma-separated channel IDs to restrict the bot to
ALLOWED_CHANNEL_IDS=

# ==============================================================================
# LLM PROVIDER CONFIGURATION
# ==============================================================================
# Options: 'gemini' or 'groq'
LLM_PROVIDER=groq

# Google Gemini API Settings
GEMINI_API_KEY=
EMBEDDING_MODEL=text-embedding-004

# Groq API Settings (Fast on-demand inference)
GROQ_API_KEY=gsk_your_groq_key_here
LLM_MODEL=qwen/qwen3.8-27b
```

> **Dynamic Role Tagging**: When a role ID (`CORE_MEMBER_ROLE_ID` / `VOLUNTEER_ROLE_ID`) is provided, or when the bot finds roles named `Core Member` or `Volunteer` in your server, it will dynamically ping `<@&ROLE_ID>` so organizers receive instant notifications.

---

## Building the Knowledge Base

The bot answers queries using files in `knowledge/`. Both **Markdown (`.md`)** and **PDF (`.pdf`)** files are automatically parsed and chunked.

### Indexing Knowledge Files
Run the indexing script:
```bash
python scripts/index_knowledge.py
```
*Output:*
```text
Indexed 15 files into 51 chunks. Saved index to data/faiss.index
```

### Adding New Documentation
1. Place any new `.md` or `.pdf` file into `knowledge/` (e.g. `knowledge/sponsor-bounties.pdf`).
2. Run `python scripts/index_knowledge.py`, or simply type `/reloadkb` in Discord!

---

## Running the Bot

### Local Development Run
```bash
python bot.py
```

### Testing Q&A from Terminal (CLI)
You can test queries and examine chunk scores without launching Discord:
```bash
python scripts/ask.py "What are the hackathon rules regarding team size?"
python scripts/ask.py "Where can I get the presentation template for round 1?"
python scripts/ask.py "How do I get to GNIT Kolkata campus from Sodepur station?"
```

---

## Discord Slash Commands

| Slash Command | Description | Permission |
|---|---|---|
| `/status` | Displays bot health, LLM provider, chunk count, and database metrics | Core Member / Volunteer / Admin |
| `/reloadkb` | Re-scans `knowledge/` and rebuilds the FAISS index on the fly without downtime | Core Member / Volunteer / Admin |
| `/ask <question>` | Ask any hackathon question directly via slash command | Everyone |
| `/asktest <question>` | Privately tests question retrieval, showing sources and cosine similarity scores | Core Member / Volunteer / Admin |
| `/clearcache` | Clears conversation history sliding window for the channel or server | Core Member / Volunteer / Admin |
| `/unanswered [limit]` | Lists questions that triggered fallback to identify gaps in FAQ docs | Core Member / Volunteer / Admin |

---

## 24/7 Production Deployment

### Option 1: Docker & Docker Compose (Recommended)

1. Make sure `.env` is configured with your `DISCORD_TOKEN` and API keys.
2. Build and start the container in detached mode:
```bash
docker compose up -d --build
```
3. View logs:
```bash
docker compose logs -f
```
4. Stop container:
```bash
docker compose down
```

### Option 2: Linux `systemd` Service (Ubuntu / Debian / EC2)

1. Clone repo to `/home/ubuntu/Recur` and set up Python venv:
```bash
cd /home/ubuntu/Recur
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
python scripts/index_knowledge.py
```
2. Copy the unit file:
```bash
sudo cp hackbot.service /etc/systemd/system/hackbot.service
```
3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable hackbot
sudo systemctl start hackbot
```
4. Check status & logs:
```bash
sudo systemctl status hackbot
sudo journalctl -u hackbot -f
```

### Option 3: PM2 Process Manager
```bash
npm install -g pm2
pm2 start bot.py --name "recur-hackbot" --interpreter python
pm2 save
pm2 startup
```

---

## Automated Testing

Run the full pytest suite (all 17 unit tests):
```bash
python -m pytest tests/ -v
```

Run static code analysis:
```bash
python -m pyflakes .
```
