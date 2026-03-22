# aosa Bakehouse & Roastery — Ordering Platform

Flask + React + SQLite + LangChain (Gemini 2.0 Flash)

---

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in your keys
export $(cat .env | xargs)
python app.py
# → http://localhost:5555
```

---

## Table QR Code Flow

**Admin setup (one-time):**
1. Go to Admin → Venues
2. Enter prefix (e.g. "Table") and count (e.g. 20) → click **Generate QR Codes**
3. All 20 QR codes appear instantly — click **🖨 Print All** to print them
4. Stick each QR on the matching table

**Customer experience:**
1. Customer sits down at Table 7
2. Opens phone camera → scans QR on the table
3. Browser opens `http://yoursite/?venue=abc&table=Table%207`
4. App loads instantly — skips splash, shows the menu directly
5. Order form has **Table 7 pre-filled and locked** (shown as "📍 Table 7 · Set by QR")

---

## Features

| Feature | Details |
|---|---|
| 📋 Table QR codes | Per-table QRs in admin; bulk generate up to 100 at once |
| 🤖 Mia (AI chat) | LangChain agent with full conversation memory & menu search tool |
| 💳 PayPal | Sandbox-ready; swap one env var for live |
| 📊 Analytics | Date-filtered charts, revenue, peak hours, top dishes |
| 🔧 Admin panel | Venues, tables, menu manager, live orders, analytics |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | — | Gemini API key (required for Mia) |
| `ADMIN_PASSWORD` | `admin123` | Admin panel password |
| `PAYPAL_CLIENT_ID` | placeholder | PayPal sandbox client ID |
| `PAYPAL_CLIENT_SECRET` | placeholder | PayPal sandbox secret |
