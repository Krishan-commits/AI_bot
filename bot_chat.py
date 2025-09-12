# -*- coding: utf-8 -*-
import os, json, time, sqlite3, asyncio, logging, traceback
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from collections import deque
from typing import List, Optional, Tuple

import httpx, pandas as pd
from dotenv import load_dotenv
import google.generativeai as genai
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import BadRequest
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    CallbackQueryHandler, PollHandler, ContextTypes, filters
)
from telegram.request import HTTPXRequest

# ---- config ----
load_dotenv()
os.makedirs("logs", exist_ok=True)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash-lite')

MAX_MSG_LEN = 4096
ITEMS_PER_PAGE = 20
RATE_LIMIT = 50
CACHE_SIZE = 10000
NUM_WORKERS = 5

HELP_TEXT = """Hi {name}! I'm your MCQ explainer bot.

Commands:
/AI [-s|-d] [-hi|-en] — Explain latest/replied MCQ
/explain — Pick multiple MCQs/polls to explain
/polls — List stored polls
/poll <id> — Show poll details
/search <keyword> — Search MCQs
/export <csv|pdf> — Export stored MCQs
/stats — Usage stats
/leaderboard — Top users
/queue — Pending explanations
/help — This message
"""

# ---- compact, colored logger (console) + plain file logger ----
class ShortColorFormatter(logging.Formatter):
    COLORS = {
        'INFO': '\033[37m',   # dim white
        'POLL': '\033[33m',   # soft yellow
        'CMD': '\033[32m',    # soft green
        'QUEUE': '\033[35m',  # magenta
        'WORKER': '\033[34m', # blue
        'GEMINI': '\033[31m', # muted red
        'ERROR': '\033[91m',  # bright red
        'ENDC': '\033[0m'
    }
    def format(self, record):
        tag = getattr(record, 'log_type', record.levelname)
        color = self.COLORS.get(tag, self.COLORS['INFO'])
        now = datetime.now().strftime('%H:%M:%S')
        return f"{color}{now} ▶ {tag}: {record.getMessage()}{self.COLORS['ENDC']}"


logger = logging.getLogger("MCQBot")
logger.setLevel(logging.INFO)

# console (colored)
ch = logging.StreamHandler()
ch.setFormatter(ShortColorFormatter())
logger.addHandler(ch)

# file (plain, rotating)
fh = TimedRotatingFileHandler("logs/bot.log", when="midnight", backupCount=7, encoding="utf-8")
fh.setFormatter(logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s"))
logger.addHandler(fh)
logger.propagate = False

def log_message(msg: str, tag: str = "INFO"):
    rec = logger.makeRecord(name="MCQBot", level=logging.INFO, fn='', lno=0, msg=msg, args=(), exc_info=None)
    rec.log_type = tag
    logger.handle(rec)

# ---- DB helpers ----
DB_PATH = "mcq_bot.db"
def db_connect():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    c = db_connect()
    cur = c.cursor()
    cur.executescript("""
    CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, message_id TEXT UNIQUE, text TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, user_id INTEGER, username TEXT, chat_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS polls (
        id INTEGER PRIMARY KEY AUTOINCREMENT, poll_id TEXT UNIQUE, question TEXT, options_json TEXT,
        correct_option_id INTEGER, type TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        user_id INTEGER, username TEXT, chat_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS explanation_cache (
        id INTEGER PRIMARY KEY AUTOINCREMENT, question_hash TEXT UNIQUE, question_text TEXT,
        options_text TEXT, explanation_type TEXT, language TEXT, explanation TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE IF NOT EXISTS rate_limits (
        user_id INTEGER PRIMARY KEY, request_count INTEGER, last_reset REAL
    );
    CREATE TABLE IF NOT EXISTS user_stats (
        user_id INTEGER PRIMARY KEY, username TEXT, explanations_requested INTEGER DEFAULT 0,
        explanations_hi INTEGER DEFAULT 0, explanations_en INTEGER DEFAULT 0,
        polls_created INTEGER DEFAULT 0, last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)
    c.commit()
    c.close()

init_db()

# ---- small utilities ----
def short_chunks(text: str, max_len: int = MAX_MSG_LEN, reserve: int = 200):
    safe = max_len - reserve
    if len(text) <= safe: return [text]
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 > safe:
            chunks.append(cur); cur = w
        else:
            cur = f"{cur} {w}" if cur else w
    if cur: chunks.append(cur)
    return chunks

async def send_long_message(chat_id, text, context, header="🤖 AI Explanation:", reply_to=None):
    chunks = short_chunks(text, MAX_MSG_LEN, reserve=len(header)+50)
    for i, chunk in enumerate(chunks):
        pref = f"{header}\n\n" if i == 0 else f"{header} (Part {i+1}/{len(chunks)})\n\n"
        await context.bot.send_message(chat_id=chat_id, text=pref + chunk, reply_to_message_id=reply_to)

async def auto_delete(chat_id, message_ids, context, delay=5):
    await asyncio.sleep(delay)
    for mid in message_ids:
        try: await context.bot.delete_message(chat_id=chat_id, message_id=mid)
        except Exception: pass

async def delete_after(msg, delay=5):
    await asyncio.sleep(delay)
    try: await msg.delete()
    except Exception: pass

# ---- logging wrapper for record-style logs ----
def lm(msg, tag="INFO"):
    log_message(msg, tag)

# ---- persistent stores & helpers (compact) ----
class MessageStore:
    def __init__(self):
        self.conn = db_connect()

    def add_message(self, text, message_id=None, user_id=None, username=None, chat_id=None):
        try:
            cur = self.conn.cursor()
            cur.execute("INSERT OR REPLACE INTO messages (message_id, text, user_id, username, chat_id) VALUES (?, ?, ?, ?, ?)",
                        (str(message_id) if message_id else None, text, user_id, username, chat_id))
            self.conn.commit()
        except Exception as e:
            lm(f"Error storing message: {e}", "INFO")

    def add_poll(self, poll_obj, message_id=None, user_id=None, username=None, chat_id=None):
        try:
            poll_id = str(getattr(poll_obj, 'id', None)) or str(message_id)
            options = [opt.text for opt in getattr(poll_obj, 'options', [])] if getattr(poll_obj, 'options', None) else []
            options_json = json.dumps(options)
            cur = self.conn.cursor()
            cur.execute('''INSERT OR REPLACE INTO polls
                (poll_id, question, options_json, correct_option_id, type, user_id, username, chat_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (poll_id, getattr(poll_obj, 'question', ''), options_json,
                 getattr(poll_obj, 'correct_option_id', None), getattr(poll_obj, 'type', ''), user_id, username, chat_id))
            self.conn.commit()
            q = getattr(poll_obj, 'question', '').replace("\n"," ").strip()[:160]
            lm(f"Poll {poll_id} Submitted by '{username}' [{q}]", "POLL")
        except Exception as e:
            lm(f"Error storing poll: {e}", "INFO")
            self.conn.rollback()

    def get_latest_mcq(self, chat_id=None):
        try:
            cur = self.conn.cursor()
            if chat_id:
                cur.execute("SELECT * FROM polls WHERE chat_id = ? ORDER BY timestamp DESC LIMIT 1", (chat_id,))
            else:
                cur.execute("SELECT * FROM polls ORDER BY timestamp DESC LIMIT 1")
            p = cur.fetchone()
            if p:
                return {"type":"poll","id":p["poll_id"], "poll":{
                    "question": p["question"], "options": json.loads(p["options_json"]), "correct_option_id": p["correct_option_id"], "type": p["type"]
                }}
            if chat_id:
                cur.execute("SELECT * FROM messages WHERE chat_id = ? ORDER BY timestamp DESC", (chat_id,))
            else:
                cur.execute("SELECT * FROM messages ORDER BY timestamp DESC")
            msgs = cur.fetchall()
            for m in msgs:
                if is_mcq(m["text"]):
                    return {"type":"text","id":m["message_id"], "text":m["text"]}
            return None
        except Exception as e:
            lm(f"Error getting latest MCQ: {e}", "INFO")
            return None

    def get_mcq_by_id(self, msgid):
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT * FROM polls WHERE poll_id = ?", (str(msgid),))
            p = cur.fetchone()
            if p:
                return {"type":"poll","id":p["poll_id"], "poll": {"question": p["question"], "options": json.loads(p["options_json"]), "correct_option_id": p["correct_option_id"], "type": p["type"]}}
            cur.execute("SELECT * FROM messages WHERE message_id = ?", (str(msgid),))
            m = cur.fetchone()
            if m and is_mcq(m["text"]): return {"type":"text","id":m["message_id"], "text":m["text"]}
            return None
        except Exception as e:
            lm(f"Error getting MCQ by ID: {e}", "INFO")
            return None

    def get_all_mcqs(self, limit=50):
        try:
            out=[]
            cur=self.conn.cursor()
            cur.execute("SELECT * FROM polls ORDER BY timestamp DESC LIMIT ?", (limit,))
            for p in cur.fetchall():
                out.append({"id":p["poll_id"],"type":"poll","data":{"poll":{"question":p["question"],"options":json.loads(p["options_json"]),"correct_option_id":p["correct_option_id"],"type":p["type"]}}})
            cur.execute("SELECT * FROM messages ORDER BY timestamp DESC LIMIT ?", (limit,))
            for m in cur.fetchall():
                if is_mcq(m["text"]): out.append({"id":m["message_id"],"type":"text","data":{"text":m["text"]}})
            return out
        except Exception as e:
            lm(f"Error get_all_mcqs: {e}", "INFO"); return []

    def find_poll_submitter(self, poll_id) -> str:
        # Try to find username who submitted poll earlier (if any)
        try:
            cur = self.conn.cursor()
            cur.execute("SELECT username FROM polls WHERE poll_id = ? LIMIT 1", (str(poll_id),))
            r = cur.fetchone()
            if r and r["username"]: return r["username"]
        except Exception as e:
            lm(f"DB lookup error for poll submitter: {e}", "INFO")
        return "Unknown"

def is_mcq(text: str) -> bool:
    if not text: return False
    lines = text.splitlines()
    if len(lines) < 2: return False
    opts = 0
    for ln in lines[1:]:
        ln = ln.strip()
        if ln and len(ln) > 1 and ln[0].upper() in "ABCD" and ln[1] in [')','.','-',':']: opts += 1
    return opts >= 2

# ---- cache + rate limiter (compact) ----
class ExplanationCache:
    def __init__(self, max_size=CACHE_SIZE):
        self.cache = {}
        self.max_size = max_size
        self.conn = db_connect()

    def key(self, question, options, explanation_type, language):
        opts = "|".join(options) if options else ""
        return f"{hash(question + opts + explanation_type + language)}"

    def get(self, question, options, explanation_type, language):
        try:
            k = self.key(question, options, explanation_type, language)
            if k in self.cache: return self.cache[k]
            cur = self.conn.cursor()
            cur.execute("SELECT explanation FROM explanation_cache WHERE question_hash = ?", (k,))
            r = cur.fetchone()
            if r:
                self.cache[k] = r[0]; return r[0]
        except Exception as e:
            lm(f"Cache get err: {e}", "INFO")
        return None

    def set(self, question, options, explanation_type, language, explanation):
        try:
            k = self.key(question, options, explanation_type, language)
            self.cache[k] = explanation
            if len(self.cache) > self.max_size:
                oldest = next(iter(self.cache)); del self.cache[oldest]
            cur = self.conn.cursor()
            cur.execute('''INSERT OR REPLACE INTO explanation_cache
                (question_hash, question_text, options_text, explanation_type, language, explanation)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (k, question, json.dumps(options or []), explanation_type, language, explanation))
            self.conn.commit()
        except Exception as e:
            lm(f"Cache set err: {e}", "INFO")

class RateLimiter:
    def __init__(self, max_per_min=RATE_LIMIT):
        self.max = max_per_min
        self.conn = db_connect()
    def check(self, user_id:int)->bool:
        try:
            now = time.time()
            cur = self.conn.cursor()
            cur.execute("SELECT request_count, last_reset FROM rate_limits WHERE user_id = ?", (user_id,))
            r = cur.fetchone()
            if not r:
                cur.execute("INSERT INTO rate_limits (user_id, request_count, last_reset) VALUES (?,1,?)", (user_id, now))
                self.conn.commit(); return True
            cnt, last = r["request_count"], r["last_reset"]
            if now - last > 60:
                cur.execute("UPDATE rate_limits SET request_count = 1, last_reset = ? WHERE user_id = ?", (now, user_id)); self.conn.commit(); return True
            if cnt >= self.max: return False
            cur.execute("UPDATE rate_limits SET request_count = request_count + 1 WHERE user_id = ?", (user_id,)); self.conn.commit(); return True
        except Exception as e:
            lm(f"RateLimiter err: {e}", "INFO"); return True

# ---- Gemini explainer compact ----
class GeminiExplainer:
    def __init__(self):
        self.model = gemini_model
        self.cache = ExplanationCache()
        self.rate = RateLimiter()

    async def generate_explanation(self, question:str, options:List[str], correct_answer:Optional[str]=None,
                                   explanation_type:str="short", language:str="hi", user_id:Optional[int]=None, exam_name:str=""):
        if user_id and not self.rate.check(user_id): return "Rate limit exceeded. Please try again later."
        cached = self.cache.get(question, options, explanation_type, language)
        if cached:
            qd = question.replace("\n"," ").strip()[:60]
            lm(f"[Cache] {qd} [{explanation_type},{language}]", "GEMINI")
            return cached
        try:
            opts_txt = "\n".join([f"{chr(65+i)}. {o}" for i,o in enumerate(options or [])])
            lang_instr = "Explain in English." if language == "en" else "Explain in Hindi."
            brev = "concise" if explanation_type=="short" else "comprehensive"
            prompt = f"Question: {question}\nOptions:\n{opts_txt}\n"
            if correct_answer: prompt += f"Correct Answer: {correct_answer}\n"
            prompt += f"{lang_instr} Provide a {brev} explanation."
            qd = question.replace("\n"," ").strip()[:60]
            lm(f"Processing: {qd} [{explanation_type},{language}]", "GEMINI")
            response = await asyncio.to_thread(self.model.generate_content, prompt)
            text = getattr(response, "text", str(response))
            self.cache.set(question, options, explanation_type, language, text)
            return text
        except Exception as e:
            lm(f"Gemini Error: {e}", "GEMINI")
            return f"Error: could not generate explanation. {e}"

# ---- global singletons ----
message_store = MessageStore()
explainer = GeminiExplainer()
explanation_queue = asyncio.Queue()
pending_explanations = {}
# note: we don't log "Worker X started" to reduce noise

# ---- worker ----
async def explanation_worker(worker_id:int):
    while True:
        job = await explanation_queue.get()
        try:
            q_display = job["question"].replace("\n"," ").strip()[:60]
            lm(f"W{worker_id} processing: {q_display} [{job['type']},{job['lang']}]", "WORKER")
            explanation = await explainer.generate_explanation(
                job["question"], job["options"], job.get("correct_answer"),
                job["type"], job["lang"], job["user_id"], job.get("exam_name","")
            )
            header = f"💡 {job['type'].title()} Explanation ({'English' if job['lang']=='en' else 'Hindi'})"
            await send_long_message(job["chat_id"], explanation, job["context"], header=header, reply_to=job.get("message_id"))
            lm(f"W{worker_id} completed: {q_display}", "WORKER")
        except Exception as e:
            lm(f"W{worker_id} error: {e}", "WORKER")
            try: await job["context"].bot.send_message(job["chat_id"], "⚠️ Error generating explanation. Please try again.")
            except Exception: pass
        finally:
            explanation_queue.task_done()

# ---- helpers to extract MCQ data ----
def extract_mcq_data(mcq_data) -> Tuple[str, List[str], Optional[str], str]:
    q, options, correct, exam = "", [], None, ""
    if "poll" in mcq_data:
        p = mcq_data["poll"]
        raw = p.get("question","") or ""
        if "\n" in raw:
            exam, q = raw.split("\n",1)[0].strip(), raw.split("\n",1)[1].strip()
        else:
            q = raw.strip()
        options = p.get("options", []) or []
        if p.get("type") == "quiz" and p.get("correct_option_id") is not None:
            idx = p.get("correct_option_id")
            if idx < len(options): correct = options[idx]
    else:
        txt = mcq_data.get("text","")
        lines = txt.splitlines()
        q = lines[0].strip() if lines else ""
        for ln in lines[1:]:
            if ln.strip() and ln[0].upper() in "ABCD" and ln[1] in [')','.','-',':']:
                options.append(ln[2:].strip())
        for ln in lines:
            if any(k in ln.lower() for k in ["answer:", "ans:", "correct:"]):
                ans = ln.split(":",1)[1].strip()
                if len(ans)==1 and ans.upper() in "ABCD":
                    idx = ord(ans.upper()) - 65
                    if idx < len(options): correct = options[idx]
                else: correct = ans
                break
    return q, options, correct, exam

# ---- telegram handlers (compact & reused helpers) ----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    lm(f"User {user.first_name} ({user.id}) started the bot", "CMD")
    kb = [
        [InlineKeyboardButton("📊 Stats", callback_data="show_stats"), InlineKeyboardButton("🏆 Leaderboard", callback_data="show_leaderboard")],
        [InlineKeyboardButton("🔍 Search", switch_inline_query_current_chat=""), InlineKeyboardButton("📤 Export", callback_data="show_export")]
    ]
    resp = await update.message.reply_text(HELP_TEXT.format(name=user.first_name), reply_markup=InlineKeyboardMarkup(kb))
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, resp.message_id], context, 60))

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE): await start(update, context)

async def polls_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user; lm(f"User {user.first_name} ({user.id}) requested polls list", "CMD")
    mcqs = message_store.get_all_mcqs(limit=20)
    polls=[m for m in mcqs if m["type"]=="poll"]
    if not polls:
        r = await update.message.reply_text("No stored polls.")
    else:
        lines=[]
        for idx,p in enumerate(polls,1):
            q = p["data"]["poll"]["question"]
            if "\n" in q: q = q.split("\n",1)[1].strip()
            lines.append(f"{idx}. {p['id']} - {q[:80]}")
        r = await update.message.reply_text("Stored polls:\n" + "\n".join(lines))
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

async def poll_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user; lm(f"User {user.first_name} ({user.id}) requested poll details", "CMD")
    if not context.args:
        r = await update.message.reply_text("Usage: /poll <id>")
        asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30)); return
    pid = context.args[0]
    p = message_store.get_mcq_by_id(pid)
    if not p or "poll" not in p:
        r = await update.message.reply_text("Poll not found.")
    else:
        poll = p["poll"]; opts = poll["options"]
        text = f"Poll id={p['id']}\nQuestion: {poll['question']}\nOptions:\n" + "\n".join([f"{i}. {o}" for i,o in enumerate(opts,1)])
        r = await update.message.reply_text(text)
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user; lm(f"User {user.first_name} ({user.id}) opened explain menu", "CMD")
    mcqs = message_store.get_all_mcqs()
    if not mcqs:
        r = await update.message.reply_text("No recent MCQs or polls found.")
        asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30)); return
    uid = str(user.id)
    pending_explanations[uid] = {"selected": {mcqs[0]['id']} if mcqs else set(), "type":"short","lang":"hi","page":0}
    await send_explain_menu(update.message, mcqs, uid, context)
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id], context, 500))

async def send_explain_menu(message, mcqs, uid, context):
    try:
        state = pending_explanations[uid]; page = state["page"]
        start = page*ITEMS_PER_PAGE; items = mcqs[start:start+ITEMS_PER_PAGE]
        kb=[]
        for m in items:
            if m["type"]=="poll":
                q = m["data"]["poll"]["question"]; q = q.split("\n",1)[1].strip() if "\n" in q else q
            else: q = m["data"]["text"].split("\n")[0]
            sel = "✔" if str(m["id"]) in state["selected"] else " "
            label = q[:40] + ("..." if len(q)>40 else "")
            kb.append([InlineKeyboardButton(f"[{sel}] {label}", callback_data=f"toggle_{m['id']}")])
        nav=[]
        if page>0: nav.append(InlineKeyboardButton("⬅ Prev", callback_data="prev"))
        if (page+1)*ITEMS_PER_PAGE < len(mcqs): nav.append(InlineKeyboardButton("Next ➡", callback_data="next"))
        if nav: kb.append(nav)
        kb.append([InlineKeyboardButton("✅ English" if state["lang"]=="en" else "English", callback_data="setlang_en"),
                   InlineKeyboardButton("✅ Hindi" if state["lang"]=="hi" else "Hindi", callback_data="setlang_hi")])
        kb.append([InlineKeyboardButton("✅ Short" if state["type"]=="short" else "Short", callback_data="settype_short"),
                   InlineKeyboardButton("✅ Detailed" if state["type"]=="detailed" else "Detailed", callback_data="settype_detailed")])
        kb.append([InlineKeyboardButton("✅ Generate", callback_data="generate"), InlineKeyboardButton("❌ Cancel", callback_data="cancel")])
        menu = await message.reply_text("Select MCQs/Polls to explain:", reply_markup=InlineKeyboardMarkup(kb))
        pending_explanations[uid]["menu_message_id"] = menu.message_id
    except Exception as e:
        lm(f"send_explain_menu err: {e}", "INFO"); await message.reply_text("❌ An error occurred. Please try /explain again.")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query; await q.answer()
    user = q.from_user; uid = str(user.id)
    # Try delete previous menu to refresh
    if uid in pending_explanations and "menu_message_id" in pending_explanations[uid]:
        try: await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=pending_explanations[uid]["menu_message_id"])
        except Exception: pass
    # protect expired sessions
    if (q.data.startswith("toggle_") or q.data in ["prev","next","setlang_hi","setlang_en","settype_short","settype_detailed","cancel","generate"]) and uid not in pending_explanations:
        await q.edit_message_text("Session expired. Use /explain again."); return
    try:
        mcqs = message_store.get_all_mcqs()
        if q.data.startswith("toggle_"):
            mid = q.data.split("_",1)[1]
            if mid in pending_explanations[uid]["selected"]: pending_explanations[uid]["selected"].remove(mid)
            else: pending_explanations[uid]["selected"].add(mid)
        elif q.data == "prev":
            if pending_explanations[uid]["page"]>0: pending_explanations[uid]["page"] -= 1
        elif q.data == "next":
            if (pending_explanations[uid]["page"]+1)*ITEMS_PER_PAGE < len(mcqs): pending_explanations[uid]["page"] += 1
        elif q.data.startswith("setlang_"): pending_explanations[uid]["lang"] = q.data.split("_",1)[1]
        elif q.data.startswith("settype_"): pending_explanations[uid]["type"] = q.data.split("_",1)[1]
        elif q.data == "cancel":
            del pending_explanations[uid]; await q.edit_message_text("Cancelled."); return
        elif q.data == "generate":
            state = pending_explanations[uid]
            if not state["selected"]: await q.edit_message_text("No MCQs selected."); return
            ids = list(state["selected"])
            sel = [message_store.get_mcq_by_id(i) for i in ids if message_store.get_mcq_by_id(i)]
            if not sel: await q.edit_message_text("No valid MCQs selected."); return
            queue_msgs=[]
            for idx, mcq in enumerate(sel,1):
                question, options, correct, exam = extract_mcq_data(mcq)
                pos = explanation_queue.qsize() + 1
                status_msg = await q.message.reply_text(f"🔄 Your explanation request (MCQ {idx}/{len(sel)}) has been queued. You are #{pos} in the queue.")
                queue_msgs.append(status_msg.message_id)
                reply_to_id = q.message.message_id if q.message else None
                await explanation_queue.put({
                    "user_id": user.id, "chat_id": update.effective_chat.id, "message_id": reply_to_id,
                    "question": question, "options": options, "correct_answer": correct,
                    "exam_name": exam, "lang": state["lang"], "type": state["type"], "context": context
                })
                update_user_stats(user.id, user.first_name, explanations=1, lang=state["lang"])
            asyncio.create_task(auto_delete(update.effective_chat.id, queue_msgs, context, 10))
            del pending_explanations[uid]
            await q.edit_message_text(f"✅ Queued {len(sel)} explanation(s).")
            return
        elif q.data == "show_stats":
            rows = get_user_stats(user.id)
            if not rows: await q.message.reply_text("No statistics available yet.")
            else:
                text = f"📊 Your Statistics:\n• Explanations requested: {rows['explanations_requested']} (Hi: {rows['explanations_hi']}, En: {rows['explanations_en']})\n• Polls created: {rows['polls_created']}\n• Last active: {rows['last_active']}\n"
                await q.message.reply_text(text)
        elif q.data == "show_leaderboard":
            rows = get_leaderboard(5)
            if not rows: await q.message.reply_text("No data for leaderboard yet.")
            else:
                lines=["🏆 Leaderboard:"]
                for i,rw in enumerate(rows,1):
                    uname = rw["username"] or "Unknown"
                    lines.append(f"{i}. {uname} — {rw['explanations_requested']} (Hi: {rw['explanations_hi']}, En: {rw['explanations_en']})")
                await q.message.reply_text("\n".join(lines))
        elif q.data == "show_export":
            await q.message.reply_text("Use /export csv or /export pdf to download your MCQs.")
        # re-render menu if still active
        if uid in pending_explanations:
            await send_explain_menu(q.message, mcqs, uid, context)
    except Exception as e:
        lm(f"Error in handle_callback: {e}", "INFO"); await q.message.reply_text("⚠️ An error occurred while processing your request.")

async def ai_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user
    args=context.args or []
    explanation_type = "detailed" if "-d" in args else "short"
    language = "en" if "-en" in args else "hi"
    lm(f"User {user.first_name} ({user.id}) used /AI [{explanation_type},{language}]", "CMD")
    mcqs=[]
    if update.message.reply_to_message:
        rep = update.message.reply_to_message
        if rep.poll:
            mcqs.append({"type":"poll","id":rep.poll.id,"poll":{"question":rep.poll.question,"options":[opt.text for opt in rep.poll.options],"correct_option_id":rep.poll.correct_option_id,"type":rep.poll.type}})
        else:
            m = message_store.get_mcq_by_id(str(rep.message_id))
            if m: mcqs.append(m)
    if not mcqs:
        latest = message_store.get_latest_mcq(update.effective_chat.id)
        if latest: mcqs.append(latest)
    if not mcqs:
        await update.message.reply_text("No MCQ found. Please send or reply to a MCQ."); return
    qtxt = extract_mcq_data(mcqs[0])[0][:60]
    lm(f"Queued: '{qtxt}'", "QUEUE")
    status = await update.message.reply_text("🔄 Generating explanation...")
    asyncio.create_task(delete_after(status, 5)); asyncio.create_task(delete_after(update.message,5))
    queue_msgs=[]
    for idx, mcq in enumerate(mcqs,1):
        question, options, correct, exam = extract_mcq_data(mcq)
        pos = explanation_queue.qsize() + 1
        s = await update.message.reply_text(f"🔄 Your explanation request (MCQ {idx}/{len(mcqs)}) has been queued. You are #{pos} in the queue.")
        queue_msgs.append(s.message_id)
        await explanation_queue.put({
            "user_id": user.id, "chat_id": update.effective_chat.id,
            "message_id": update.message.reply_to_message.message_id if update.message.reply_to_message else update.message.message_id,
            "question": question, "options": options, "correct_answer": correct,
            "exam_name": exam, "lang": language, "type": explanation_type, "context": context
        })
    asyncio.create_task(auto_delete(update.effective_chat.id, queue_msgs, context, 10))
    update_user_stats(user.id, user.first_name, explanations=1, lang=language)

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.message.text:
        u = update.effective_user
        message_store.add_message(update.message.text, str(update.message.message_id), u.id, u.first_name, update.effective_chat.id)

async def handle_poll_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # a user posted a poll message (full message object)
    if update.message and update.message.poll:
        usr = update.message.from_user
        uid = usr.id if usr else 0
        uname = (usr.username or usr.first_name) if usr else "Admin"
        message_store.add_poll(update.message.poll, str(update.message.poll.id), uid, uname, update.effective_chat.id)
        update_user_stats(uid, uname, polls=1)

async def handle_poll_update(update: Update, context: ContextTypes.DEFAULT_TYPE):
    # the API sometimes sends poll updates (without message.from_user)
    if update.poll:
        poll_id = getattr(update.poll, 'id', None)
        username = message_store.find_poll_submitter(poll_id)  # try DB lookup
        # store/update poll record with username found (user_id left 0 if unknown)
        message_store.add_poll(update.poll, poll_id, 0, username, None)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    lm(f"Exception while handling update: {context.error}", "ERROR")
    if update and hasattr(update,'effective_chat'):
        try: await context.bot.send_message(chat_id=update.effective_chat.id, text="An error occurred while processing your request. Please try again.")
        except Exception: pass

async def queue_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    qsize = explanation_queue.qsize()
    if qsize == 0:
        r = await update.message.reply_text("✅ The explanation queue is empty.")
    else:
        r = await update.message.reply_text(f"⏳ There are currently {qsize} explanation request(s) in the queue.")
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

# ---- user stats helpers ----
def update_user_stats(user_id, username, explanations=0, polls=0, lang=None):
    try:
        conn = db_connect(); cur = conn.cursor()
        exp_hi = 1 if (explanations and lang=="hi") else 0
        exp_en = 1 if (explanations and lang=="en") else 0
        cur.execute('''INSERT INTO user_stats (user_id, username, explanations_requested, explanations_hi, explanations_en, polls_created, last_active)
            VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(user_id) DO UPDATE SET
                username=excluded.username,
                explanations_requested=explanations_requested + excluded.explanations_requested,
                explanations_hi=explanations_hi + excluded.explanations_hi,
                explanations_en=explanations_en + excluded.explanations_en,
                polls_created=polls_created + excluded.polls_created,
                last_active=CURRENT_TIMESTAMP
        ''', (user_id, username, explanations, exp_hi, exp_en, polls))
        conn.commit(); conn.close()
    except Exception as e:
        lm(f"update_user_stats err: {e}", "INFO")

def get_user_stats(user_id):
    try:
        conn = db_connect(); conn.row_factory = sqlite3.Row; cur = conn.cursor()
        cur.execute("SELECT * FROM user_stats WHERE user_id = ?", (user_id,))
        r = cur.fetchone(); conn.close()
        return r
    except Exception as e:
        lm(f"get_user_stats err: {e}", "INFO"); return None

def get_leaderboard(limit=5):
    try:
        conn = db_connect(); conn.row_factory = sqlite3.Row; cur = conn.cursor()
        cur.execute("SELECT * FROM user_stats ORDER BY explanations_requested DESC LIMIT ?", (limit,))
        rows = cur.fetchall(); conn.close(); return rows
    except Exception as e:
        lm(f"get_leaderboard err: {e}", "INFO"); return []

# ---- search, export, stats, leaderboard handlers ----
async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.effective_user; s=get_user_stats(user.id)
    if not s: r = await update.message.reply_text("No statistics available yet.")
    else:
        text = f"📊 Your Statistics:\n• Explanations requested: {s['explanations_requested']} (Hindi: {s['explanations_hi']}, English: {s['explanations_en']})\n• Polls created: {s['polls_created']}\n• Last active: {s['last_active']}\n"
        r = await update.message.reply_text(text)
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

async def leaderboard_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    rows = get_leaderboard(5)
    if not rows: r = await update.message.reply_text("No data for leaderboard yet.")
    else:
        lines=["🏆 Leaderboard:"]
        for i,row in enumerate(rows,1):
            uname = row["username"] or "Unknown"
            lines.append(f"{i}. {uname} — {row['explanations_requested']} (Hi: {row['explanations_hi']}, En: {row['explanations_en']})")
        r = await update.message.reply_text("\n".join(lines))
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

async def search_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        r = await update.message.reply_text("Usage: /search <keyword>")
        asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30)); return
    kw = " ".join(context.args).strip()
    conn = db_connect(); cur = conn.cursor()
    cur.execute("SELECT * FROM polls WHERE question LIKE ? ORDER BY timestamp DESC LIMIT 10", (f"%{kw}%",))
    polls = cur.fetchall()
    cur.execute("SELECT * FROM messages WHERE text LIKE ? ORDER BY timestamp DESC LIMIT 10", (f"%{kw}%",))
    messages = cur.fetchall()
    conn.close()
    if not polls and not messages:
        r = await update.message.reply_text(f"No MCQs found for '{kw}'.")
    else:
        lines=[f"🔍 Search results for '{kw}':"]
        for p in polls: lines.append(f"• [Poll] {p['question'][:80]}")
        for m in messages: lines.append(f"• [Text] {m['text'][:80]}")
        r = await update.message.reply_text("\n".join(lines[:20]))
    asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        r = await update.message.reply_text("Usage: /export <csv|pdf>")
        asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30)); return
    fmt = context.args[0].lower()
    conn = db_connect(); conn.row_factory=sqlite3.Row; cur = conn.cursor()
    cur.execute("SELECT question, options_json, correct_option_id FROM polls ORDER BY timestamp DESC")
    polls = cur.fetchall(); conn.close()
    if not polls:
        r = await update.message.reply_text("No data to export.")
        asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30)); return
    if fmt == "csv":
        data=[]
        for p in polls:
            opts = json.loads(p["options_json"]); correct = opts[p["correct_option_id"]] if p["correct_option_id"] is not None and p["correct_option_id"] < len(opts) else ""
            data.append({"Question":p["question"], "Options":" | ".join(opts), "Correct":correct})
        df = pd.DataFrame(data); fn = "mcqs_export.csv"; df.to_csv(fn, index=False, encoding="utf-8")
        await update.message.reply_document(document=open(fn,"rb"), filename=fn); os.remove(fn)
    elif fmt == "pdf":
        fn = "mcqs_export.pdf"; doc = SimpleDocTemplate(fn); styles = getSampleStyleSheet(); story=[]
        for p in polls[:50]:
            opts=json.loads(p["options_json"])
            text = f"Q: {p['question']}\n" + "\n".join([f"{chr(65+i)}. {o}" for i,o in enumerate(opts)]) + ("\n" + f"✅ Correct: {opts[p['correct_option_id']]}" if p["correct_option_id"] is not None and p["correct_option_id"] < len(opts) else "")
            story.append(Paragraph(text, styles["Normal"])); story.append(Spacer(1,12))
        doc.build(story); await update.message.reply_document(document=open(fn,"rb"), filename=fn); os.remove(fn)
    else:
        r = await update.message.reply_text("Invalid format. Use /export csv or /export pdf")
        asyncio.create_task(auto_delete(update.effective_chat.id, [update.message.message_id, r.message_id], context, 30))

# ---- startup helpers ----
async def on_startup(app):
    # spawn workers but don't log "Worker X started" repeatedly
    for i in range(NUM_WORKERS):
        asyncio.create_task(explanation_worker(i))
    lm(f"✅ Bot started with Gemini AI and {NUM_WORKERS} explanation workers", "INFO")

def build_app():
    request = HTTPXRequest(connect_timeout=30, read_timeout=30, pool_timeout=30, http_version="1.1")
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).request(request).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("AI", ai_command))
    app.add_handler(CommandHandler("explain", explain_command))
    app.add_handler(CommandHandler("polls", polls_command))
    app.add_handler(CommandHandler("poll", poll_command))
    app.add_handler(CommandHandler("queue", queue_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("leaderboard", leaderboard_command))
    app.add_handler(CommandHandler("search", search_command))
    app.add_handler(CommandHandler("export", export_command))
    app.add_handler(CallbackQueryHandler(handle_callback))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_handler(MessageHandler(filters.POLL, handle_poll_message))
    app.add_handler(PollHandler(handle_poll_update))
    app.add_error_handler(error_handler)
    app.post_init = on_startup
    return app

def main():
    lm("Bot starting with AI...", "INFO")
    app = build_app()
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        lm("Bot stopped manually with Ctrl+C", "INFO")
    except Exception as e:
        lm(f"Bot crashed with error: {e}", "ERROR")
        lm("Restarting bot in 5 seconds...", "INFO")
        time.sleep(5)
        os.execv(__file__, ["python"] + [])
