# ======================
# Standard Library
# ======================
import concurrent.futures
import hashlib
import html
import json
import os
import re
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from urllib.parse import parse_qs, urljoin, urlparse
from contextlib import contextmanager
from html import escape


# ======================
# Third-Party Libraries
# ======================
import requests
import streamlit as st
from bs4 import BeautifulSoup
import logging
import coloredlogs
import pandas as pd

# Configure colored console logging
coloredlogs.install(level='INFO', fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ======================
# Config / Globals
# ======================
# 🔒 Global lock for thread-safe logging and progress updates
log_lock = threading.Lock()
progress_lock = threading.Lock()

# --------- Global fallback variables for threads ---------
_global_log_messages = []
_global_scraping_progress = {
    "current": 0,
    "total": 0,
    "tid": None,
    "page": None,
    "status": "Not started"
}
_global_sending_progress = {
    "current": 0,
    "total": 0,
    "status": "Not started"
}

# --------- Thread control events ---------
scraping_active_flag = threading.Event()
scraping_paused_flag = threading.Event()
sending_active_flag = threading.Event()
sending_paused_flag = threading.Event()

# --------- Config ----------
BASE_URL = "https://www.rajasthangyan.com/question"
USER_AGENT = {"User-Agent": "Mozilla/5.0"}
DB_PATH = "mcq_bot.db"
TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

# Telegram limits
TG_Q_LIMIT = 300       # Question length
TG_OPT_LIMIT = 100     # Option length
TG_OPT_COUNT = 10      # Max options
TG_EXPL_LIMIT = 200    # Explanation length

# ======================
# Database Context Manager
# ======================
@contextmanager
def db_connection():
    """Context manager for database connections"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    try:
        yield conn
    finally:
        conn.close()

# ======================
# Session State Initialization
# ======================
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "log": "",
        "scraping_active": False,
        "sending_active": False,
        "scraping_paused": False,
        "sending_paused": False,
        "scraping_progress": 0,
        "sending_progress": 0,
        "total_to_scrape": 0,
        "total_to_send": 0,
        "current_tid_index": 0,
        "current_tid_page": 0,
        "scraped_count": 0,
        "sent_count": 0,
        "scraping_status": "Not started",
        "sending_status": "Not started",
        "current_tid": None,
        "log_messages": [],
        "last_update_time": 0,
        "db_initialized": False,
        "table_settings": {
            "show_full_question": False,
            "show_full_explanation": False,
            "show_full_options": False,
            "show_correct_option": True,
            "show_explanation": True,
            "show_options": True,
            "show_exam": True,
            "show_status": True,
            "show_id": True,
            "show_q_no": True,
            "column_order": ["ID", "Q.No", "Question", "Exam", "Options", "Correct Option", "Explanation", "Status"]
        },
        "selected_rows": pd.DataFrame(),
        "filtered_data": pd.DataFrame(),
        # New variables for thread-safe updates
        "last_log_update": 0,
        "last_progress_update": 0,
        # Thread status tracking
        "scraping_thread": None,
        "sending_thread": None,
        "scraping_stopped": True,
        "sending_stopped": True
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ======================
# Enhanced Logging System
# ======================
class LogLevel:
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    DEBUG = "DEBUG"

def log(message: str, level: str = LogLevel.INFO, source: str = "SYSTEM"):
    """
    Thread-safe logging function that works both inside and outside of Streamlit
    """
    global _global_log_messages
    
    timestamp = datetime.now().strftime("%H:%M:%S")
    log_entry = {
        "timestamp": timestamp,
        "message": message,
        "level": level,
        "source": source
    }
    
    # Log to console using coloredlogs
    if level == LogLevel.INFO:
        logger.info(f"[{source}] {message}")
    elif level == LogLevel.WARNING:
        logger.warning(f"[{source}] {message}")
    elif level == LogLevel.ERROR:
        logger.error(f"[{source}] {message}")
    elif level == LogLevel.SUCCESS:
        logger.info(f"[{source}] {message}")
    elif level == LogLevel.DEBUG:
        logger.debug(f"[{source}] {message}")
    
    # Always add to global log list (thread-safe)
    with log_lock:
        _global_log_messages.append(log_entry)
        if len(_global_log_messages) > 1000:
            _global_log_messages = _global_log_messages[-1000:]
    
    # Only allow Streamlit session access in main thread
    if threading.current_thread() == threading.main_thread():
        try:
            # Initialize log_messages if it doesn't exist
            if "log_messages" not in st.session_state:
                st.session_state["log_messages"] = []
            
            st.session_state.log_messages.append(log_entry)
            if len(st.session_state.log_messages) > 1000:
                st.session_state.log_messages = st.session_state.log_messages[-1000:]
            
            # Format for display
            level_colors = {
                LogLevel.INFO: "blue",
                LogLevel.WARNING: "orange",
                LogLevel.ERROR: "red",
                LogLevel.SUCCESS: "green",
                LogLevel.DEBUG: "gray"
            }
            
            color = level_colors.get(level, "black")
            formatted_message = f"<div style='margin-bottom: 5px; border-left: 3px solid {color}; padding-left: 10px;'><span style='color: gray; font-size: 0.8em;'>{timestamp}</span> <span style='color: {color}; font-weight: bold;'>[{level}]</span> <span style='font-weight: bold;'>{source}:</span> {html.escape(message)}</div>"
            
            # Initialize log if it doesn't exist
            if "log" not in st.session_state:
                st.session_state["log"] = ""
            
            # Prepend to log (newest on top)
            st.session_state.log = formatted_message + st.session_state.log
        except Exception as e:
            print(f"Logging error: {e}")

# ======================
# Progress Tracking System
# ======================
def update_scraping_progress(
    current: int, 
    total: int, 
    tid: Optional[int] = None, 
    page: Optional[int] = None,
    status: Optional[str] = None
):
    """Thread-safe scraping progress update that works both inside and outside of Streamlit"""
    global _global_scraping_progress
    
    # Update global progress
    with progress_lock:
        _global_scraping_progress["current"] = current
        _global_scraping_progress["total"] = total
        if tid is not None:
            _global_scraping_progress["tid"] = tid
        if page is not None:
            _global_scraping_progress["page"] = page
        if status is not None:
            _global_scraping_progress["status"] = status
    
    # Only allow Streamlit session access in main thread
    if threading.current_thread() == threading.main_thread():
        try:
            # Initialize session state variables if they don't exist
            if "current_tid_index" not in st.session_state:
                st.session_state["current_tid_index"] = 0
            if "total_to_scrape" not in st.session_state:
                st.session_state["total_to_scrape"] = 0
            if "current_tid" not in st.session_state:
                st.session_state["current_tid"] = None
            if "current_tid_page" not in st.session_state:
                st.session_state["current_tid_page"] = 0
            if "scraping_status" not in st.session_state:
                st.session_state["scraping_status"] = "Not started"
            if "scraping_progress" not in st.session_state:
                st.session_state["scraping_progress"] = 0
            
            # Update values
            if current is not None:
                st.session_state.current_tid_index = current
            if total is not None:
                st.session_state.total_to_scrape = total
            if tid is not None:
                st.session_state.current_tid = tid
            if page is not None:
                st.session_state.current_tid_page = page
            if status is not None:
                st.session_state.scraping_status = status
            
            # Calculate percentage
            if total > 0:
                st.session_state.scraping_progress = (current / total) * 100
        except Exception as e:
            print(f"Progress update error: {e}")

def update_sending_progress(
    current: int, 
    total: int, 
    status: Optional[str] = None
):
    """Thread-safe sending progress update that works both inside and outside of Streamlit"""
    global _global_sending_progress
    
    # Update global progress
    with progress_lock:
        _global_sending_progress["current"] = current
        _global_sending_progress["total"] = total
        if status is not None:
            _global_sending_progress["status"] = status
    
    # Only allow Streamlit session access in main thread
    if threading.current_thread() == threading.main_thread():
        try:
            # Initialize session state variables if they don't exist
            if "sent_count" not in st.session_state:
                st.session_state["sent_count"] = 0
            if "total_to_send" not in st.session_state:
                st.session_state["total_to_send"] = 0
            if "sending_status" not in st.session_state:
                st.session_state["sending_status"] = "Not started"
            if "sending_progress" not in st.session_state:
                st.session_state["sending_progress"] = 0
            
            # Update values
            if current is not None:
                st.session_state.sent_count = current
            if total is not None:
                st.session_state.total_to_send = total
            if status is not None:
                st.session_state.sending_status = status
            
            # Calculate percentage
            if total > 0:
                st.session_state.sending_progress = (current / total) * 100
        except Exception as e:
            print(f"Progress update error: {e}")

# ======================
# Thread Control Functions
# ======================
def initialize_thread_events():
    """Initialize thread events based on session state (only call from main thread)"""
    if threading.current_thread() == threading.main_thread():
        if st.session_state.scraping_active:
            scraping_active_flag.set()
        else:
            scraping_active_flag.clear()
        
        if st.session_state.scraping_paused:
            scraping_paused_flag.set()
        else:
            scraping_paused_flag.clear()
        
        if st.session_state.sending_active:
            sending_active_flag.set()
        else:
            sending_active_flag.clear()
        
        if st.session_state.sending_paused:
            sending_paused_flag.set()
        else:
            sending_paused_flag.clear()

def update_session_state_from_events():
    """Update session state from thread events for UI consistency (only call from main thread)"""
    if threading.current_thread() == threading.main_thread():
        st.session_state.scraping_active = scraping_active_flag.is_set()
        st.session_state.scraping_paused = scraping_paused_flag.is_set()
        st.session_state.sending_active = sending_active_flag.is_set()
        st.session_state.sending_paused = sending_paused_flag.is_set()

def start_scraping():
    """Start scraping process"""
    scraping_active_flag.set()
    scraping_paused_flag.clear()
    if threading.current_thread() == threading.main_thread():
        st.session_state.scraping_active = True
        st.session_state.scraping_paused = False
        st.session_state.scraping_stopped = False
    log("Scraping started", LogLevel.INFO, "SYSTEM")

def pause_scraping():
    """Pause scraping process"""
    scraping_paused_flag.set()
    if threading.current_thread() == threading.main_thread():
        st.session_state.scraping_paused = True
        st.session_state.scraping_stopped = False
    log("Scraping paused", LogLevel.INFO, "SYSTEM")

def resume_scraping():
    """Resume scraping process"""
    scraping_paused_flag.clear()
    if threading.current_thread() == threading.main_thread():
        st.session_state.scraping_paused = False
        st.session_state.scraping_stopped = False
    log("Scraping resumed", LogLevel.INFO, "SYSTEM")

def stop_scraping():
    """Stop scraping process"""
    scraping_active_flag.clear()
    scraping_paused_flag.clear()
    if threading.current_thread() == threading.main_thread():
        st.session_state.scraping_active = False
        st.session_state.scraping_paused = False
        st.session_state.scraping_stopped = True
    log("Scraping stopped", LogLevel.INFO, "SYSTEM")

def start_sending():
    """Start sending process"""
    sending_active_flag.set()
    sending_paused_flag.clear()
    if threading.current_thread() == threading.main_thread():
        st.session_state.sending_active = True
        st.session_state.sending_paused = False
        st.session_state.sending_stopped = False
    log("Sending started", LogLevel.INFO, "SYSTEM")

def pause_sending():
    """Pause sending process"""
    sending_paused_flag.set()
    if threading.current_thread() == threading.main_thread():
        st.session_state.sending_paused = True
        st.session_state.sending_stopped = False
    log("Sending paused", LogLevel.INFO, "SYSTEM")

def resume_sending():
    """Resume sending process"""
    sending_paused_flag.clear()
    if threading.current_thread() == threading.main_thread():
        st.session_state.sending_paused = False
        st.session_state.sending_stopped = False
    log("Sending resumed", LogLevel.INFO, "SYSTEM")

def stop_sending():
    """Stop sending process"""
    sending_active_flag.clear()
    sending_paused_flag.clear()
    if threading.current_thread() == threading.main_thread():
        st.session_state.sending_active = False
        st.session_state.sending_paused = False
        st.session_state.sending_stopped = True
    log("Sending stopped", LogLevel.INFO, "SYSTEM")

# ======================
# Utility Functions
# ======================
def clean(txt: Optional[str]) -> str:
    if not txt:
        return ""
    s = html.unescape(txt)
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    # fix doubled quotes
    s = s.replace('""', '"')
    return re.sub(r"\s+", " ", s).strip()


def parse_id_input(s: str) -> List[int]:
    """Parse ID input string into list of integers"""
    s = (s or "").strip()
    if not s:
        return []
    ids = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                a_i, b_i = int(a), int(b)
                if a_i <= b_i:
                    ids.update(range(a_i, b_i + 1))
            except:
                continue
        else:
            try:
                ids.add(int(part))
            except:
                continue
    return sorted(ids)

def qhash_for(mcq: dict) -> str:
    """Generate hash for MCQ to detect duplicates"""
    h = (mcq.get("question","") + "|" + "|".join(mcq.get("options",[]))).encode("utf-8")
    return hashlib.sha256(h).hexdigest()

def validate_chat_id(chat_id: str) -> str:
    """Validate and format chat ID"""
    chat_id = chat_id.strip()
    if not chat_id:
        return ""
    if chat_id.startswith('@'):
        return chat_id
    try:
        if chat_id.isdigit() and int(chat_id) > 0:
            return chat_id
        elif chat_id.startswith('-') and chat_id[1:].isdigit():
            return chat_id
        elif chat_id.isdigit() and int(chat_id) < 0:
            return chat_id
        else:
            return f"@{chat_id}"
    except:
        return f"@{chat_id}"

# ======================
# Parser / Scraper Functions
# ======================
def parse_block(block, q_no: int):
    """Parse a single MCQ block from HTML"""
    q_elem = block.select_one("dl > dt > strong")
    question = clean(q_elem.get_text(" ", strip=True)) if q_elem else ""
    question = re.sub(r"^प्रश्न\s*\d+[:\-]?\s*", "", question)
    exam_elem = block.select_one("dl > dt > span")
    exam = clean(exam_elem.get_text(" ", strip=True)) if exam_elem else ""
    options = [clean(li.get_text(" ", strip=True)) for li in block.select("dd > ul > li")]
    answer = ""
    explanation = ""
    ans_block = block.select_one("div.rg-c-content")
    if ans_block:
        ans_tag = ans_block.find("strong", string=lambda t: t and "उत्तर" in t)
        if ans_tag:
            raw = ans_tag.get_text(" ", strip=True)
            answer = re.sub(r"^उत्तर\s*[:\-]?\s*", "", raw).strip()
        exp_tag = ans_block.find("strong", string=lambda t: t and "व्याख्या" in t)
        if exp_tag:
            parts = []
            cur = exp_tag.next_sibling
            while cur:
                if hasattr(cur, "get_text"):
                    txt = clean(cur.get_text(" ", strip=True))
                else:
                    txt = clean(str(cur))
                if txt:
                    parts.append(txt)
                cur = cur.next_sibling
            explanation = " ".join(parts).strip()
    if not question or len(options) < 2:
        return None
    return {"q_no": q_no, "question": question, "exam": exam, "options": options, "answer": answer, "explanation": explanation}

class QNoCounter:
    """Thread-safe counter for question numbers"""
    def __init__(self, start: int = 1):
        self.lock = threading.Lock()
        self.value = start

    def next(self, count: int = 1) -> int:
        with self.lock:
            current = self.value
            self.value += count
            return current

def scrape_tid(tid: int, start_q: Optional[int] = None, sleep_sec: float = 0.6, max_pages: Optional[int] = None):
    """Scrape a single TID"""
    # Handle default start_q
    if start_q is None:
        start_q = 1
        
    url = f"{BASE_URL}?tid={tid}&start=0&sort=n"
    mcqs = []
    q_no = start_q
    page = 0
    
    while url and (max_pages is None or page < max_pages) and scraping_active_flag.is_set():
        # Check if paused
        while scraping_paused_flag.is_set() and scraping_active_flag.is_set():
            time.sleep(0.5)
        
        try:
            update_scraping_progress(
                current=page,
                total=100,  # Using a placeholder total since we don't know the total pages
                tid=tid,
                page=page + 1,
                status=f"Fetching TID {tid}, page {page + 1}"
            )
            
            r = requests.get(url, headers=USER_AGENT, timeout=25)
            if r.status_code != 200:
                log(f"TID {tid} page {page+1} status={r.status_code}", LogLevel.ERROR, "SCRAPER")
                break
                
            soup = BeautifulSoup(r.text, "html.parser")
            page += 1
            
            log(f"TID {tid} page {page} fetched, count={len(soup.find_all('dl'))}", LogLevel.INFO, "SCRAPER")
            
            for block in soup.find_all("dl"):
                parsed = parse_block(block, q_no)
                if parsed:
                    mcqs.append(parsed)
                    q_no += 1
            
            pagination_div = soup.find("p", style="text-align: center;")
            if pagination_div and "page no." in pagination_div.get_text().lower():
                log(f"TID {tid} pagination info: {pagination_div.get_text()}", LogLevel.DEBUG, "SCRAPER")
            
            nxt = soup.find("a", title="NEXT") or soup.find("a", string=re.compile("अगला|Next|→|›|>"))
            if not nxt:
                nxt = soup.find("a", href=lambda x: x and "start=" in x and "sort=n" in x)
            
            if nxt and nxt.get("href"):
                next_url = nxt["href"]
                url = urljoin(BASE_URL, next_url)
                log(f"TID {tid} Next page found: {url}", LogLevel.DEBUG, "SCRAPER")
                parsed_url = urlparse(url)
                query_params = parse_qs(parsed_url.query)
                start_val = int(query_params.get("start", [0])[0])
                log(f"TID {tid} Next start value: {start_val}", LogLevel.DEBUG, "SCRAPER")
            else:
                url = None
                log(f"TID {tid} No more pages", LogLevel.INFO, "SCRAPER")
                
            time.sleep(sleep_sec)
            
        except Exception as e:
            log(f"TID {tid} page {page+1} error: {e}", LogLevel.ERROR, "SCRAPER")
            break
            
    return mcqs, q_no

def scrape_tid_parallel_pages(
    tid: int, 
    start_q: Optional[int] = None, 
    sleep_sec: float = 0.6, 
    max_pages: Optional[int] = None
):
    """Scrape a single TID using parallel page fetching, Streamlit-safe."""

    if start_q is None:
        start_q = 1

    first_url = f"{BASE_URL}?tid={tid}&start=0&sort=n"
    r = requests.get(first_url, headers=USER_AGENT, timeout=25)
    if r.status_code != 200:
        log(f"TID {tid} first page failed: {r.status_code}", LogLevel.ERROR, "SCRAPER")
        return [], start_q

    soup = BeautifulSoup(r.text, "html.parser")

    # Detect total pages
    pagination_div = soup.find("p", style="text-align: center;")
    total_pages = 1
    if pagination_div and "page no." in pagination_div.get_text().lower():
        m = re.search(r"\((\d+)/(\d+)\)", pagination_div.get_text())
        if m:
            total_pages = int(m.group(2))
            if max_pages:
                total_pages = min(total_pages, max_pages)

    log(f"TID {tid} total_pages={total_pages}", LogLevel.INFO, "SCRAPER")

    # Build URLs
    page_urls = [f"{BASE_URL}?tid={tid}&start={i*10}&sort=n" for i in range(total_pages)]

    def fetch_page(url, page_no):
        """Worker thread: fetch + parse only (no Streamlit)."""
        try:
            if sleep_sec > 0:
                time.sleep(sleep_sec)
            r = requests.get(url, headers=USER_AGENT, timeout=25)
            if r.status_code != 200:
                return page_no, []
            soup = BeautifulSoup(r.text, "html.parser")
            mcqs = []
            for block in soup.find_all("dl"):
                parsed = parse_block(block, 0)
                if parsed:
                    mcqs.append(parsed)
            return page_no, mcqs
        except Exception:
            return page_no, []

    # Run workers
    page_results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(fetch_page, url, i+1): i+1 for i, url in enumerate(page_urls)}
        for future in concurrent.futures.as_completed(futures):
            if not scraping_active_flag.is_set():
                break
            page_no, mcqs = future.result()
            page_results[page_no] = mcqs
            
            # Log the progress
            log(f"TID {tid} page {page_no} fetched, count={len(mcqs)}", LogLevel.INFO, "SCRAPER")
            
            # Update progress in a thread-safe way
            update_scraping_progress(
                current=len(page_results),
                total=total_pages,
                tid=tid,
                page=page_no,
                status=f"TID {tid} page {page_no} fetched"
            )

    # Reassemble ordered results
    all_mcqs = []
    q_no = start_q
    for page_no in sorted(page_results.keys()):
        for mcq in page_results[page_no]:
            mcq["q_no"] = q_no
            q_no += 1
            all_mcqs.append(mcq)

    return all_mcqs, q_no

def scrape_tid_wrapper(tid, counter: QNoCounter, sleep_sec, max_pages, result_list, index, use_parallel_pages):
    """Wrapper for scraping a single TID in parallel"""
    try:
        if use_parallel_pages:
            mcqs, _ = scrape_tid_parallel_pages(tid, start_q=None, sleep_sec=sleep_sec, max_pages=max_pages)
        else:
            mcqs, _ = scrape_tid(tid, start_q=None, sleep_sec=sleep_sec, max_pages=max_pages)
        
        for m in mcqs:
            m["q_no"] = counter.next()
        result_list[index] = (tid, mcqs, None, None)
    except Exception as e:
        result_list[index] = (tid, [], None, str(e))

def scrape_tids_parallel(tids, sleep_sec=0.6, max_pages=None, max_workers=3, use_parallel_pages=False):
    """Scrape multiple TIDs in parallel"""
    log(f"Starting parallel scrape for {len(tids)} TIDs with {max_workers} workers", LogLevel.INFO, "SCRAPER")
    
    with db_connection() as con:
        cur = con.cursor()
        cur.execute("SELECT MAX(q_no) FROM mcqs")
        r = cur.fetchone()
        start_q = int(r[0]) + 1 if r and r[0] else 1
        
    counter = QNoCounter(start=start_q)
    results = [None] * len(tids)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(
                scrape_tid_wrapper,
                tid,
                counter,
                sleep_sec,
                max_pages,
                results,
                i,
                use_parallel_pages
            ): i for i, tid in enumerate(tids)
        }
        
        for i, future in enumerate(concurrent.futures.as_completed(future_to_index)):
            if not scraping_active_flag.is_set():
                for f in future_to_index:
                    f.cancel()
                break
                
            index = future_to_index[future]
            try:
                future.result()
                update_scraping_progress(current=index + 1, total=len(tids))
            except Exception as e:
                log(f"Thread for TID {tids[index]} generated an exception: {e}", LogLevel.ERROR, "SCRAPER")
                
    all_mcqs = []
    errors = []
    for tid, mcqs, _, error in results:
        if error:
            errors.append(f"TID {tid}: {error}")
        else:
            all_mcqs.extend(mcqs)
            
    next_q = counter.value
    return all_mcqs, next_q, errors

# ======================
# Database Functions
# ======================
def init_db(path=DB_PATH):
    """Initialize the database and create tables if they don't exist"""
    try:
        con = sqlite3.connect(path, check_same_thread=False)
        cur = con.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS mcqs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                qhash TEXT UNIQUE,
                q_no INTEGER,
                question TEXT,
                exam TEXT,
                options TEXT,
                answer TEXT,
                explanation TEXT,
                status TEXT,
                last_error TEXT,
                sent_at TEXT,
                created_at TEXT
            )
        """)
        con.commit()
        con.close()
        log(f"Database initialized successfully at {path}", LogLevel.INFO, "DATABASE")
        return True
    except Exception as e:
        log(f"Error initializing database: {e}", LogLevel.ERROR, "DATABASE")
        return False

def insert_mcq(con, mcq: dict, status="Scraped"):
    """Insert an MCQ into the database"""
    cur = con.cursor()
    qh = qhash_for(mcq)
    try:
        cur.execute("""
            INSERT INTO mcqs (qhash,q_no,question,exam,options,answer,explanation,status,created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (qh, mcq.get("q_no"), mcq.get("question"), mcq.get("exam"),
              json.dumps(mcq.get("options"), ensure_ascii=False), mcq.get("answer"),
              mcq.get("explanation"), status, datetime.now(timezone.utc).isoformat()))
        con.commit()
        return True, None
    except sqlite3.IntegrityError:
        return False, "duplicate"
    except Exception as e:
        return False, str(e)

def fetch_preview(con, limit=200, q_filter: Optional[str] = None):
    """Fetch MCQs from database for preview"""
    cur = con.cursor()
    if q_filter:
        cur.execute("SELECT id,q_no,question,exam,options,answer,explanation,status FROM mcqs WHERE question LIKE ? OR exam LIKE ? ORDER BY id DESC LIMIT ?", 
                   (f"%{q_filter}%", f"%{q_filter}%", limit))
    else:
        cur.execute("SELECT id,q_no,question,exam,options,answer,explanation,status FROM mcqs ORDER BY id DESC LIMIT ?", (limit,))
    return cur.fetchall()

def fetch_pending(con, limit=500):
    """Fetch pending MCQs from database"""
    cur = con.cursor()
    cur.execute("SELECT id,q_no,question,exam,options,answer,explanation FROM mcqs WHERE status IN ('Scraped','Queued','Failed') ORDER BY id LIMIT ?", (limit,))
    rows = cur.fetchall()
    return [{"id":r[0],"q_no":r[1],"question":r[2],"exam":r[3],"options":json.loads(r[4]),"answer":r[5],"explanation":r[6]} for r in rows]

def fetch_by_ids(con, ids: List[int]):
    """Fetch MCQs by IDs from database"""
    if not ids:
        return []
    cur = con.cursor()
    placeholders = ",".join("?" for _ in ids)
    cur.execute(f"SELECT id,q_no,question,exam,options,answer,explanation FROM mcqs WHERE id IN ({placeholders})", tuple(ids))
    rows = cur.fetchall()
    return [{"id":r[0],"q_no":r[1],"question":r[2],"exam":r[3],"options":json.loads(r[4]),"answer":r[5],"explanation":r[6]} for r in rows]

def update_status(con, row_id: int, status: str, last_error: Optional[str] = None):
    """Update the status of an MCQ in the database"""
    cur = con.cursor()
    sent_at = datetime.now(timezone.utc).isoformat() if status == "Sent" else None
    cur.execute("UPDATE mcqs SET status=?, last_error=?, sent_at=? WHERE id=?", (status, last_error, sent_at, row_id))
    con.commit()

# ======================
# Telegram Functions
# ======================
def tg_send_message(token: str, chat_id: str, text: str):
    """Send a message to Telegram"""
    url = TELEGRAM_API.format(token=token, method="sendMessage")
    payload = {"chat_id": chat_id, "text": text}
    try:
        r = requests.post(url, data=payload, timeout=30)
        return r.ok, r.text
    except Exception as e:
        return False, str(e)
        
def sanitize_html(text):
    # Escape everything
    safe = escape(text)
    # Optionally allow specific tags back in
    for tag in ALLOWED_TAGS:
        safe = re.sub(f"&lt;{tag}&gt;", f"<{tag}>", safe)
        safe = re.sub(f"&lt;/{tag}&gt;", f"</{tag}>", safe)
    return safe
ALLOWED_TAGS = {"b", "i", "u", "s", "a", "code", "pre", "tg-spoiler"}    
def tg_send_poll(token: str, chat_id: str, question: str, options: List[str], is_quiz: bool = False, correct_option_id: Optional[int] = None, explanation: Optional[str] = None):
    """Send a poll to Telegram"""
    url = TELEGRAM_API.format(token=token, method="sendPoll")
    data = {
        "chat_id": chat_id,
        "question": question,
        "options": json.dumps(options, ensure_ascii=False),
        "is_anonymous": "true",
        "allows_multiple_answers": "false",
        "type": "quiz" if is_quiz else "regular"
    }
    if is_quiz and correct_option_id is not None:
        data["correct_option_id"] = int(correct_option_id)
        if explanation:
            safe_expl = sanitize_html(explanation)
            data["explanation"] = safe_expl if len(safe_expl) <= TG_EXPL_LIMIT else safe_expl[:TG_EXPL_LIMIT]
            data["explanation_parse_mode"] = "HTML"
    for attempt in range(3):
        try:
            r = requests.post(url, data=data, timeout=35)
            if r.status_code == 200:
                return True, r.text
            else:
                return False, r.text
        except requests.exceptions.Timeout:
            if attempt == 2:
                return False, "Timeout after 3 attempts"
            time.sleep(2)
        except Exception as e:
            return False, str(e)

def normalize_text(s: str) -> str:
    """Remove labels and normalize spaces/lowercase for matching"""
    if not s:
        return ""
    # Remove leading labels in brackets like (अ), (a), (1)
    s = re.sub(r"^\(.*?\)\s*", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()

def detect_correct_option(mcq: dict) -> int:
    """Detect the correct option index for an MCQ"""
    ans = (mcq.get("answer") or "").strip()
    if not ans:
        return -1
    opts = mcq.get("options", []) or []

    a = normalize_text(ans)

    # Label mapping
    mapping = {
        "a": 0, "b": 1, "c": 2, "d": 3,
        "1": 0, "2": 1, "3": 2, "4": 3,
        "अ": 0, "ब": 1, "स": 2, "द": 3,
    }

    # If answer is just a label
    if a in mapping:
        idx = mapping[a]
        if idx < len(opts):
            return idx

    # Match normalized text against each option
    for i, o in enumerate(opts):
        o_norm = normalize_text(o)
        if a == o_norm or a in o_norm:
            return i

    return -1

# ======================
# Thread Functions
# ======================
def scraping_worker(tids, sleep_sec, max_pages, use_parallel_pages):
    """Worker thread for scraping"""
    try:
        log(f"Scraping worker started for TIDs: {tids}", LogLevel.INFO, "SCRAPER")
        
        with db_connection() as con:
            all_mcqs, next_q, errors = scrape_tids_parallel(
                tids=tids,
                sleep_sec=sleep_sec,
                max_pages=max_pages,
                max_workers=3,
                use_parallel_pages=use_parallel_pages
            )
            
            for mcq in all_mcqs:
                insert_mcq(con, mcq)
                
            if errors:
                for error in errors:
                    log(error, LogLevel.ERROR, "SCRAPER")
                    
        log(f"Scraping worker completed. Processed {len(all_mcqs)} MCQs", LogLevel.SUCCESS, "SCRAPER")
        
    except Exception as e:
        log(f"Scraping worker error: {e}", LogLevel.ERROR, "SCRAPER")
    finally:
        # Clear events when thread finishes
        scraping_active_flag.clear()
        scraping_paused_flag.clear()
        log("Scraping worker finished", LogLevel.INFO, "SCRAPER")

def sending_worker(token, chat_id, ids, send_rate, send_all=False):
    """Worker thread for sending MCQs as polls. Splits Q into message+poll if too long."""
    try:
        log(f"Sending worker started for {'all pending' if send_all else f'IDs: {ids}'}", LogLevel.INFO, "SENDER")

        with db_connection() as con:
            mcqs = fetch_pending(con) if send_all else fetch_by_ids(con, ids)
            log(f"Found {len(mcqs)} MCQs to send", LogLevel.INFO, "SENDER")

            for i, mcq in enumerate(mcqs):
                if not sending_active_flag.is_set():
                    break

                while sending_paused_flag.is_set() and sending_active_flag.is_set():
                    time.sleep(0.5)
                if not sending_active_flag.is_set():
                    break

                # Build question text
                q_no = mcq.get("q_no", "")
                exam = mcq.get("exam", "")
                base_q = mcq.get("question", "")
                question = f"Q. {q_no} {base_q}"
                if exam:
                    question += f" - {exam}"

                options = mcq["options"]
                correct_idx = detect_correct_option(mcq)
                explanation = mcq["explanation"]

                # Split case: question too long
                if len(question) > TG_Q_LIMIT:
                    msg_text = question  # Full text in a separate message
                    tg_send_message(token, chat_id, msg_text)
                    poll_q = f"Q. {q_no}"
                    if exam:
                        poll_q += f" - {exam}"
                else:
                    poll_q = question

                # Trim options
                options = [
                    o[:TG_OPT_LIMIT] if len(o) > TG_OPT_LIMIT else o
                    for o in options[:TG_OPT_COUNT]
                ]

                success, response = tg_send_poll(
                    token, chat_id, poll_q, options,
                    is_quiz=True,
                    correct_option_id=correct_idx,
                    explanation=explanation
                )

                if success:
                    update_status(con, mcq["id"], "Sent")
                    log(f"Sent MCQ ID {mcq['id']} successfully", LogLevel.INFO, "SENDER")
                else:
                    update_status(con, mcq["id"], "Failed", response)
                    log(f"Failed to send MCQ ID {mcq['id']}: {response}", LogLevel.ERROR, "SENDER")

                update_sending_progress(current=i + 1, total=len(mcqs), status=f"Sent {i+1}/{len(mcqs)}")

                if send_rate > 0:
                    time.sleep(send_rate)

        log("Sending worker completed", LogLevel.SUCCESS, "SENDER")

    except Exception as e:
        log(f"Sending worker error: {e}", LogLevel.ERROR, "SENDER")
    finally:
        sending_active_flag.clear()
        sending_paused_flag.clear()
        log("Sending worker finished", LogLevel.INFO, "SENDER")



# ======================
# UI Components
# ======================
def render_sidebar():
    """Render the sidebar with settings"""
    st.sidebar.header("Settings")
    
    # Table display options in a collapsible section
    with st.sidebar.expander("Table Columns", expanded=True):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.table_settings["show_id"] = st.sidebar.checkbox("ID", value=st.session_state.table_settings["show_id"])
            st.session_state.table_settings["show_q_no"] = st.sidebar.checkbox("Q.No", value=st.session_state.table_settings["show_q_no"])
            st.session_state.table_settings["show_exam"] = st.sidebar.checkbox("Exam", value=st.session_state.table_settings["show_exam"])
            st.session_state.table_settings["show_options"] = st.sidebar.checkbox("Options", value=st.session_state.table_settings["show_options"])
        with col2:
            st.session_state.table_settings["show_correct_option"] = st.sidebar.checkbox("Correct Option", value=st.session_state.table_settings["show_correct_option"])
            st.session_state.table_settings["show_explanation"] = st.sidebar.checkbox("Explanation", value=st.session_state.table_settings["show_explanation"])
            st.session_state.table_settings["show_status"] = st.sidebar.checkbox("Status", value=st.session_state.table_settings["show_status"])
    
    # Text display options in a collapsible section
    with st.sidebar.expander("Text Display", expanded=True):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            st.session_state.table_settings["show_full_question"] = st.sidebar.checkbox("Full Question", value=st.session_state.table_settings["show_full_question"])
            st.session_state.table_settings["show_full_explanation"] = st.sidebar.checkbox("Full Explanation", value=st.session_state.table_settings["show_full_explanation"])
        with col2:
            st.session_state.table_settings["show_full_options"] = st.sidebar.checkbox("Full Options", value=st.session_state.table_settings["show_full_options"])
    
    # Export options in a collapsible section
    with st.sidebar.expander("Export Options", expanded=True):
        col1, col2 = st.sidebar.columns(2)
        with col1:
            export_json = st.sidebar.button("Export JSON")
        with col2:
            export_txt = st.sidebar.button("Export TXT")
    
    # Preview filter in a collapsible section
    with st.sidebar.expander("Preview Filter", expanded=True):
        q_filter = st.sidebar.text_input("Question/Exam filter", value="")
        preview_limit = st.sidebar.slider("Preview rows", 10, 30000, 30000)
    
    return {
        "export_json": export_json,
        "export_txt": export_txt,
        "q_filter": q_filter,
        "preview_limit": preview_limit
    }

def render_control_panels():
    """Render the control panels for scraping and sending"""
    # Create two columns for scraper and sender
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Scraping Controls")
        
        # TID input with button in the same line
        with st.container():
            col1_1, col1_2 = st.columns([3, 1])
            with col1_1:
                tid_input = st.text_input("TIDs (comma/ranges)", value="1", key="tid_input")
            with col1_2:
                scrape_btn = st.button("Start", type="primary", key="scrape_btn")
        
        # Scraping options in a grouped form
        with st.container():
            st.markdown("**Scraping Options**")
            col1_1, col1_2 = st.columns(2)
            with col1_1:
                sleep_sec = st.number_input("Delay (s)", min_value=0.0, max_value=5.0, value=0.6, step=0.1, key="sleep_sec")
                use_parallel = st.checkbox("Parallel TIDs", value=True, key="use_parallel")
            with col1_2:
                use_parallel_pages = st.checkbox("Parallel Pages", value=True, key="use_parallel_pages")
                max_workers = st.slider("Workers", 1, 10, 3, disabled=not use_parallel, key="max_workers")
        
        # Action buttons in a grouped row
        with st.container():
            st.markdown("**Scraping Actions**")
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                # Toggle button for pause/resume
                if st.session_state.scraping_active and not st.session_state.scraping_paused:
                    pause_scrape_btn = st.button("Pause", key="pause_scrape_btn")
                else:
                    resume_scrape_btn = st.button("Resume", key="resume_scrape_btn", disabled=not st.session_state.scraping_active)
            with col1_2:
                stop_scrape_btn = st.button("Stop", key="stop_scrape_btn", disabled=not st.session_state.scraping_active)
            with col1_3:
                # Status display
                status_text = "Active" if st.session_state.scraping_active else "Inactive"
                if st.session_state.scraping_paused:
                    status_text = "Paused"
                st.markdown(f"**Status:** {status_text}")
    
    with col2:
        st.header("Sending Controls")
        
        # Telegram settings in a grouped form
        with st.container():
            st.markdown("**Telegram Settings**")
            default_token = TELEGRAM_TOKEN or ""
            default_chat = TELEGRAM_CHAT_ID or ""
            tg_token = st.text_input("Bot Token", value=default_token, type="password", key="tg_token")
            tg_chat = st.text_input("Chat ID", value=default_chat, help="Format: -1001234567890", key="tg_chat")
            send_rate = st.number_input("Delay (s)", min_value=0.1, max_value=10.0, value=1.0, step=0.1, key="send_rate")
        
        # Selected IDs with button in the same line
        with st.container():
            col2_1, col2_2 = st.columns([3, 1])
            with col2_1:
                selected_ids_text = st.text_input("Selected IDs", key="selected_ids_text")
            with col2_2:
                send_selected_btn = st.button("Send Selected", type="primary", key="send_selected_btn")
        
        # Action buttons in a grouped row
        with st.container():
            st.markdown("**Sending Actions**")
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                # Toggle button for pause/resume
                if st.session_state.sending_active and not st.session_state.sending_paused:
                    pause_send_btn = st.button("Pause", key="pause_send_btn")
                else:
                    resume_send_btn = st.button("Resume", key="resume_send_btn", disabled=not st.session_state.sending_active)
            with col2_2:
                stop_send_btn = st.button("Stop", key="stop_send_btn", disabled=not st.session_state.sending_active)
            with col2_3:
                send_all_btn = st.button("Send All", key="send_all_btn")
            
            # Status display
            status_text = "Active" if st.session_state.sending_active else "Inactive"
            if st.session_state.sending_paused:
                status_text = "Paused"
            st.markdown(f"**Status:** {status_text}")
    
    return {
        "scrape_btn": scrape_btn,
        "pause_scrape_btn": pause_scrape_btn if 'pause_scrape_btn' in locals() else None,
        "resume_scrape_btn": resume_scrape_btn if 'resume_scrape_btn' in locals() else None,
        "stop_scrape_btn": stop_scrape_btn,
        "send_selected_btn": send_selected_btn,
        "send_all_btn": send_all_btn,
        "pause_send_btn": pause_send_btn if 'pause_send_btn' in locals() else None,
        "resume_send_btn": resume_send_btn if 'resume_send_btn' in locals() else None,
        "stop_send_btn": stop_send_btn,
        "tid_input": tid_input,
        "sleep_sec": sleep_sec,
        "use_parallel": use_parallel,
        "use_parallel_pages": use_parallel_pages,
        "max_workers": max_workers,
        "tg_token": tg_token,
        "tg_chat": tg_chat,
        "send_rate": send_rate,
        "selected_ids_text": selected_ids_text
    }

def render_progress_section():
    """Render the progress section with dynamic progress bars and controls"""
    st.subheader("Progress")
    
    # Create columns for progress bars
    col1, col2 = st.columns(2)
    
    with col1:
        # Scraping progress
        with st.container():
            st.markdown("**Scraping Progress**")
            scraping_col1, scraping_col2 = st.columns([3, 1])
            with scraping_col1:
                st.markdown(f"Status: {st.session_state.scraping_status}")
            with scraping_col2:
                st.markdown(f"{int(st.session_state.scraping_progress)}%")
            
            scraping_progress_bar = st.progress(st.session_state.scraping_progress / 100)
            
            # Additional scraping details
            if st.session_state.current_tid:
                st.caption(f"TID: {st.session_state.current_tid}, Page: {st.session_state.current_tid_page}")
    
    with col2:
        # Sending progress
        with st.container():
            st.markdown("**Sending Progress**")
            sending_col1, sending_col2 = st.columns([3, 1])
            with sending_col1:
                st.markdown(f"Status: {st.session_state.sending_status}")
            with sending_col2:
                st.markdown(f"{int(st.session_state.sending_progress)}%")
            
            sending_progress_bar = st.progress(st.session_state.sending_progress / 100)
            
            # Additional sending details
            if st.session_state.total_to_send > 0:
                st.caption(f"Sent: {st.session_state.sent_count} of {st.session_state.total_to_send}")

def render_combined_table_and_logs(rows):
    """Render a single combined table with both preview and selection features, followed by logs"""
    # Create a single container for both table and logs
    container = st.container()
    
    with container:
        # Table section
        st.subheader("MCQ Data Table")
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(rows, columns=["ID", "Q.No", "Question", "Exam", "Options", "Answer", "Explanation", "Status"])
        
        # Get table settings
        settings = st.session_state.table_settings
        
        # Format columns based on user preferences
        if not settings["show_full_question"]:
            df["Question"] = df["Question"].str.slice(0, 160) + "..."
        
        if not settings["show_full_explanation"]:
            df["Explanation"] = df["Explanation"].str.slice(0, 160) + "..."
        
        if not settings["show_full_options"]:
            df["Options"] = df["Options"].apply(lambda x: str(x)[:160] + "..." if len(str(x)) > 160 else str(x))
        
        # Add search and filter options in a grouped row
        with st.container():
            st.markdown("**Table Filters**")
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                search_term = st.text_input("Search in table", key="table_search")
            with col2:
                status_filter = st.selectbox("Filter by status", ["All", "Scraped", "Sent", "Failed", "Queued"])
            with col3:
                download_csv = st.download_button(
                    label="Download CSV",
                    data=df.to_csv(index=False).encode('utf-8'),
                    file_name='mcq_preview.csv',
                    mime='text/csv'
                )
        
        # Apply filters
        filtered_df = df.copy()
        
        if search_term:
            filtered_df = filtered_df[
                filtered_df["Question"].str.contains(search_term, case=False) |
                filtered_df["Exam"].str.contains(search_term, case=False) |
                filtered_df["Explanation"].str.contains(search_term, case=False) |
                filtered_df["Options"].str.contains(search_term, case=False)
            ]
        
        if status_filter != "All":
            filtered_df = filtered_df[filtered_df["Status"] == status_filter]
        
        # Store filtered data in session state for action buttons
        st.session_state.filtered_data = filtered_df
        
        # Build column configuration based on user preferences
        column_config = {}
        
        if settings["show_id"]:
            column_config["ID"] = st.column_config.NumberColumn(format="%d")
        
        if settings["show_q_no"]:
            column_config["Q.No"] = st.column_config.NumberColumn(format="%d")
        
        if settings["show_exam"]:
            column_config["Exam"] = st.column_config.TextColumn(width="medium")
        
        if settings["show_options"]:
            column_config["Options"] = st.column_config.TextColumn(width="large")
        
        if settings["show_correct_option"]:
            column_config["Answer"] = st.column_config.TextColumn(width="medium")
        
        if settings["show_explanation"]:
            column_config["Explanation"] = st.column_config.TextColumn(width="large")
        
        if settings["show_status"]:
            column_config["Status"] = st.column_config.TextColumn(width="small")
        
        # Always show question
        column_config["Question"] = st.column_config.TextColumn(width="large")
        
        # Select columns to display based on user preferences
        columns_to_display = []
        if settings["show_id"]:
            columns_to_display.append("ID")
        if settings["show_q_no"]:
            columns_to_display.append("Q.No")
        columns_to_display.append("Question")
        if settings["show_exam"]:
            columns_to_display.append("Exam")
        if settings["show_options"]:
            columns_to_display.append("Options")
        if settings["show_correct_option"]:
            columns_to_display.append("Answer")
        if settings["show_explanation"]:
            columns_to_display.append("Explanation")
        if settings["show_status"]:
            columns_to_display.append("Status")
        
        # Display the combined table with both preview and selection features
        edited_df = st.data_editor(
            filtered_df[columns_to_display],
            width='stretch',
            height=400,
            hide_index=True,
            column_config=column_config,
            num_rows="dynamic",
            key="combined_table"
        )
        
        # Store selected rows in session state
        st.session_state.selected_rows = edited_df
        
        # Add action buttons for selected rows in a grouped row
        if not edited_df.empty:
            with st.container():
                st.markdown("**Selected Rows Actions**")
                col1, col2, col3 = st.columns(3)
                with col1:
                    selected_ids = edited_df["ID"].tolist()
                    if st.button("Send Selected Rows"):
                        st.session_state.selected_ids_text = ",".join(map(str, selected_ids))
                        st.rerun()
                with col2:
                    if st.button("Export Selected Rows"):
                        st.download_button(
                            label="Download Selected",
                            data=edited_df.to_csv(index=False).encode('utf-8'),
                            file_name='selected_mcq.csv',
                            mime='text/csv'
                        )
                with col3:
                    if st.button("Delete Selected Rows"):
                        # This would require implementing a delete function
                        st.warning("Delete functionality not implemented yet")
        
        # Add statistics in a grouped section
        with st.container():
            st.markdown("**Statistics**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total MCQs", len(df))
            with col2:
                st.metric("Scraped", len(df[df["Status"] == "Scraped"]))
            with col3:
                st.metric("Sent", len(df[df["Status"] == "Sent"]))
            with col4:
                st.metric("Failed", len(df[df["Status"] == "Failed"]))
        
        # Logs section
        st.subheader("Logs")
        
        # Log filter options in a grouped row
        with st.container():
            st.markdown("**Log Filters**")
            log_filter_col1, log_filter_col2, log_filter_col3 = st.columns([1, 1, 1])
            with log_filter_col1:
                log_level_filter = st.selectbox(
                    "Filter by level",
                    ["ALL", LogLevel.INFO, LogLevel.WARNING, LogLevel.ERROR, LogLevel.SUCCESS, LogLevel.DEBUG],
                    key="log_level_filter"
                )
            with log_filter_col2:
                log_source_filter = st.selectbox(
                    "Filter by source",
                    ["ALL", "SCRAPER", "SENDER", "SYSTEM", "DATABASE", "TELEGRAM"],
                    key="log_source_filter"
                )
            with log_filter_col3:
                auto_scroll = st.checkbox("Auto-scroll", value=True, key="auto_scroll")
                clear_log = st.button("Clear Log", key="clear_log")
        
        if clear_log:
            st.session_state.log = ""
            st.session_state.log_messages = []
            # Clear global log as well
            with log_lock:
                _global_log_messages.clear()
            log("Log cleared", LogLevel.INFO, "SYSTEM")
        
        # Filter log messages
        filtered_messages = st.session_state.log_messages
        if log_level_filter != "ALL":
            filtered_messages = [m for m in filtered_messages if m["level"] == log_level_filter]
        if log_source_filter != "ALL":
            filtered_messages = [m for m in filtered_messages if m["source"] == log_source_filter]
        
        # Display log messages
        log_html = ""
        for msg in reversed(filtered_messages):  # Newest first
            level_colors = {
                LogLevel.INFO: "blue",
                LogLevel.WARNING: "orange",
                LogLevel.ERROR: "red",
                LogLevel.SUCCESS: "green",
                LogLevel.DEBUG: "gray"
            }
            
            color = level_colors.get(msg["level"], "black")
            log_html += f"""
            <div style='margin-bottom: 5px; border-left: 3px solid {color}; padding-left: 10px;'>
                <span style='color: gray; font-size: 0.8em;'>{msg['timestamp']}</span>
                <span style='color: {color}; font-weight: bold;'>[{msg['level']}]</span>
                <span style='font-weight: bold;'>{msg['source']}:</span>
                {html.escape(msg['message'])}
            </div>
            """
        
        st.markdown(log_html, unsafe_allow_html=True)
        
        if auto_scroll:
            st.components.v1.html(
                """
                <script>
                window.scrollTo(0, document.body.scrollHeight);
                </script>
                """,
                height=0
            )

# ======================
# Main Application
# ======================
def main():
    # Initialize session state
    init_session_state()
    
    # Initialize thread events based on session state
    initialize_thread_events()
    
    # Initialize database if not already done
    if not st.session_state.db_initialized:
        if init_db():
            st.session_state.db_initialized = True
    
    # Set page config
    st.set_page_config(page_title="AI_MCQ_bot", layout="wide")
    st.title("AI_MCQ_bot — Scraper + Telegram Polls")
    
    # Render sidebar
    sidebar_data = render_sidebar()
    
    # Render control panels
    control_data = render_control_panels()
    
    # Handle control actions
    if control_data["scrape_btn"]:
        if not scraping_active_flag.is_set():
            tids = parse_id_input(control_data["tid_input"])
            if tids:
                start_scraping()
                st.session_state.scraping_thread = threading.Thread(
                    target=scraping_worker,
                    args=(
                        tids,
                        control_data["sleep_sec"],
                        None,  # max_pages
                        control_data["use_parallel_pages"]
                    ),
                    daemon=True
                )
                st.session_state.scraping_thread.start()
            else:
                st.error("Please enter valid TIDs")
    
    if control_data["pause_scrape_btn"]:
        pause_scraping()
    
    if control_data["resume_scrape_btn"]:
        resume_scraping()
    
    if control_data["stop_scrape_btn"]:
        stop_scraping()
    
    if control_data["send_selected_btn"]:
        if not sending_active_flag.is_set():
            ids = parse_id_input(control_data["selected_ids_text"])
            if ids and control_data["tg_token"] and control_data["tg_chat"]:
                start_sending()
                st.session_state.sending_thread = threading.Thread(
                    target=sending_worker,
                    args=(
                        control_data["tg_token"],
                        control_data["tg_chat"],
                        ids,
                        control_data["send_rate"],
                        False  # send_all
                    ),
                    daemon=True
                )
                st.session_state.sending_thread.start()
            else:
                st.error("Please enter valid IDs, Bot Token, and Chat ID")
    
    if control_data["send_all_btn"]:
        if not sending_active_flag.is_set():
            if control_data["tg_token"] and control_data["tg_chat"]:
                start_sending()
                st.session_state.sending_thread = threading.Thread(
                    target=sending_worker,
                    args=(
                        control_data["tg_token"],
                        control_data["tg_chat"],
                        [],  # ids
                        control_data["send_rate"],
                        True  # send_all
                    ),
                    daemon=True
                )
                st.session_state.sending_thread.start()
            else:
                st.error("Please enter valid Bot Token and Chat ID")
    
    if control_data["pause_send_btn"]:
        pause_sending()
    
    if control_data["resume_send_btn"]:
        resume_sending()
    
    if control_data["stop_send_btn"]:
        stop_sending()
    
    # Update session state from events for UI consistency
    update_session_state_from_events()
    
    # Render progress section
    render_progress_section()
    
    # Render combined table and logs
    with db_connection() as con:
        rows = fetch_preview(con, limit=sidebar_data["preview_limit"], q_filter=sidebar_data["q_filter"])
        render_combined_table_and_logs(rows)
    
    # Handle export buttons
    if sidebar_data["export_json"]:
        with db_connection() as con:
            rows = fetch_preview(con, limit=10000)
            df = pd.DataFrame(rows, columns=["ID", "Q.No", "Question", "Exam", "Options", "Answer", "Explanation", "Status"])
            st.download_button(
                label="Download JSON",
                data=df.to_json(orient='records', indent=2),
                file_name='mcq_data.json',
                mime='application/json'
            )
    
    if sidebar_data["export_txt"]:
        with db_connection() as con:
            rows = fetch_preview(con, limit=10000)
            txt_content = ""
            for row in rows:
                txt_content += f"ID: {row[0]}\n"
                txt_content += f"Q.No: {row[1]}\n"
                txt_content += f"Question: {row[2]}\n"
                txt_content += f"Exam: {row[3]}\n"
                txt_content += f"Options: {row[4]}\n"
                txt_content += f"Answer: {row[5]}\n"
                txt_content += f"Explanation: {row[6]}\n"
                txt_content += f"Status: {row[7]}\n"
                txt_content += "-" * 50 + "\n"
            st.download_button(
                label="Download TXT",
                data=txt_content,
                file_name='mcq_data.txt',
                mime='text/plain'
            )

if __name__ == "__main__":
    main()