import asyncio
import logging
import os
import re
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
import google.generativeai as genai
from collections import deque
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

# Constants
MAX_MESSAGE_LENGTH = 4096
MESSAGE_STORE_SIZE = 50

# Utility functions
def escape_markdown(text):
    """Escape special characters for Markdown."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return re.sub(r'([{}])'.format(re.escape(escape_chars)), r'\\\1', text)

def split_text(text, max_length=MAX_MESSAGE_LENGTH):
    """Split long text into chunks within Telegram message limits."""
    if len(text) <= max_length:
        return [text]
    
    chunks = []
    current_chunk = ""
    paragraphs = text.split('\n\n')
    
    for paragraph in paragraphs:
        if len(current_chunk) + len(paragraph) + 2 > max_length:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                # If a single paragraph is too long, split by sentences
                sentences = paragraph.split('. ')
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 > max_length:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            # If a single sentence is too long, split by words
                            words = sentence.split(' ')
                            for word in words:
                                if len(current_chunk) + len(word) + 1 > max_length:
                                    chunks.append(current_chunk)
                                    current_chunk = word
                                else:
                                    current_chunk += (' ' + word) if current_chunk else word
                    else:
                        current_chunk += ('. ' + sentence) if current_chunk else sentence
        else:
            current_chunk += ('\n\n' + paragraph) if current_chunk else paragraph
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

async def send_long_message(chat_id, text, context, header="🤖 AI Explanation:"):
    """Send a long message by splitting it into multiple parts."""
    chunks = split_text(text)
    
    for i, chunk in enumerate(chunks):
        if i == 0:
            message_text = f"{header}\n\n{chunk}"
        else:
            message_text = f"{header} (Part {i+1}/{len(chunks)})\n\n{chunk}"
        
        await context.bot.send_message(chat_id=chat_id, text=message_text)

# Message Store
class MessageStore:
    def __init__(self, max_size=MESSAGE_STORE_SIZE):
        self.messages = deque(maxlen=max_size)
        self.polls = deque(maxlen=max_size)
    
    def add_message(self, text, message_id=None):
        self.messages.append({"text": text, "id": message_id})
    
    def add_poll(self, poll, message_id=None):
        self.polls.append({"poll": poll, "id": message_id})
    
    def get_latest_mcq(self):
        if self.polls:
            return self.polls[-1]
        
        for message in reversed(self.messages):
            if self._is_mcq(message["text"]):
                return message
        return None
    
    def get_mcq_by_id(self, message_id):
        for poll_data in self.polls:
            if poll_data["id"] == message_id:
                return poll_data
        
        for message_data in self.messages:
            if message_data["id"] == message_id:
                return message_data
        
        return None
    
    def get_all_mcqs(self):
        mcqs = []
        
        for poll_data in self.polls:
            mcqs.append({
                "type": "poll",
                "data": poll_data,
                "id": poll_data["id"]
            })
        
        for message_data in self.messages:
            if self._is_mcq(message_data["text"]):
                mcqs.append({
                    "type": "text",
                    "data": message_data,
                    "id": message_data["id"]
                })
        
        return mcqs
    
    def _is_mcq(self, text):
        lines = text.split('\n')
        if len(lines) < 3:
            return False
            
        option_count = 0
        for line in lines[1:]:
            line = line.strip()
            if line and len(line) > 1 and line[0] in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd'] and line[1] in [')', '.', ':', '-']:
                option_count += 1
                
        return option_count >= 2

# Gemini Explanation Generator
class GeminiExplainer:
    def __init__(self):
        self.model = gemini_model
    
    async def generate_explanation(self, question, options, correct_answer=None, explanation_type="detailed", language="hi"):
        try:
            options_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
            
            lang_instruction = "Please provide the explanation in English language." if language == "en" else "Please provide the explanation in Hindi language."
            
            if correct_answer:
                prompt = f"""
                Question: {question}
                
                Options:
                {options_text}
                
                Correct Answer: {correct_answer}
                
                {lang_instruction}
                
                Please provide a {'concise' if explanation_type == 'short' else 'comprehensive'} explanation.
                """
            else:
                prompt = f"""
                Question: {question}
                
                Options:
                {options_text}
                
                {lang_instruction}
                
                Please provide a {'brief analysis' if explanation_type == 'short' else 'detailed analysis'} of this poll.
                """
            
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Sorry, I couldn't generate an explanation for this question."

# Global Variables
explainer = GeminiExplainer()
message_store = MessageStore()
pending_explanations = {}

# Telegram Bot Handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm your MCQ explainer bot.\n\n"
        "Commands:\n"
        "• /gemini [short|detailed] - Explain a MCQ\n"
        "• /explain - Show recent polls/MCQs\n"
        "• /help - Show help\n\n"
        "I support both English and Hindi explanations!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    help_text = """
    Available commands:
    /start - Welcome message
    /help - Show this help
    /gemini [short|detailed] - Explain a MCQ
    /explain - Show recent polls/MCQs

    Usage:
    • Reply to a MCQ with /gemini short for a brief explanation
    • Reply to a MCQ with /gemini detailed for a comprehensive explanation
    • Use /gemini without replying to explain the latest MCQ
    • Use /explain to see a list of recent polls/MCQs

    Language Options:
    • After using /gemini, select Hindi or English
    """
    await update.message.reply_text(help_text)

async def explain_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    mcqs = message_store.get_all_mcqs()
    
    if not mcqs:
        await update.message.reply_text("No recent polls or MCQs found.")
        return
    
    keyboard = []
    for i, mcq in enumerate(mcqs[:10]):
        if mcq["type"] == "poll":
            question = mcq["data"]["poll"].question
            btn_text = f"📊 Poll: {question[:30]}{'...' if len(question) > 30 else ''}"
        else:
            lines = mcq["data"]["text"].split('\n')
            question = lines[0] if lines else "Unknown question"
            btn_text = f"📝 MCQ: {question[:30]}{'...' if len(question) > 30 else ''}"
        
        keyboard.append([InlineKeyboardButton(btn_text, callback_data=f"exp_{mcq['id']}")])
    
    keyboard.append([InlineKeyboardButton("❌ Cancel", callback_data="exp_cancel")])
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.message.reply_text("Select a poll or MCQ to explain:", reply_markup=reply_markup)

async def gemini_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    args = context.args
    explanation_type = "detailed"
    
    if args:
        if args[0].lower() in ["short", "s"]:
            explanation_type = "short"
        elif args[0].lower() in ["detailed", "d", "long", "l"]:
            explanation_type = "detailed"
        else:
            await update.message.reply_text("Invalid type. Use 'short' or 'detailed'.")
            return
    
    mcq_data = None
    
    if update.message.reply_to_message:
        replied_message = update.message.reply_to_message
        message_id = replied_message.message_id
        mcq_data = message_store.get_mcq_by_id(message_id)
        
        if not mcq_data:
            if replied_message.poll:
                mcq_data = {"poll": replied_message.poll, "id": message_id}
            elif replied_message.text and message_store._is_mcq(replied_message.text):
                mcq_data = {"text": replied_message.text, "id": message_id}
    
    if not mcq_data:
        mcq_data = message_store.get_latest_mcq()
    
    if not mcq_data:
        await update.message.reply_text("No MCQ found. Please send or reply to a MCQ.")
        return
    
    keyboard = [
        [
            InlineKeyboardButton("🇬🇧 English", callback_data=f"lang_en_{explanation_type}"),
            InlineKeyboardButton("🇮🇳 हिंदी", callback_data=f"lang_hi_{explanation_type}")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    lang_message = await update.message.reply_text("Select language:", reply_markup=reply_markup)
    
    pending_explanations[lang_message.message_id] = {
        "mcq_data": mcq_data,
        "explanation_type": explanation_type,
        "user_id": update.effective_user.id,
        "chat_id": update.effective_chat.id
    }

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    callback_data = query.data
    
    if callback_data == "exp_cancel":
        await query.edit_message_text("Cancelled.")
        return
    
    if callback_data.startswith("exp_"):
        message_id = int(callback_data.split("_")[1])
        mcq_data = message_store.get_mcq_by_id(message_id)
        
        if not mcq_data:
            await query.edit_message_text("MCQ not found.")
            return
        
        keyboard = [
            [
                InlineKeyboardButton("🇬🇧 English", callback_data=f"gen_en_detailed"),
                InlineKeyboardButton("🇮🇳 हिंदी", callback_data=f"gen_hi_detailed")
            ],
            [
                InlineKeyboardButton("Short 🇬🇧", callback_data=f"gen_en_short"),
                InlineKeyboardButton("Short 🇮🇳", callback_data=f"gen_hi_short")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await query.edit_message_text("Select explanation type:", reply_markup=reply_markup)
        
        pending_explanations[query.message.message_id] = {
            "mcq_data": mcq_data,
            "user_id": query.from_user.id,
            "chat_id": query.message.chat_id
        }
        return
    
    if callback_data.startswith("lang_") or callback_data.startswith("gen_"):
        parts = callback_data.split("_")
        language = parts[1]
        explanation_type = parts[2] if len(parts) > 2 else "detailed"
        
        message_id = query.message.message_id
        if message_id not in pending_explanations:
            await query.edit_message_text("Request expired. Please try again.")
            return
        
        context_data = pending_explanations[message_id]
        mcq_data = context_data["mcq_data"]
        
        await query.edit_message_text(f"🔄 Generating {explanation_type} explanation in {'English' if language == 'en' else 'Hindi'}...")
        
        question, options, correct_answer, is_quiz = extract_mcq_data(mcq_data)
        
        explanation = await explainer.generate_explanation(
            question, options, correct_answer, explanation_type, language
        )
        
        header = f"💡 {'Short' if explanation_type == 'short' else 'Detailed'} Explanation ({'English' if language == 'en' else 'हिंदी'}):\n\n"
        await send_long_message(query.message.chat_id, explanation, context, header)
        
        if message_id in pending_explanations:
            del pending_explanations[message_id]

def extract_mcq_data(mcq_data):
    question = ""
    options = []
    correct_answer = None
    is_quiz = False
    
    if "poll" in mcq_data:
        poll = mcq_data["poll"]
        question = poll.question
        options = [option.text for option in poll.options]
        is_quiz = (getattr(poll, 'type', 'regular') == 'quiz')
        
        if is_quiz and getattr(poll, 'correct_option_id', None) is not None:
            correct_index = poll.correct_option_id
            if correct_index < len(options):
                correct_answer = options[correct_index]
    else:
        text = mcq_data["text"]
        lines = text.split('\n')
        question = lines[0]
        
        for line in lines[1:]:
            line = line.strip()
            if line and len(line) > 1 and line[0] in ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd'] and line[1] in [')', '.', ':', '-']:
                option_text = line[1:].strip()
                if option_text.startswith((')', '.', ':', '-')):
                    option_text = option_text[1:].strip()
                options.append(option_text)
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in ["answer:", "correct:", "ans:"]):
                answer_part = line.split(":", 1)[1].strip()
                if len(answer_part) == 1 and answer_part.upper() in ['A', 'B', 'C', 'D']:
                    correct_answer = options[ord(answer_part.upper()) - ord('A')]
                else:
                    correct_answer = answer_part
                break
    
    return question, options, correct_answer, is_quiz

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.text:
        message_store.add_message(update.message.text, update.message.message_id)

async def handle_poll(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message and update.message.poll:
        poll = update.message.poll
        message_store.add_poll(poll, update.message.message_id)
        
        keyboard = [[InlineKeyboardButton("🔍 Explain", callback_data=f"exp_{update.message.message_id}")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        is_quiz = (getattr(poll, 'type', 'regular') == 'quiz')
        if is_quiz and getattr(poll, 'correct_option_id', None) is not None:
            correct_answer = poll.options[poll.correct_option_id].text
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Quiz: {poll.question}\nCorrect: {correct_answer}",
                reply_markup=reply_markup
            )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Poll: {poll.question}",
                reply_markup=reply_markup
            )

# Main Application
def main():
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("gemini", gemini_command))
    application.add_handler(CommandHandler("explain", explain_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.POLL, handle_poll))

    application.run_polling()

if __name__ == "__main__":
    main()