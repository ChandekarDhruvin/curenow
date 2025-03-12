import logging
from psycopg2.extras import DictCursor
from Database.db_connection import get_db

def create_messages_table():
    """Create messages table if it doesn't exist"""
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS messages (
                        id SERIAL PRIMARY KEY,
                        session_id TEXT NOT NULL,
                        user_message TEXT NOT NULL,
                        bot_response TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logging.info("Messages table created successfully.")
        except Exception as e:
            logging.error(f"Error creating table: {e}")

def save_chat(session_id, user_message, bot_response):
    """Save user and bot messages into database"""
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("""
                    INSERT INTO messages (session_id, user_message, bot_response)
                    VALUES (%s, %s, %s)
                """, (session_id, user_message, bot_response))
                logging.info("Chat saved successfully.")
        except Exception as e:
            logging.error(f"Error saving chat: {e}")

def get_chat_history(session_id):
    """Retrieve chat history for a given session ID"""
    db = get_db()
    if db:
        try:
            with db.cursor(cursor_factory=DictCursor) as cur:
                cur.execute("""
                    SELECT user_message, bot_response, timestamp 
                    FROM messages
                    WHERE session_id = %s 
                    ORDER BY timestamp ASC
                """, (session_id,))
                return cur.fetchall() or []
        except Exception as e:
            logging.error(f"Error retrieving chat history: {e}")
    return []

def clear_all_chats():
    """Clear all chat history"""
    db = get_db()
    if db:
        try:
            with db.cursor() as cur:
                cur.execute("TRUNCATE TABLE messages RESTART IDENTITY;")
                logging.info("All chats cleared successfully.")
                return True
        except Exception as e:
            logging.error(f"Error clearing chat history: {e}")
            return False
    return False

def create_appointments_table():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS appointments (
            id SERIAL PRIMARY KEY,
            doctor_id INT,
            patient_name VARCHAR(100),
            age INT,
            gender VARCHAR(10),
            symptoms TEXT,
            appointment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

# Save appointment details with new fields
def save_appointment(patient_name, age, gender, symptoms, doctor_id, appointment_date):
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO appointments (patient_name, age, gender, symptoms, doctor_id, appointment_date)
    VALUES (%s, %s, %s, %s, %s, %s)
    """, (patient_name, age, gender, symptoms, doctor_id, appointment_date))
    conn.commit()
    conn.close()

def get_doctors():
    conn = get_db()
    cursor = conn.cursor()
    cursor.execute("SELECT id, name, specialization FROM doctors")
    doctors = cursor.fetchall()
    conn.close()
    return doctors