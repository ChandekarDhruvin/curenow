import os
import psycopg2
import logging
from flask import g

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db():
    """Get a database connection, reconnect if lost"""
    if 'db' not in g:
        try:
            g.db = psycopg2.connect(DATABASE_URL)
            g.db.autocommit = True  # Ensure auto-commit mode
        except Exception as e:
            logging.error(f"Database connection error: {e}")
            g.db = None
    return g.db

def close_db(error=None):
    """Close database connection at the end of request"""
    db = g.pop('db', None)
    if db is not None:
        db.close()


