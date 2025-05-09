import sqlite3
from sqlite3 import Error
from werkzeug.security import generate_password_hash, check_password_hash
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import json
import os
from flask import flash
class User:
    def __init__(self, db_path: str = 'keystroke_auth.db'):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_path = os.path.join(root_dir, 'keystroke_auth.db')
        self.db_path = db_path
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        print('debug: creating users tables')
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            print('debug: creating users table')
            # Users table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                failed_attempts INTEGER DEFAULT 0,
                locked_until TIMESTAMP NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP NULL
            )
            ''')
            print('debug: created users table')
            conn.commit()
        except Error as e:
            raise Exception(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()

    def create_user(self, username: str, password: str) -> int:
        print("debug: inside create_user")
        """Create a new user with hashed password"""
        password_hash = generate_password_hash(password)
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'INSERT INTO users (username, password_hash) VALUES (?, ?)',
                (username, password_hash)
            )
            user_id = cursor.lastrowid
            conn.commit()
            return user_id
        except sqlite3.IntegrityError:
            flash('Username already exists', 'error')
            #raise ValueError("Username already exists")
            return None
        except Error as e:
            flash('Username error',e, 'error')
            #raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def authenticate_user(self, username: str, password: str) -> Tuple[bool, Optional[int]]:
        """Authenticate user and return (success, user_id) tuple"""
        print("debug: inside authenticate_user")
        if self.is_account_locked(username):
            return False, None
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT id, password_hash FROM users WHERE username = ?',
                (username,))
            user = cursor.fetchone()
            if user is None:
                return False, None
            elif user and check_password_hash(user[1], password):
                # Reset failed attempts on successful login
                cursor.execute(
                    'UPDATE users SET failed_attempts = 0, locked_until = NULL, last_login = ? WHERE id = ?',
                    (datetime.now(), user[0]))
                conn.commit()
                return True, user[0]
            else:
                # Increment failed attempts
                self._increment_failed_attempts(username)
                return False, None
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def is_account_locked(self, username: str) -> bool:
        """Check if account is locked due to too many failed attempts"""
        try:
            print("debug: inside is_account_locked")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'SELECT locked_until FROM users WHERE username = ?',
                (username,))
            result = cursor.fetchone()
            
            if result and result[0]:
                locked_until = datetime.strptime(result[0], '%Y-%m-%d %H:%M:%S')
                return locked_until > datetime.now()
            return False
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def _increment_failed_attempts(self, username: str):
        """Increment failed login attempts and lock account if threshold reached"""
        try:
            print("debug: inside _increment_failed_attempts")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                'UPDATE users SET failed_attempts = failed_attempts + 1 WHERE username = ?',
                (username,))
            
            # Check if we need to lock the account
            cursor.execute(
                'SELECT failed_attempts FROM users WHERE username = ?',
                (username,))
            attempts = cursor.fetchone()[0]
            
            if attempts >= 5:  # Lock after 5 failed attempts
                lockout_time = datetime.now() + timedelta(minutes=5)
                cursor.execute(
                    'UPDATE users SET locked_until = ? WHERE username = ?',
                    (lockout_time.strftime('%Y-%m-%d %H:%M:%S'), username))
            
            conn.commit()
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    