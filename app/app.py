import os
from datetime import datetime, timedelta
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_from_directory
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
from sqlite3 import Error
import json
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
import base64
import uuid

from werkzeug.utils import secure_filename



from config import get_config
from utils.plot_generator import PlotGenerator
from models.user import *
from models.keystroke import *
from models.mouse import *

app = Flask(__name__)
config = get_config(os.getenv('FLASK_ENV'))
config.init_app(app)
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)
app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'secured_files')
app.config['MAX_LOGIN_ATTEMPTS'] = 5
app.config['LOCKOUT_TIME'] = 300  # 5 minutes in seconds

DATABASE='keystroke_auth.db'




# Database initialization
def init_db():
    conn = None
    try:
        
        conn = sqlite3.connect(DATABASE)
        cursor = conn.cursor()
        #print('debug: creating db ')

        # Secured files table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS secured_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            filepath TEXT NOT NULL,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        #print("debug: created Secured files table")

        # Access logs table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS access_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            action TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
        ''')
        #print("debug: created Access logs table")

        conn.commit()
    except Error as e:
        print(f"Database error: {e}")
    finally:
        if conn:
            conn.close()

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def log_access(user_id, action):
    print("debug: inside log_access")
    try:
        conn = get_db_connection()
        conn.execute('INSERT INTO access_logs (user_id, action) VALUES (?, ?)', 
                    (user_id, action))
        conn.commit()
    except Error as e:
        print(f"Error logging access: {e}")
    finally:
        conn.close()

def reset_failed_attempts(username):
    conn = get_db_connection()
    conn.execute('UPDATE users SET failed_attempts = 0, locked_until = NULL WHERE username = ?', (username,))
    conn.commit()
    conn.close()


# Routes
@app.route('/')
def index():
    print("debug:inside index.html")
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        print("debug: inside login.html")
        username = request.form['username']
        password = request.form['password']
        
        authenticated,user_id=user_manager.authenticate_user(username=username,password=password)
        if authenticated and user_id:
            session.permanent = True
            session['user_id'] = user_id
            session['username'] = username
            log_access(user_id, 'login')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))

            
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        print("debug:inside signup.html")
        username = request.form['username']
        password = request.form['password']
        confirm_password = request.form['confirm-password']
        
        

        if password != confirm_password:
            flash('Passwords do not match', 'error')
            return redirect(url_for('signup'))
        
        user = user_manager.create_user(username=username,password=password)
        if not user:
            return render_template('signup.html')
        #print('debug: user created , user: ', user)
        session['user_id'] = user
        session['username'] = username
        #print('debug:',session)
        return redirect(url_for('signupdata'))
    
    return render_template('signup.html')

@app.route('/signupdata', methods=['GET', 'POST'])
def signupdata():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        try:
            print("debug: insde signupdata.html")
            data = request.get_json()
            #print("debug: data: ",data)   
            keystroke_manager.record_keystroke_event(user_id=session['user_id'],keystroke_data=data.get('keystroke_data', []))
            mouse_manager.record_mouse_event(user_id=session['user_id'],mouse_data= data.get('mouse_data', []))            
            return jsonify({'success': True, 'redirect': url_for('dashboard')})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    return render_template('signupdata.html')

@app.route('/dashboard')
def dashboard():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    print('debug: inside dashboard.html')
    user_id = session['user_id']

    # Get keystroke data
    keystroke_events_data=keystroke_manager.get_keystroke_events(user_id=user_id)
    keystroke_features_data = keystroke_manager.get_keystroke_features(user_id=user_id)
    # Get total count
    keystroke_events_data_count = len(keystroke_events_data)
    keystroke_features_data_count=len(keystroke_features_data) 
    #print('debug: key_count: ',keystroke_events_data_count,keystroke_features_data_count)

    #get mouse data
    mouse_events_data=mouse_manager.get_mouse_events(user_id=user_id)
    mouse_features_data=mouse_manager.get_mouse_features(user_id=user_id)
    #get count
    mouse_events_data_count=len(mouse_events_data)
    mouse_features_data_count=len(mouse_features_data)

    profile =keystroke_manager.generate_typing_profile(user_id=user_id)
    #print("debug: profile: ",profile)
    # Get typing speed
    print(f"Typing speed: {profile['statistics']['typing_speed_wpm_pp']:.1f} WPM-pp")
    print(f"Typing speed: {profile['statistics']['typing_speed_wpm_rr']:.1f} WPM-rr")

    # See most common key transitions
    print("Common patterns:", profile['statistics']['common_patterns'][:3])
    
    
    conn = get_db_connection()
    # Get secured files
    secured_files = conn.execute('''
        SELECT filename 
        FROM secured_files 
        WHERE user_id = ?
    ''', (user_id,)).fetchall()
    secured_files = [dict(file) for file in secured_files]
    #print("debug: fetch secured_file from db :",secured_files)

    # Get access logs
    access_logs = conn.execute('''
        SELECT action, timestamp 
        FROM access_logs 
        WHERE user_id = ?
        ORDER BY timestamp DESC
        LIMIT 10
    ''', (user_id,)).fetchall()
    #print("debug: fetch access_log from db :",dict(access_logs[0]))

    # Get total users (for admin view)
    total_users = conn.execute('SELECT COUNT(*) FROM users').fetchone()[0]


    #print("debug: total user:", total_users)
    conn.close()

    plotter = PlotGenerator()
    # Generate keystroke plot
    keystroke_plot = None
    keystroke_heatmap_plot = None  

    #print("debug: keystroke_features_data: ",keystroke_features_data)
    if keystroke_features_data:
        dwell_times = [k['dwell_time'] for k in keystroke_features_data if k['dwell_time'] is not None]
        flight_times = [k['flight_time'] for k in keystroke_features_data if k['flight_time'] is not None]

        
        if dwell_times and flight_times:  # Only generate if we have data
            keystroke_plot = plotter.generate_keystroke_timing_plot(
                dwell_times=dwell_times,
                flight_times=flight_times
            )
        
        # Only generate heatmap if we have keystroke data
        keystroke_heatmap_plot = plotter.generate_keystroke_heatmap(processed_data=keystroke_features_data)

    return render_template('dashboard.html', 
                         keystroke_events_data=keystroke_events_data,
                         keystroke_events_data_count=keystroke_events_data_count,
                         keystroke_features_data=keystroke_features_data,
                         keystroke_features_data_count=keystroke_features_data_count,
                         
                         mouse_events_data=mouse_events_data,
                         mouse_events_data_count=mouse_events_data_count,
                         mouse_features_data=mouse_features_data,
                         mouse_features_data_count=mouse_features_data_count,
                         
                         secured_files=secured_files,
                         access_logs=access_logs,
                         total_users=total_users,
                         keystroke_plot=keystroke_plot,
                         keystroke_heatmap_plot=keystroke_heatmap_plot
                         )

@app.route('/plot/<path:filename>')
def plot(filename):
    return send_from_directory('static/plot', filename)

# Add this custom filter for datetime formatting
@app.template_filter('datetimeformat')
def datetimeformat(value, format='%Y-%m-%d %H:%M:%S'):
    if isinstance(value, str):
        try:
            value = datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            return value
    return value.strftime(format)

@app.route('/logout')
def logout():
    if 'user_id' in session:
        print('debug:loging out')
        log_access(session['user_id'], 'logout')
        session.clear()
    return redirect(url_for('index'))

@app.route('/manage_files', methods=['GET', 'POST'])
def manage_files():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    print("debug: inside manage_files")
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'add' and 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                try:
                    # Ensure upload folder exists
                    upload_folder = app.config['UPLOAD_FOLDER']
                    os.makedirs(upload_folder, exist_ok=True)
                    print(f"debug: Upload folder: {upload_folder}")
                    
                    # Secure filename
                    original_filename = secure_filename(file.filename)
                    unique_filename = f"{uuid.uuid4().hex}_{original_filename}"
                    filepath = os.path.join(upload_folder, unique_filename)
                    
                    print(f"debug: Saving file to: {filepath}")
                    file.save(filepath)
                    
                    # Verify file was saved
                    if not os.path.exists(filepath):
                        flash('File upload failed - file not saved', 'error')
                    else:
                        conn = get_db_connection()
                        conn.execute('''
                            INSERT INTO secured_files (user_id, filename, filepath)
                            VALUES (?, ?, ?)
                        ''', (user_id, original_filename, filepath))
                        conn.commit()
                        conn.close()
                        flash('File uploaded successfully', 'success')
                except Exception as e:
                    print(f"debug: Upload error: {str(e)}")
                    flash(f'File upload failed: {str(e)}', 'error')
        
        elif action == 'remove':
            file_id = request.form.get('file_id')
            if file_id:
                conn = get_db_connection()
                file = conn.execute('SELECT filepath FROM secured_files WHERE id = ? AND user_id = ?', 
                                    (file_id, user_id)).fetchone()
                if file:
                    try:
                        if os.path.exists(file['filepath']):
                            os.remove(file['filepath'])
                        conn.execute('DELETE FROM secured_files WHERE id = ?', (file_id,))
                        conn.commit()
                        flash('File removed successfully', 'success')
                    except Exception as e:
                        flash(f'Error removing file: {str(e)}', 'error')
                conn.close()
        
        return redirect(url_for('manage_files'))
    
    conn = get_db_connection()
    files = conn.execute('SELECT id, filename FROM secured_files WHERE user_id = ?', (user_id,)).fetchall()
    conn.close()
    
    # Convert Row objects to dictionaries
    files = [dict(file) for file in files]
    print('debug: files: ', files)
    return render_template('manage_files.html', secured_files=files)

@app.route('/open_file/<int:file_id>')
def open_file(file_id):
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    print("debug: inside open_file")
    conn = get_db_connection()
    file = conn.execute('''
        SELECT id, filename, filepath 
        FROM secured_files 
        WHERE id = ? AND user_id = ?
    ''', (file_id, session['user_id'])).fetchone()
    conn.close()
    
    if not file:
        flash('File not found or access denied', 'error')
        return redirect(url_for('manage_files'))
    
    file = dict(file)  # Convert to dictionary
    log_access(session['user_id'], f"accessed file: {file['filename']}")
    
    # For text files, open in editor
    if file['filename'].endswith(('.txt', '.py', '.js', '.html', '.css', '.md')):
        try:
            with open(file['filepath'], 'r') as f:
                content = f.read()
            
            # Get file modification time
            modified_time = datetime.fromtimestamp(os.path.getmtime(file['filepath']))
            
            return render_template('editor.html', 
                                file_id=file['id'],
                                file_name=file['filename'], 
                                file_content=content,
                                file_modified_time=modified_time.strftime('%Y-%m-%d %H:%M:%S'))
        except Exception as e:
            print(f"debug: Error opening file: {str(e)}")
            flash('Could not open file for editing', 'error')
            return redirect(url_for('manage_files'))
    
    # For other files, download
    try:
        return send_from_directory(
            os.path.dirname(file['filepath']),
            os.path.basename(file['filepath']),
            as_attachment=True,
            download_name=file['filename']
        )
    except Exception as e:
        print(f"debug: Error downloading file: {str(e)}")
        flash('Could not download file', 'error')
        return redirect(url_for('manage_files'))
    
@app.route('/save_file/<int:file_id>', methods=['POST'])
def save_file(file_id):
    if 'user_id' not in session:
        return jsonify({'success': False, 'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    if not data or 'content' not in data:
        return jsonify({'success': False, 'error': 'Invalid request'}), 400
    
    conn = get_db_connection()
    file = conn.execute('''
        SELECT filepath 
        FROM secured_files 
        WHERE id = ? AND user_id = ?
    ''', (file_id, session['user_id'])).fetchone()
    
    if not file:
        conn.close()
        return jsonify({'success': False, 'error': 'File not found or access denied'}), 404
    
    try:
        with open(file['filepath'], 'w') as f:
            f.write(data['content'])
        
        log_access(session['user_id'], f"edited file: {file['filepath']}")
        conn.close()
        return jsonify({'success': True})
    except Exception as e:
        conn.close()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/process_Keystrokes_mouse/<int:file_id>', methods=['POST'])
def process_Keystrokes_mouse(file_id):
    if 'user_id' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.get_json()
    if not data or 'keystrokes' not in data:
        return jsonify({'error': 'Invalid request'}), 400
    
    try:
        print(data)
        print(f"DEBUG: Received {len(data['keystrokes'])} keystrokes for verification")

        user_id = session['user_id']
        ''' authenticator = KeystrokeAuthenticator(min_samples=3)
        profile = authenticator.build_profile(keystroke_manager.get_keystroke_features(user_id=user_id))
        print("\nProfile Statistics:")
        print(f"- Clusters: {profile['n_clusters']}")
        print(f"- Noise Points: {profile['n_noise']}")
        print(f"- Threshold: {profile['threshold']:.2f}")'''
        # Record keystrokes in database
        #print("DEBUG: Recording keystrokes in database",data['keystrokes'])
        keystroke_manager.record_keystroke_event(user_id=user_id, keystroke_data=data['keystrokes'])
        print("DEBUG: Keystrokes recorded successfully")
        
        # Get recent features for verification
        # Only verify every 8 keystrokes to reduce load
        if len(data['keystrokes']) >= 8:
            features = keystroke_manager.get_keystroke_features(user_id=user_id, limit=max(20, min(50, len(data['keystrokes']) * 2)))
            print(f"DEBUG: Retrieved {len(features)} features for verification")

            # Check if we have enough features for verification
            if len(features) >= 15:  # Only verify if we have enough data
                dwell_times = [f['dwell_time'] for f in features[-15:]]
                flight_times = [f['flight_time'] for f in features[-15:]]
                print(f"DEBUG: Dwell times: {dwell_times}")
                print(f"DEBUG: Flight times: {flight_times}")

                # Verify typing pattern
                verification = keystroke_manager.verify_enhanced_pattern(
                    user_id=user_id,
                    features=features[-15:]
                )
                print(f"DEBUG: Verification result: {verification}")
                #print("\nVerifying genuine session:")
                '''result = authenticator.verify_session(features[-15:])
                print(f"- Match: {result['match']}")
                print(f"- Confidence: {result['confidence']:.2f}")
                print(f"- Avg Score: {result['avg_anomaly_score']:.2f}")'''
        
            else:
                verification = {
                    'match': True, 
                    'confidence': 0, 
                    'message': 'Collecting more data'}
                print("DEBUG: Not enough data for verification")
        else:
            verification = {
                'match': True, 
                'confidence': 0, 
                'message': 'Need more keystrokes for verification'
            }
            print("DEBUG: Not enough data for verification")
        
        
        # Verify genuine session
        if data['mouse_data']:
            verify_mouse=mouse_manager.verify_user(user_id=user_id, mouse_data=data['mouse_data'])
            print(f"DEBUG: Verification mouse result: {verify_mouse}")
            verification['match']= verification['match'] and verify_mouse

        return jsonify({
            'success': True,
            'verification': verification
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/verify_auth_challenge/<int:file_id>', methods=['POST'])
def verify_auth_challenge(file_id):
    data = request.get_json()
    typed_text = data.get('typed_text')
    target_text = data.get('target_text')
    keystrokes = data.get('keystrokes', [])
    mouse_data = data.get('mouse_data', [])
    print("debug: inside verify_auth_challenge")
    #print("DEBUG: Received keystrokes", keystrokes)
    #print("DEBUG: Received mouse data", mouse_data)
    verify_mouse = True
    if mouse_data:
        verify_mouse=mouse_manager.verify_user(user_id=session['user_id'], mouse_data=mouse_data)
    print(f"DEBUG: Verification mouse result: {verify_mouse}")
    verify_keystrokes={}
    verify_keystrokes['match']= False
    verify_keystrokes['confidence']= 0
    if keystrokes:
        keystroke_features=keystroke_manager.process_keystroke(keystroke_data=keystrokes)
        #print(f"DEBUG: Keystroke features: {keystroke_features}")
        verify_keystrokes=keystroke_manager.verify_enhanced_pattern(user_id=session['user_id'], features=keystroke_features)
    if not verify_mouse and not verify_keystrokes:
        return jsonify(success=False, reason="Mouse or keystroke mismatch")
    # Your custom analysis function
    result = verify_keystrokes
    result['match'] = result['match'] or verify_mouse
    print(f"DEBUG: Verification result: {result}")
    if result['match'] or result['confidence'] > 0.8:
        print("debug: authentication successful")
        return jsonify(success=True)
    else:
        print("debug: authentication failed")
        return jsonify(success=False, confidence=result['confidence'])


if __name__ == '__main__':
    user_manager=User(DATABASE)     #initializes user table automatically
    keystroke_manager=EnhancedKeystrokeAnalyzer(DATABASE)       #initializes keystroke events and keystroke features table automatically
    mouse_manager=MouseAnalyzer(DATABASE)       #initializes  mouse_events , mouse_features and mouse_clusters table automatically
    init_db()
    
    app.run(debug=True)