import sqlite3
from sqlite3 import Error
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import json
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
import base64
import math
import os

class MouseAnalyzer:
    def __init__(self, db_path: str = 'keystroke_auth.db'):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_path = os.path.join(root_dir, 'keystroke_auth.db')
        self.db_path = db_path
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize database tables if they don't exist"""
        print("debug: inside mouse/_initialize_tables")
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Mouse events table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mouse_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                x INTEGER NOT NULL,
                y INTEGER NOT NULL,
                event_type TEXT NOT NULL,  
                button TEXT,  
                pressed BOOLEAN DEFAULT NULL,  
                timestamp REAL NOT NULL,  
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')
            #print("debug: created mouse_events db")
            
            # Mouse features table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mouse_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                distance REAL NOT NULL,
                velocity REAL NOT NULL,
                angle REAL NOT NULL,  
                acceleration REAL,
                curvature REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')
            #print("debug: created mouse_features db")
            
            # Mouse behavioral clusters
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS mouse_clusters (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                cluster_id INTEGER NOT NULL,
                velocity_mean REAL NOT NULL,
                velocity_std REAL NOT NULL,
                angle_mean REAL NOT NULL,
                angle_std REAL NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')
            #print("debug: created mouse_clusters db")
            
            conn.commit()
        except Error as e:
            raise Exception(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()

    def record_mouse_event(self, user_id: int, mouse_data : List[dict[ Union[str, float, int]]]):
        """
        Record a raw mouse event
        
        Args:
            user_id: ID of the user
            x: X coordinate
            y: Y coordinate
            event_type: 'move', 'click', or 'scroll'
            button: For clicks - 'left', 'right', 'middle'
        """
        print("debug: inside record_mouse_event")
        try:
            events=[]
            #print("debug: inserting mouse data")
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            for row in mouse_data:
                    #print(row)
                    if row['event']=='move':
                        cursor.execute(
                            '''INSERT INTO mouse_events 
                            (user_id, x, y, event_type, timestamp)
                            VALUES ( ?, ?, ?, ?, ?)''',
                            (user_id, row['x'], row['y'], row['event'], row['time'])
                        )
                    elif row['event']=='click':
                        cursor.execute(
                            '''INSERT INTO mouse_events 
                            (user_id, x, y, event_type, button, pressed, timestamp)
                            VALUES ( ?, ?, ?, ?, ?, ?, ?)''',
                            (user_id, row['x'], row['y'], row['event'],row['button'], row['pressed'] ,row['time'])
                        )
                    events.append(row)            
                    conn.commit()
            #print("debug: succusfully installed mouse data")
            self.process_mouse_events(user_id=user_id,events=events)
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def process_mouse_events(self, user_id: int, events: List[dict]) -> int:
        """
        Process raw mouse events into movement features and store in features table
        
        Returns:
            Number of movement features processed
        """
        print("debug: inside process_mouse_events")
        if len(events) < 2:
            return 0

        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            processed_features = 0
            
            # Calculate movement features between consecutive points
            for i in range(1, len(events)):
                prev = events[i-1]
                curr = events[i]
                
                # Time difference in seconds
                time_diff = curr['time'] - prev['time']
                if time_diff == 0:
                    continue
                if time_diff < 0:
                    raise ValueError("Time difference cannot be negative.")
                    continue
                
                # Distance in pixels
                dx = curr['x'] - prev['x']
                dy = curr['y'] - prev['y']
                distance = math.sqrt(dx**2 + dy**2)
                
                # Velocity in pixels/second
                velocity = distance / time_diff
                
                # Angle in radians (-π to π)
                angle = math.atan2(dy, dx)
                
                # Acceleration (needs at least 3 points)
                acceleration = 0
                if i > 1:
                    prev_velocity = math.sqrt(
                        (prev['x'] - events[i-2]['x'])**2 + 
                        (prev['y'] - events[i-2]['y'])**2
                    ) / (prev['time'] - events[i-2]['time'])
                    acceleration = (velocity - prev_velocity) / time_diff
                
                # Curvature (change in angle)
                curvature = 0
                if i > 1:
                    prev_angle = math.atan2(
                        prev['y'] - events[i-2]['y'],
                        prev['x'] - events[i-2]['x']
                    )
                    curvature = abs(angle - prev_angle) / distance if distance > 0 else 0
                
                # Store the features
                cursor.execute('''
                    INSERT INTO mouse_features 
                    (user_id, distance, velocity, angle, acceleration, curvature)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, distance, velocity, angle, acceleration, curvature))
                
                processed_features += 1
            
            conn.commit()
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_mouse_events(self, user_id: int) -> List[Dict]:
        """
        Retrieve processed mouse events for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of dictionaries containing movement features
        """
        print("debug: inside get_mouse_events")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT x, y, event_type, button, pressed, timestamp
                FROM mouse_events
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id, ))
            
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_mouse_features(self, user_id: int) -> List[Dict]:
        """
        Retrieve processed mouse features for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of dictionaries containing movement features
        """
        print("debug: inside get_mouse_features")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT distance, velocity, angle, acceleration, curvature, timestamp
                FROM mouse_features
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id, ))
            
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def genarate_mouse_movement_profile(self, user_id: int) -> List[Dict]:
        """
        Generate mouse movement profile for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of dictionaries containing mouse movement profile
        """
        print("debug: inside genarate_mouse_movement_profile")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT distance, velocity, angle, acceleration, curvature, timestamp
                FROM mouse_features
                WHERE user_id = ?
                ORDER BY timestamp DESC
            ''', (user_id, ))
            
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def cluster_mouse_features(self, user_id: int) -> List[Dict]:
        """
        Cluster mouse features for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of dictionaries containing clustered mouse features
        """
        print("debug: inside cluster_mouse_features")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT distance, velocity, angle, acceleration, curvature
                FROM mouse_features
                WHERE user_id = ?
            ''', (user_id, ))
            
            features = np.array(cursor.fetchall())
            
            if len(features) < 2:
                return []
            
            # DBSCAN clustering
            dbscan = DBSCAN(eps=0.5, min_samples=5).fit(features)
            labels = dbscan.labels_
            
            # Store clusters in the database
            unique_labels = set(labels)
            clusters = []
            
            for label in unique_labels:
                if label == -1:
                    continue  # Skip noise
                
                cluster_points = features[labels == label]
                
                velocity_mean = np.mean(cluster_points[:, 1])
                velocity_std = np.std(cluster_points[:, 1])
                angle_mean = np.mean(cluster_points[:, 2])
                angle_std = np.std(cluster_points[:, 2])
                
                cursor.execute('''
                    INSERT INTO mouse_clusters 
                    (user_id, cluster_id, velocity_mean, velocity_std, angle_mean, angle_std)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (user_id, label, velocity_mean, velocity_std, angle_mean, angle_std))
                
                clusters.append({
                    'user_id': user_id,
                    'cluster_id': label,
                    'velocity_mean': velocity_mean,
                    'velocity_std': velocity_std,
                    'angle_mean': angle_mean,
                    'angle_std': angle_std
                })
            
            conn.commit()
            
            return clusters
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_mouse_clusters(self, user_id: int) -> List[Dict]:
        """
        Retrieve clustered mouse features for a user
        
        Args:
            user_id: ID of the user
            
        Returns:
            List of dictionaries containing clustered mouse features
        """
        print("debug: inside get_mouse_clusters")
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT cluster_id, velocity_mean, velocity_std, angle_mean, angle_std
                FROM mouse_clusters
                WHERE user_id = ?
                ORDER BY cluster_id DESC
            ''', (user_id, ))
            
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def verify_user(self, user_id: int, mouse_data: List[dict]) -> bool:
        """
        Verify a user based on mouse movement patterns using a hybrid approach
        with DBSCAN clustering and Isolation Forest
        
        Args:
            user_id: ID of the user
            mouse_data: List of mouse events to verify
            
        Returns:
            True if verification is successful, False otherwise
        """
        print("debug: inside verify_user")
        try:
            # Retrieve stored features for the user
            stored_features = self.get_mouse_features(user_id)
            #print("debug: stored features", stored_features)
            if not stored_features:
                return False
            
            # Cluster the stored features
            self.cluster_mouse_features(user_id)
            stored_clusters = self.get_mouse_clusters(user_id)
            #print("debug: stored clusters", stored_clusters)
            if not stored_features:
                return False
            
            # Process the new mouse data
            self.process_mouse_events(user_id=user_id, events=mouse_data)
            
            # Retrieve the processed features
            new_features = self.get_mouse_features(user_id)[len(stored_features):]
            #print("debug: new features", new_features)
            if not new_features:
                return True
                        
            if not stored_clusters:
                return False
            
            # Prepare data for Isolation Forest
            stored_features_values = [list(feature.values())[:-1] for feature in stored_features]
            new_features_values = [list(feature.values())[:-1] for feature in new_features]
            
            #print("debug: stored features values", stored_features_values)
            #print("debug: new features values", new_features_values)
            # Use Isolation Forest for anomaly detection
            model = IsolationForest(contamination=0.1)
            model.fit(stored_features_values)
            
            # Predict anomalies in new features
            predictions = model.predict(new_features_values)
            iso_result= True
            # If any prediction is -1 (anomaly), return False
            if not all(pred == 1 for pred in predictions):
                iso_result= False
            
            DBSCAN_result= True
            # Verify clusters
            for new_feature in new_features:
                matched_cluster = False
                for cluster in stored_clusters:
                    velocity_mean = cluster['velocity_mean']
                    velocity_std = cluster['velocity_std']
                    angle_mean = cluster['angle_mean']
                    angle_std = cluster['angle_std']
                    
                    # Check if the new feature falls within the cluster's range
                    if (abs(new_feature['velocity'] - velocity_mean) <= 2 * velocity_std and
                        abs(new_feature['angle'] - angle_mean) <= 2 * angle_std):
                        matched_cluster = True
                        break
                
                if not matched_cluster:
                    DBSCAN_result= False
                    break
            # If any models agree, return True
            if iso_result or DBSCAN_result:
                return True
            else:
                return False
            
        # Handle any exceptions that occur during verification
        except Exception as e:
            raise Exception(f"Verification error: {e}")
        