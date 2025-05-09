import sqlite3
from sqlite3 import Error
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
import json
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import io
import base64
import os
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from collections import deque

class EnhancedKeystrokeAnalyzer:
    def __init__(self, db_path: str = 'keystroke_auth.db'):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        db_path = os.path.join(root_dir, 'keystroke_auth.db')
        self.db_path = db_path
        self._initialize_tables()

    def _initialize_tables(self):
        """Initialize database tables with enhanced features"""
        print('debug: inside _initialize_tables')
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Keystroke events table (unchanged)
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS keystroke_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key_code TEXT NOT NULL,
                event_type TEXT NOT NULL,  
                timestamp REAL NOT NULL,  
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')
            
            # Enhanced keystroke features table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS keystroke_features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                key1 TEXT NOT NULL,
                key2 TEXT NOT NULL,
                dwell_time REAL,        -- press1 to release1
                flight_time REAL,       -- release1 to press2
                pp_time REAL,           -- press1 to press2
                rr_time REAL,          -- release1 to release2
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
            )
            ''')
            
            # Create index for faster queries
            cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_features 
            ON keystroke_features (user_id, key1, key2)
            ''')
            
            conn.commit()
        except Error as e:
            raise Exception(f"Database initialization error: {e}")
        finally:
            if conn:
                conn.close()

    def record_keystroke_event(self, user_id: int, keystroke_data: List[dict[str, Union[str, float, int]]]):
        """Record raw keystroke events and process them"""
        print("debug: inside record_keystroke_event")
        try:
            events = []
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for row in keystroke_data:
                cursor.execute(
                    '''INSERT INTO keystroke_events 
                    (user_id, key_code, event_type, timestamp)
                    VALUES (?, ?, ?, ?)''',
                    (user_id, row['key'], row['event'], row['time'])
                )
                events.append(row)
                
            conn.commit()
            self.process_keystroke_events(events=events, user_id=user_id)
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def process_keystroke_events(self, user_id: int, events: List[dict]):
        """
        Robust processing of keystroke events:
        - Dwell time (press-release same key)
        - Flight time (release-press between keys)
        - Press-Press (PP)
        - Release-Release (RR)
        - Handles modifiers and duplicates
        - Prevents missed digraphs
        """
        print("debug: inside updated process_keystroke_events")
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Sort all events by time
            sorted_events = sorted(events, key=lambda x: x['time'])

            key_down_times = {}       # {key_id: timestamp}
            key_up_times = {}         # {key_id: timestamp}
            full_sequence = []        # [(event_type, key, timestamp)]

            for e in sorted_events:
                key = e['key'].lower()
                timestamp = e['time']
                event_type = e['event']
                full_sequence.append((event_type, key, timestamp))

                if event_type == 'press':
                    key_down_times.setdefault(key, []).append(timestamp)
                elif event_type == 'release':
                    key_up_times.setdefault(key, []).append(timestamp)

            # Dwell Times
            for key in key_down_times:
                downs = key_down_times[key]
                ups = key_up_times.get(key, [])
                for press_time in downs:
                    # Find first release after this press
                    release_time = next((u for u in ups if u > press_time), None)
                    if release_time:
                        dwell = release_time - press_time
                        cursor.execute('''
                            INSERT INTO keystroke_features (user_id, key1, key2, dwell_time, flight_time, pp_time, rr_time)
                            VALUES (?, ?, ?, ?, NULL, NULL, NULL)
                        ''', (user_id, key, key, dwell))

            # Digraphs (from full_sequence)
            presses = [(k, t) for e, k, t in full_sequence if e == 'press']
            releases = [(k, t) for e, k, t in full_sequence if e == 'release']

            for i in range(len(presses) - 1):
                k1, t1 = presses[i]
                k2, t2 = presses[i + 1]

                # Press-Press (pp), Release-Release (rr), Flight (release[k1] -> press[k2]), Dwell[k2]
                pp = t2 - t1

                # Find release for k1
                release1 = next((r[1] for r in releases if r[0] == k1 and r[1] > t1), None)
                # Find release for k2
                release2 = next((r[1] for r in releases if r[0] == k2 and r[1] > t2), None)
                # Find dwell for k2
                dwell2 = release2 - t2 if release2 else None
                # Flight (release of k1 to press of k2)
                flight = t2 - release1 if release1 else None
                # RR
                rr = release2 - release1 if release1 and release2 else None

                # Insert only if we have dwell2 and at least one timing metric
                if dwell2 is not None:
                    cursor.execute('''
                        INSERT INTO keystroke_features (user_id, key1, key2, dwell_time, flight_time, pp_time, rr_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        user_id, k1, k2,
                        dwell2,
                        flight if flight is not None else None,
                        pp if pp is not None else None,
                        rr if rr is not None else None
                    ))
            cursor.execute("""
                DELETE FROM keystroke_features
                WHERE flight_time IS NULL
                AND pp_time IS NULL
                AND rr_time IS NULL
            """)

            conn.commit()

        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_keystroke_events(self, user_id: int, limit: int = None) -> List[Dict]:  
        """
        Retrieve raw keystroke events for a user
        
        Args:
            user_id: ID of the user
            limit: Maximum number of events to return
            
        Returns:
            List of dictionaries containing keystroke events
        """
        try:
            print("debug: inside get_keystroke_events")
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT key_code, event_type, timestamp
                FROM keystroke_events 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
                
            cursor.execute(query, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()
    
    def get_keystroke_features(self, user_id: int, limit: int = None) -> List[Dict]:
        """
        Retrieve enhanced keystroke features with all timing metrics
        
        Args:
            user_id: ID of the user
            limit: Maximum number of features to return
            
        Returns:
            List of dictionaries containing all timing features
        """
        try:
            print("debug: inside get_keystroke_features")
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT key1, key2, dwell_time, flight_time, pp_time, rr_time, timestamp
                FROM keystroke_features 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            '''
            
            if limit:
                query += f' LIMIT {limit}'
                
            cursor.execute(query, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def get_keystroke_features_test(self, user_id: int, limit: int = None) -> List[Dict]:
        """
        Retrieve enhanced keystroke features with all timing metrics
        
        Args:
            user_id: ID of the user
            limit: Maximum number of features to return
            
        Returns:
            List of dictionaries containing all timing features
        """
        try:
            print("debug: inside get_keystroke_features_test")
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            query = '''
                SELECT key1, key2, dwell_time, flight_time, pp_time, rr_time
                FROM keystroke_features_test
                WHERE user_id = ? 
                
            '''
            
            if limit:
                query += f' LIMIT {limit}'
                
            cursor.execute(query, (user_id,))
            return [dict(row) for row in cursor.fetchall()]
        except Error as e:
            raise Exception(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def calculate_enhanced_metrics(self, user_id: int) -> Dict:
        """
        Calculate comprehensive typing metrics using all available features
        """
        print('debug: inside calculate_enhanced_metrics')
        features = self.get_keystroke_features(user_id)
        
        if not features or len(features) < 5:
            return {}
        
        # Filter valid features
        valid_features = [
            f for f in features 
            if f['dwell_time'] and f['dwell_time'] > 10
            and f['flight_time'] and f['flight_time'] > 10
            and f['pp_time'] and f['pp_time'] > 0
            and f['rr_time'] and f['rr_time'] > 0
        ]
        
        if len(valid_features) < 3:
            return {}
        
        # Convert to numpy arrays
        dwell = np.array([f['dwell_time'] for f in valid_features])
        flight = np.array([f['flight_time'] for f in valid_features])
        pp = np.array([f['pp_time'] for f in valid_features])
        rr = np.array([f['rr_time'] for f in valid_features])
        
        # Calculate basic statistics
        metrics = {
            'dwell': {
                'mean': float(np.mean(dwell)),
                'std': float(np.std(dwell)),
                'median': float(np.median(dwell))
            },
            'flight': {
                'mean': float(np.mean(flight)),
                'std': float(np.std(flight)),
                'median': float(np.median(flight))
            },
            'press_press': {
                'mean': float(np.mean(pp)),
                'std': float(np.std(pp)),
                'median': float(np.median(pp))
            },
            'release_release': {
                'mean': float(np.mean(rr)),
                'std': float(np.std(rr)),
                'median': float(np.median(rr))
            }
        }
        
        # Calculate timing relationships
        metrics['dwell_flight_ratio'] = float(np.median(dwell / flight))
        metrics['pp_rr_ratio'] = float(np.median(pp / rr))
        
        # Calculate typing speed using multiple methods
        avg_char_time = np.mean(pp) / 1000  # in seconds
        metrics['typing_speed_wpm_pp'] = float((60 / avg_char_time) / 5)
        
        avg_char_time_rr = np.mean(rr) / 1000
        metrics['typing_speed_wpm_rr'] = float((60 / avg_char_time_rr) / 5)
        
        # Key transition patterns
        key_counts = defaultdict(int)
        for f in valid_features:
            key_pair = f"{f['key1']}-{f['key2']}"
            key_counts[key_pair] += 1
        
        metrics['common_patterns'] = sorted(
            [(k, v) for k, v in key_counts.items() if v >= 2],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Consistency scores
        metrics['consistency'] = {
            'dwell': float(1 / (1 + (metrics['dwell']['std'] / metrics['dwell']['mean']))),
            'flight': float(1 / (1 + (metrics['flight']['std'] / metrics['flight']['mean']))),
            'overall': float(np.mean([
                1 / (1 + (metrics['dwell']['std'] / metrics['dwell']['mean'])),
                1 / (1 + (metrics['flight']['std'] / metrics['flight']['mean']))
            ]))
        }
        
        return metrics

    def generate_typing_profile(self, user_id: int) -> Dict:
        """
        Generate comprehensive profile with visualization using all features
        """
        print("debug: inside generate_enhanced_profile")
        features = self.get_keystroke_features(user_id)
        
        if not features:
            return {}
        
        # Prepare data - using dwell, flight, pp, rr times
        X = np.array([
            [f['dwell_time'], f['flight_time'], f['pp_time'], f['rr_time']] 
            for f in features
            if f['dwell_time'] and f['flight_time'] and f['pp_time'] and f['rr_time']
        ])
        
        if len(X) < 10:
            return {}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster using DBSCAN with auto-tuned parameters
        neigh = NearestNeighbors(n_neighbors=5)
        neigh.fit(X_scaled)
        distances, _ = neigh.kneighbors(X_scaled)
        eps = np.percentile(distances[:, -1], 75)
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        clusters = dbscan.fit_predict(X_scaled)
        
        # Generate 2D visualization (first two dimensions)
        plt.figure(figsize=(12, 8))
        
        unique_clusters = set(clusters)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_clusters)))
        
        for cluster, color in zip(unique_clusters, colors):
            if cluster == -1:
                plt.scatter(
                    X[clusters == cluster, 0],
                    X[clusters == cluster, 1],
                    c='gray', alpha=0.2, label='Noise'
                )
            else:
                plt.scatter(
                    X[clusters == cluster, 0],
                    X[clusters == cluster, 1],
                    c=[color], label=f'Cluster {cluster}'
                )
        
        plt.title('Enhanced Keystroke Profile (Dwell vs Flight Times)')
        plt.xlabel('Dwell Time (ms)')
        plt.ylabel('Flight Time (ms)')
        plt.legend()
        
        # Save plot to base64
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_base64 = base64.b64encode(img.getvalue()).decode('utf-8')
        plt.close()
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster in unique_clusters:
            if cluster != -1:
                cluster_points = X[clusters == cluster]
                stats = {
                    'cluster_id': int(cluster),
                    'size': int(cluster_points.shape[0]),
                    'dwell_mean': float(np.mean(cluster_points[:, 0])),
                    'flight_mean': float(np.mean(cluster_points[:, 1])),
                    'pp_mean': float(np.mean(cluster_points[:, 2])),
                    'rr_mean': float(np.mean(cluster_points[:, 3]))
                }
                cluster_stats.append(stats)
        
        return {
            'statistics': self.calculate_enhanced_metrics(user_id),
            'clusters': cluster_stats,
            'n_clusters': len(unique_clusters) - 1,
            'n_noise': int(np.sum(clusters == -1)),
            'visualization': plot_base64
        }

    def verify_enhanced_pattern(self, user_id: int, features: List[Dict]) -> Dict:
        """
        Verify typing pattern using both IsolationForest and DBSCAN
        """
        def error_response(message):
            return {
                'match': False,
                'confidence': 0,
                'message': message,
                'method': 'error'
            }

        try:
            if not features or len(features) < 10:
                return error_response("Insufficient data (need at least 10 samples)")

            historical_data = self.get_keystroke_features(user_id, limit=500)
            if not historical_data or len(historical_data) < 25:
                return error_response("Insufficient historical data")

            historical_data = historical_data[:len(historical_data)-20]

            X_hist = np.array([
                [f['dwell_time'], f['flight_time'], f['pp_time'], f['rr_time']] 
                for f in historical_data
                if (f['dwell_time'] and f['dwell_time'] > 10 and f['dwell_time'] < 2000 and
                    f['flight_time'] and f['flight_time'] > 10 and f['flight_time'] < 2000 and
                    f['pp_time'] and f['pp_time'] > 0 and f['pp_time'] < 2000 and
                    f['rr_time'] and f['rr_time'] > 0 and f['rr_time'] < 2000)
            ])

            X_current = np.array([
                [f['dwell_time'], f['flight_time'], f['pp_time'], f['rr_time']] 
                for f in features
                if (f.get('dwell_time') and f.get('dwell_time') > 10 and f.get('dwell_time') < 2000 and
                    f.get('flight_time') and f.get('flight_time') > 10 and f.get('flight_time') < 2000 and
                    f.get('pp_time') and f.get('pp_time') > 0 and f.get('pp_time') < 2000 and
                    f.get('rr_time') and f.get('rr_time') > 0 and f.get('rr_time') < 2000)
            ])

            if len(X_current) < 10:
                return error_response("Not enough valid samples in current data")

            # Scale data
            scaler = RobustScaler()
            X_hist_scaled = scaler.fit_transform(X_hist)
            X_current_scaled = scaler.transform(X_current)

            # IsolationForest
            clf_iso = IsolationForest(
                n_estimators=200,
                max_samples=min(256, len(X_hist_scaled)),
                contamination=0.05,  # Expect 5% outliers
                random_state=42,
                verbose=0
            )
            clf_iso.fit(X_hist_scaled)
            # Get scores for historical data to determine threshold
            hist_scores = clf_iso.score_samples(X_hist_scaled)
            threshold = np.percentile(hist_scores, 10)  # Use 10th percentile as threshold
            
            # Score current data
            iso_scores = clf_iso.score_samples(X_current_scaled) 
            avg_iso_score = float(np.mean(iso_scores))
            iso_match = avg_iso_score >= threshold 
            
            # Calculate confidence based on distance from threshold
            score_range = np.max(hist_scores) - threshold
            iso_confidence = min(1.0, max(0.0, 
                (avg_iso_score - threshold) / (score_range + 1e-8))
            )


            # DBscan
            neigh = NearestNeighbors(n_neighbors=5)
            neigh.fit(X_hist_scaled)
            distances, _ = neigh.kneighbors(X_hist_scaled)
            eps = np.percentile(distances[:, -1], 75)  # Auto-tune eps
            
            clf_db = DBSCAN(eps=eps, min_samples=5)
            combined = np.vstack([X_hist_scaled, X_current_scaled])
            dbscan_labels = clf_db.fit_predict(combined)
            
            current_labels = dbscan_labels[-len(X_current):]
            n_noise = list(current_labels).count(-1)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            
            
            dbscan_match = (n_noise / len(current_labels)) < 0.4 
            
            # Final decision requires both methods to agree
            final_match = iso_match and dbscan_match
            combined_confidence = float(np.mean([iso_confidence, (1 - (n_noise / len(current_labels)))]))
            
            return {
                'match': final_match,
                'confidence': combined_confidence,
                'avg_score': avg_iso_score,
                'scores': [float(s) for s in iso_scores],
                'n_samples': len(X_current),
                'iso_forest': {
                    'match': bool(iso_match),
                    'confidence': iso_confidence,
                    'avg_score': avg_iso_score,
                    'threshold': float(threshold),
                },
                'dbscan': {
                    'match': dbscan_match,
                    'confidence': float(1 - (n_noise / len(current_labels))),
                    'n_clusters': n_clusters,
                    'noise_points': n_noise,
                    'labels': dbscan_labels.tolist(),
                    'eps': float(eps)
                },
                'message': None if final_match else "Typing pattern deviation",
                'method': 'isolation_forest + dbscan'
            }

        except Exception as e:
            print(f"Verification error: {str(e)}")
            return error_response("System error during verification")

    def verify_enhanced_pattern_test(self, user_id: int, features: List[Dict]) -> Dict:
        """
        Verify typing pattern using both IsolationForest and DBSCAN
        """
        def error_response(message):
            return {
                'match': False,
                'confidence': 0,
                'message': message,
                'method': 'error'
            }

        try:
            if not features or len(features) < 10:
                return error_response("Insufficient data (need at least 10 samples)")

            historical_data = self.get_keystroke_features_test(user_id, limit=1000)
            if not historical_data or len(historical_data) < 25:
                return error_response("Insufficient historical data")

            historical_data = historical_data[:len(historical_data)-20]

            X_hist = np.array([
                [f['dwell_time'], f['flight_time'], f['pp_time'], f['rr_time']] 
                for f in historical_data
                if (f['dwell_time'] and f['dwell_time'] > 0.10 and f['dwell_time'] < 0.2000 and
                    f['flight_time'] and f['flight_time'] > 0.10 and f['flight_time'] < 0.2000 and
                    f['pp_time'] and f['pp_time'] > 0 and f['pp_time'] < 0.2000 and
                    f['rr_time'] and f['rr_time'] > 0 and f['rr_time'] < 0.2000)
            ])

            X_current = np.array([
                [f['dwell_time'], f['flight_time'], f['pp_time'], f['rr_time']] 
                for f in features
                if (f.get('dwell_time') and f.get('dwell_time') > 0.10 and f.get('dwell_time') < 0.2000 and
                    f.get('flight_time') and f.get('flight_time') > 0.10 and f.get('flight_time') < 0.2000 and
                    f.get('pp_time') and f.get('pp_time') > 0 and f.get('pp_time') < 0.2000 and
                    f.get('rr_time') and f.get('rr_time') > 0 and f.get('rr_time') < 0.2000)
            ])

            if len(X_current) ==0:
                return error_response(features[0])

            # Scale data
            scaler = RobustScaler()
            X_hist_scaled = scaler.fit_transform(X_hist)
            X_current_scaled = scaler.transform(X_current)

            # IsolationForest
            # Improved IsolationForest with adjusted parameters
            clf_iso = IsolationForest(
                n_estimators=200,
                max_samples=min(256, len(X_hist_scaled)),
                contamination=0.05,  # Expect 5% outliers
                random_state=42,
                verbose=0
            )
            clf_iso.fit(X_hist_scaled)
            # Get scores for historical data to determine threshold
            hist_scores = clf_iso.score_samples(X_hist_scaled)
            threshold = np.percentile(hist_scores, 10)  # Use 10th percentile as threshold
            
            # Score current data
            iso_scores = clf_iso.score_samples(X_current_scaled)
            avg_iso_score = float(np.mean(iso_scores))
            iso_match = avg_iso_score >= threshold
            
            # Calculate confidence based on distance from threshold
            score_range = np.max(hist_scores) - threshold
            iso_confidence = min(1.0, max(0.0, 
                (avg_iso_score - threshold) / (score_range + 1e-8))
            )


           # Improved DBSCAN with auto-tuned parameters
            neigh = NearestNeighbors(n_neighbors=5)
            neigh.fit(X_hist_scaled)
            distances, _ = neigh.kneighbors(X_hist_scaled)
            eps = np.percentile(distances[:, -1], 75)  # Auto-tune eps
            
            clf_db = DBSCAN(eps=eps, min_samples=5)
            combined = np.vstack([X_hist_scaled, X_current_scaled])
            dbscan_labels = clf_db.fit_predict(combined)
            
            current_labels = dbscan_labels[-len(X_current):]
            n_noise = list(current_labels).count(-1)
            n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            
            # DBSCAN match criteria:
            # 1. At least 60% of points not noise
            # 2. Majority in existing clusters (not new clusters)
            dbscan_match = (n_noise / len(current_labels)) < 0.4
            
            # Final decision requires both methods to agree
            final_match = iso_match and dbscan_match
            combined_confidence = float(np.mean([iso_confidence, (1 - (n_noise / len(current_labels)))]))
            
            return {
                'match': final_match,
                'confidence': combined_confidence,
                'avg_score': avg_iso_score,
                'scores': [float(s) for s in iso_scores],
                'n_samples': len(X_current),
                'iso_forest': {
                    'match': bool(iso_match),
                    'confidence': iso_confidence,
                    'avg_score': avg_iso_score,
                    'threshold': float(threshold),
                },
                'dbscan': {
                    'match': dbscan_match,
                    'confidence': float(1 - (n_noise / len(current_labels))),
                    'n_clusters': n_clusters,
                    'noise_points': n_noise,
                    'labels': dbscan_labels.tolist(),
                    'eps': float(eps)
                },
                'message': None if final_match else "Typing pattern deviation",
                'method': 'isolation_forest + dbscan'
            }

        except Exception as e:
            print(f"Verification error: {str(e)}")
            return error_response("System error during verification")

    def process_keystroke(self, keystroke_data: List[dict]) -> List[dict]:
        """
        Processes keystroke events to extract features:
        - Dwell time (press-release same key)
        - Flight time (release-press between keys)
        - Press-Press (PP)
        - Release-Release (RR)
        
        Returns:
            List[dict]: List of feature dictionaries
        """
        print("debug: inside process_keystroke")

        # Sort all events by time
        sorted_events = sorted(keystroke_data, key=lambda x: x['time'])

        key_down_times = {}       # {key: [timestamps]}
        key_up_times = {}         # {key: [timestamps]}
        full_sequence = []        # [(event_type, key, timestamp)]
        features = []

        for e in sorted_events:
            key = e['key'].lower()
            timestamp = e['time']
            event_type = e['event']
            full_sequence.append((event_type, key, timestamp))

            if event_type == 'press':
                key_down_times.setdefault(key, []).append(timestamp)
            elif event_type == 'release':
                key_up_times.setdefault(key, []).append(timestamp)

        # Dwell Times
        for key in key_down_times:
            downs = key_down_times[key]
            ups = key_up_times.get(key, [])
            for press_time in downs:
                release_time = next((u for u in ups if u > press_time), None)
                if release_time:
                    dwell = release_time - press_time
                    features.append({
                        'key1': key,
                        'key2': key,
                        'dwell_time': dwell,
                        'flight_time': None,
                        'pp_time': None,
                        'rr_time': None
                    })

        # Digraphs (from full_sequence)
        presses = [(k, t) for e, k, t in full_sequence if e == 'press']
        releases = [(k, t) for e, k, t in full_sequence if e == 'release']

        for i in range(len(presses) - 1):
            k1, t1 = presses[i]
            k2, t2 = presses[i + 1]

            pp = t2 - t1
            release1 = next((r[1] for r in releases if r[0] == k1 and r[1] > t1), None)
            release2 = next((r[1] for r in releases if r[0] == k2 and r[1] > t2), None)
            dwell2 = release2 - t2 if release2 else None
            flight = t2 - release1 if release1 else None
            rr = release2 - release1 if release1 and release2 else None

            if dwell2 is not None:
                features.append({
                    'key1': k1,
                    'key2': k2,
                    'dwell_time': dwell2,
                    'flight_time': flight if flight is not None else None,
                    'pp_time': pp if pp is not None else None,
                    'rr_time': rr if rr is not None else None
                })

        # Optionally filter if all timing metrics are None (like your old delete query)
        features = [
            f for f in features if not (
                f['flight_time'] is None and 
                f['pp_time'] is None and 
                f['rr_time'] is None
            )
        ]

        return features
