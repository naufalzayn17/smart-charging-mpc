#!/usr/bin/env python3
"""
LAYER 3 ADAPTIVE LEARNING SYSTEM - HEADLESS HIERARCHICAL MPC
============================================================

Layer 3: Strategic Planner dengan adaptive weight tuning
- Pattern recognition untuk usage patterns
- Adaptive MPC weight tuning berdasarkan performance
- Weather pattern learning
- Long-term optimization (24-hour planning)
- Headless operation untuk sistem standalone

Author: Dzaky Naufal K
Date: July 2, 2025
Version: 3.0 - Headless Hierarchical MPC
"""

import sys
import os
import time
import json
import logging
import numpy as np
import pandas as pd
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Configure logging for headless operation
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - Layer3-Adaptive - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/var/log/mpc_layer3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics for adaptive learning"""
    pv_utilization: float       # Fraction of available PV used
    grid_dependency: float      # Fraction of power from grid
    soc_stability: float        # SOC variance (lower is better)
    efficiency_overall: float   # Overall system efficiency
    cost_per_kwh: float        # Average cost per kWh
    switching_frequency: float  # Switches per day
    comfort_score: float       # SOC comfort score (0-1)
    battery_health_score: float # Battery health score (0-1)

@dataclass
class UsagePattern:
    """Detected usage patterns"""
    peak_pv_hours: List[int]        # Hours with peak PV availability
    low_tariff_windows: List[Tuple[int, int]]  # (start_hour, end_hour)
    typical_load_profile: List[float] # 24-hour load profile
    seasonal_factor: float          # Seasonal adjustment factor
    weekend_factor: float           # Weekend usage factor

class AdaptiveMPCWeights:
    """Adaptive MPC weight tuning system"""
    
    def __init__(self):
        self.weights = {
            'efficiency': 1.0,
            'cost': 0.5,
            'battery_health': 0.8,
            'comfort': 0.3
        }
        
        # Learning parameters
        self.learning_rate = 0.01
        self.performance_history = []
        self.adaptation_count = 0
        self.min_history_length = 10
        
        # Performance targets
        self.targets = {
            'pv_utilization': 0.85,
            'grid_dependency': 0.4,
            'efficiency_overall': 0.88,
            'cost_reduction': 0.4
        }
        
        logger.info("Adaptive MPC Weights initialized")
    
    def update_weights(self, performance_metrics: PerformanceMetrics) -> Dict[str, float]:
        """Update MPC weights based on performance feedback"""
        
        # Add to history
        self.performance_history.append({
            'timestamp': time.time(),
            'metrics': performance_metrics,
            'weights': self.weights.copy()
        })
        
        # Keep only recent history (last 100 entries)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
        
        # Need sufficient history for adaptation
        if len(self.performance_history) < self.min_history_length:
            return self.weights
        
        # Calculate performance gradients
        gradient = self._compute_gradient(performance_metrics)
        
        # Apply gradient update with clipping
        new_weights = {}
        for key in self.weights:
            new_weights[key] = np.clip(
                self.weights[key] + self.learning_rate * gradient.get(key, 0),
                0.1, 2.0  # Reasonable bounds
            )
        
        # Normalize weights to prevent runaway growth
        total_weight = sum(new_weights.values())
        for key in new_weights:
            new_weights[key] = new_weights[key] / total_weight * 2.6  # Target sum ≈ 2.6
        
        self.weights = new_weights
        self.adaptation_count += 1
        
        logger.info(f"Weights adapted (#{self.adaptation_count}): {self.weights}")
        
        return self.weights
    
    def _compute_gradient(self, current_metrics: PerformanceMetrics) -> Dict[str, float]:
        """Compute gradient for weight adaptation"""
        gradient = {}
        
        # PV utilization gradient
        pv_error = self.targets['pv_utilization'] - current_metrics.pv_utilization
        gradient['efficiency'] = pv_error * 2.0  # Increase efficiency weight if low PV utilization
        
        # Grid dependency gradient  
        grid_error = current_metrics.grid_dependency - self.targets['grid_dependency']
        gradient['cost'] = grid_error * 1.5  # Increase cost weight if high grid dependency
        
        # Efficiency gradient
        eff_error = self.targets['efficiency_overall'] - current_metrics.efficiency_overall
        gradient['efficiency'] += eff_error * 1.0
        
        # Battery health gradient (penalize extreme SOC usage)
        health_penalty = 1.0 - current_metrics.battery_health_score
        gradient['battery_health'] = health_penalty * 1.2
        
        # Comfort gradient (smooth SOC operation)
        comfort_penalty = 1.0 - current_metrics.comfort_score
        gradient['comfort'] = comfort_penalty * 0.8
        
        return gradient

class PatternRecognition:
    """Pattern recognition for usage and environmental patterns"""
    
    def __init__(self, db_path: str = '/var/log/mpc_layer3_data.db'):
        self.db_path = db_path
        self.init_database()
        
        # Pattern models
        self.load_model = None
        self.pv_model = None
        self.weather_model = None
        
        # Detected patterns
        self.usage_patterns = None
        
        logger.info("Pattern Recognition system initialized")
    
    def init_database(self):
        """Initialize database for pattern storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Pattern analysis table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS usage_patterns (
                    id INTEGER PRIMARY KEY,
                    date TEXT,
                    hour INTEGER,
                    day_of_week INTEGER,
                    month INTEGER,
                    pv_power REAL,
                    load_power REAL,
                    grid_power REAL,
                    soc REAL,
                    temperature REAL,
                    weather_condition TEXT
                )
            ''')
            
            # Learned patterns table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS learned_patterns (
                    id INTEGER PRIMARY KEY,
                    pattern_type TEXT,
                    pattern_data TEXT,
                    confidence REAL,
                    last_updated REAL
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Pattern recognition database initialized")
            
        except Exception as e:
            logger.error(f"Pattern database initialization error: {e}")
    
    def analyze_usage_patterns(self) -> UsagePattern:
        """Analyze historical data to detect usage patterns"""
        try:
            # Load recent data (last 30 days)
            conn = sqlite3.connect(self.db_path)
            query = '''
                SELECT hour, pv_power, load_power, grid_power, day_of_week, month
                FROM usage_patterns 
                WHERE date >= date('now', '-30 days')
                ORDER BY date, hour
            '''
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) < 24:  # Need at least 24 hours of data
                return self._get_default_patterns()
            
            # Analyze PV patterns
            pv_by_hour = df.groupby('hour')['pv_power'].mean()
            peak_pv_hours = pv_by_hour.nlargest(6).index.tolist()  # Top 6 hours
            
            # Analyze load patterns
            load_by_hour = df.groupby('hour')['load_power'].mean()
            typical_load_profile = load_by_hour.values.tolist()
            
            # Detect low tariff windows (off-peak hours)
            # Assume low tariff during 23:00-17:00 (avoid peak 17:00-22:00)
            low_tariff_windows = [(23, 6), (7, 16)]  # Night and morning
            
            # Seasonal and weekend factors
            seasonal_factor = self._calculate_seasonal_factor(df)
            weekend_factor = self._calculate_weekend_factor(df)
            
            patterns = UsagePattern(
                peak_pv_hours=peak_pv_hours,
                low_tariff_windows=low_tariff_windows,
                typical_load_profile=typical_load_profile,
                seasonal_factor=seasonal_factor,
                weekend_factor=weekend_factor
            )
            
            # Store learned patterns
            self._store_patterns(patterns)
            
            logger.info(f"Usage patterns analyzed: PV peaks at hours {peak_pv_hours}")
            
            return patterns
            
        except Exception as e:
            logger.error(f"Pattern analysis error: {e}")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> UsagePattern:
        """Default patterns when insufficient data"""
        return UsagePattern(
            peak_pv_hours=[10, 11, 12, 13, 14, 15],  # 10 AM - 3 PM
            low_tariff_windows=[(23, 6), (7, 16)],   # Night and morning
            typical_load_profile=[500] * 24,          # Constant 500W load
            seasonal_factor=1.0,
            weekend_factor=0.8
        )
    
    def _calculate_seasonal_factor(self, df: pd.DataFrame) -> float:
        """Calculate seasonal adjustment factor"""
        if 'month' not in df.columns or len(df) == 0:
            return 1.0
        
        current_month = datetime.now().month
        
        # Simple seasonal model for Indonesia (dry/wet season)
        if current_month in [5, 6, 7, 8, 9]:  # Dry season (more sun)
            return 1.2
        else:  # Wet season (less sun)
            return 0.8
    
    def _calculate_weekend_factor(self, df: pd.DataFrame) -> float:
        """Calculate weekend usage factor"""
        if 'day_of_week' not in df.columns or len(df) == 0:
            return 0.8
        
        # Weekend (Saturday=6, Sunday=0) typically has different usage
        weekend_data = df[df['day_of_week'].isin([0, 6])]
        weekday_data = df[~df['day_of_week'].isin([0, 6])]
        
        if len(weekend_data) > 0 and len(weekday_data) > 0:
            weekend_avg = weekend_data['load_power'].mean()
            weekday_avg = weekday_data['load_power'].mean()
            return weekend_avg / weekday_avg if weekday_avg > 0 else 0.8
        
        return 0.8
    
    def _store_patterns(self, patterns: UsagePattern):
        """Store learned patterns to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Store as JSON
            pattern_data = {
                'peak_pv_hours': patterns.peak_pv_hours,
                'low_tariff_windows': patterns.low_tariff_windows,
                'typical_load_profile': patterns.typical_load_profile,
                'seasonal_factor': patterns.seasonal_factor,
                'weekend_factor': patterns.weekend_factor
            }
            
            cursor.execute('''
                INSERT OR REPLACE INTO learned_patterns 
                (pattern_type, pattern_data, confidence, last_updated)
                VALUES (?, ?, ?, ?)
            ''', (
                'usage_pattern',
                json.dumps(pattern_data),
                0.8,  # Confidence score
                time.time()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Pattern storage error: {e}")

class Layer3AdaptivePlanner:
    """Layer 3 Strategic Planner with adaptive learning"""
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.Layer3Adaptive")
        
        # Planning horizon
        self.planning_horizon_hours = 24  # 24-hour strategic planning
        self.update_interval_hours = 6    # Update every 6 hours
        
        # Components
        self.adaptive_weights = AdaptiveMPCWeights()
        self.pattern_recognition = PatternRecognition()
        
        # Strategic parameters
        self.strategic_targets = {
            'target_soc_morning': 0.8,     # Target SOC at morning
            'target_soc_evening': 0.6,    # Target SOC at evening
            'max_grid_hours_per_day': 6,  # Max hours of grid usage per day
            'min_pv_utilization': 0.8     # Minimum PV utilization target
        }
        
        # Database
        self.db_path = '/var/log/mpc_layer3_strategic.db'
        self.init_database()
        
        self.logger.info("Layer 3 Adaptive Planner initialized")
    
    def init_database(self):
        """Initialize strategic planning database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Strategic plans table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS strategic_plans (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    plan_horizon_hours INTEGER,
                    target_soc_profile TEXT,
                    charging_schedule TEXT,
                    switching_schedule TEXT,
                    predicted_cost REAL,
                    predicted_pv_utilization REAL
                )
            ''')
            
            # Weight adaptation history
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS weight_adaptations (
                    id INTEGER PRIMARY KEY,
                    timestamp REAL,
                    weights_efficiency REAL,
                    weights_cost REAL,
                    weights_battery_health REAL,
                    weights_comfort REAL,
                    performance_trigger TEXT,
                    adaptation_reason TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Strategic database initialization error: {e}")
    
    def generate_strategic_plan(self, current_performance: PerformanceMetrics) -> Dict[str, Any]:
        """Generate 24-hour strategic plan"""
        
        start_time = time.time()
        
        try:
            # 1. Analyze usage patterns
            patterns = self.pattern_recognition.analyze_usage_patterns()
            
            # 2. Update adaptive weights based on performance
            new_weights = self.adaptive_weights.update_weights(current_performance)
            
            # 3. Generate strategic SOC profile
            soc_profile = self._generate_soc_profile(patterns)
            
            # 4. Generate charging schedule
            charging_schedule = self._generate_charging_schedule(patterns, soc_profile)
            
            # 5. Generate switching schedule
            switching_schedule = self._generate_switching_schedule(patterns)
            
            # 6. Calculate predicted performance
            predicted_cost = self._predict_daily_cost(patterns, charging_schedule)
            predicted_pv_util = self._predict_pv_utilization(patterns)
            
            strategic_plan = {
                'timestamp': datetime.now().isoformat(),
                'planning_horizon_hours': self.planning_horizon_hours,
                'weights': new_weights,
                'soc_profile': soc_profile,
                'charging_schedule': charging_schedule,
                'switching_schedule': switching_schedule,
                'patterns': {
                    'peak_pv_hours': patterns.peak_pv_hours,
                    'low_tariff_windows': patterns.low_tariff_windows,
                    'seasonal_factor': patterns.seasonal_factor
                },
                'predictions': {
                    'daily_cost': predicted_cost,
                    'pv_utilization': predicted_pv_util
                },
                'targets': self.strategic_targets,
                'computation_time': time.time() - start_time
            }
            
            # Store strategic plan
            self._store_strategic_plan(strategic_plan)
            
            self.logger.info(f"Strategic plan generated in {strategic_plan['computation_time']:.2f}s")
            self.logger.info(f"Predicted cost: Rp{predicted_cost:.0f}, PV util: {predicted_pv_util:.1%}")
            
            return strategic_plan
            
        except Exception as e:
            self.logger.error(f"Strategic planning error: {e}")
            return self._get_fallback_plan()
    
    def _generate_soc_profile(self, patterns: UsagePattern) -> List[float]:
        """Generate target SOC profile for 24 hours"""
        soc_profile = []
        
        for hour in range(24):
            if hour in patterns.peak_pv_hours:
                # Charge during peak PV hours
                target_soc = min(0.9, 0.6 + (hour - min(patterns.peak_pv_hours)) * 0.05)
            elif 6 <= hour <= 8:
                # Morning target
                target_soc = self.strategic_targets['target_soc_morning']
            elif 17 <= hour <= 22:
                # Evening usage period
                target_soc = max(0.4, self.strategic_targets['target_soc_evening'] - (hour - 17) * 0.03)
            elif 23 <= hour or hour <= 5:
                # Night time - maintain
                target_soc = 0.5
            else:
                # Default
                target_soc = 0.6
            
            soc_profile.append(target_soc)
        
        return soc_profile
    
    def _generate_charging_schedule(self, patterns: UsagePattern, soc_profile: List[float]) -> List[Dict]:
        """Generate optimized charging schedule"""
        schedule = []
        
        for hour in range(24):
            if hour in patterns.peak_pv_hours:
                # Prioritize PV charging during peak hours
                schedule.append({
                    'hour': hour,
                    'source': 'pv',
                    'target_current': 15.0,  # Moderate charging
                    'priority': 'high'
                })
            elif hour in [w[0] for w in patterns.low_tariff_windows]:
                # Grid charging during low tariff
                schedule.append({
                    'hour': hour,
                    'source': 'grid',
                    'target_current': 10.0,  # Conservative grid charging
                    'priority': 'medium'
                })
            else:
                # Maintenance or no charging
                schedule.append({
                    'hour': hour,
                    'source': 'auto',
                    'target_current': 0.0,
                    'priority': 'low'
                })
        
        return schedule
    
    def _generate_switching_schedule(self, patterns: UsagePattern) -> List[Dict]:
        """Generate power source switching schedule"""
        schedule = []
        
        # Minimize switching while optimizing cost and efficiency
        current_source = 'pv'
        
        for hour in range(24):
            if hour in patterns.peak_pv_hours:
                preferred_source = 'pv'
            elif any(w[0] <= hour <= w[1] for w in patterns.low_tariff_windows):
                preferred_source = 'grid'
            else:
                preferred_source = 'auto'  # Let MPC decide
            
            # Only switch if necessary (reduce wear)
            if preferred_source != 'auto' and preferred_source != current_source:
                schedule.append({
                    'hour': hour,
                    'switch_to': preferred_source,
                    'reason': 'strategic_optimization'
                })
                current_source = preferred_source
        
        return schedule
    
    def _predict_daily_cost(self, patterns: UsagePattern, charging_schedule: List[Dict]) -> float:
        """Predict daily electricity cost"""
        total_cost = 0.0
        
        # Simplified cost calculation
        for hour_schedule in charging_schedule:
            hour = hour_schedule['hour']
            if hour_schedule['source'] == 'grid':
                # Estimate grid power consumption
                power_kwh = hour_schedule['target_current'] * 12.8 / 1000  # Rough estimate
                
                # Apply tariff
                if 17 <= hour <= 22:  # Peak hours
                    rate = 1467  # Rp/kWh
                else:
                    rate = 1035  # Rp/kWh
                
                total_cost += power_kwh * rate
        
        return total_cost
    
    def _predict_pv_utilization(self, patterns: UsagePattern) -> float:
        """Predict PV utilization based on patterns"""
        # Simplified calculation based on peak PV hours utilization
        peak_hours = len(patterns.peak_pv_hours)
        total_daylight_hours = 12  # Assume 12 hours of daylight
        
        # Seasonal adjustment
        base_utilization = (peak_hours / total_daylight_hours) * 0.8
        adjusted_utilization = base_utilization * patterns.seasonal_factor
        
        return min(1.0, adjusted_utilization)
    
    def _store_strategic_plan(self, plan: Dict[str, Any]):
        """Store strategic plan to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO strategic_plans
                (timestamp, plan_horizon_hours, target_soc_profile, charging_schedule,
                 switching_schedule, predicted_cost, predicted_pv_utilization)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                time.time(),
                plan['planning_horizon_hours'],
                json.dumps(plan['soc_profile']),
                json.dumps(plan['charging_schedule']),
                json.dumps(plan['switching_schedule']),
                plan['predictions']['daily_cost'],
                plan['predictions']['pv_utilization']
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Strategic plan storage error: {e}")
    
    def _get_fallback_plan(self) -> Dict[str, Any]:
        """Fallback plan when strategic planning fails"""
        return {
            'timestamp': datetime.now().isoformat(),
            'planning_horizon_hours': 24,
            'weights': {
                'efficiency': 1.0,
                'cost': 0.5,
                'battery_health': 0.8,
                'comfort': 0.3
            },
            'soc_profile': [0.6] * 24,  # Maintain 60% SOC
            'charging_schedule': [],
            'switching_schedule': [],
            'predictions': {
                'daily_cost': 0.0,
                'pv_utilization': 0.5
            },
            'computation_time': 0.0,
            'fallback': True
        }

def main():
    """Main function for headless Layer 3 operation"""
    logger.info("🚀 Starting Layer 3 Adaptive Planner - Headless Mode")
    
    try:
        # Initialize Layer 3 planner
        adaptive_planner = Layer3AdaptivePlanner()
        
        # Main strategic planning loop
        while True:
            # Create dummy performance metrics for testing
            current_performance = PerformanceMetrics(
                pv_utilization=0.75,
                grid_dependency=0.45,
                soc_stability=0.1,
                efficiency_overall=0.85,
                cost_per_kwh=1200.0,
                switching_frequency=8.0,
                comfort_score=0.8,
                battery_health_score=0.9
            )
            
            # Generate strategic plan
            strategic_plan = adaptive_planner.generate_strategic_plan(current_performance)
            
            # Log strategic decisions
            logger.info(f"Strategic plan completed:")
            logger.info(f"  - Weights: {strategic_plan['weights']}")
            logger.info(f"  - Predicted cost: Rp{strategic_plan['predictions']['daily_cost']:.0f}")
            logger.info(f"  - Predicted PV utilization: {strategic_plan['predictions']['pv_utilization']:.1%}")
            
            # Sleep for update interval (Layer 3 runs every 6 hours)
            time.sleep(6 * 3600)  # 6 hours
            
    except KeyboardInterrupt:
        logger.info("Layer 3 Adaptive Planner stopped by user")
    except Exception as e:
        logger.error(f"Layer 3 error: {e}")
        raise

if __name__ == "__main__":
    main()
