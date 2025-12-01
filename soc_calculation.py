#!/usr/bin/env python3
"""
SOC CALCULATION MODULE
====================

Advanced SOC estimation using coulomb counting with temperature compensation
and voltage-based validation for accurate battery state estimation.

Author: Dzaky Naufal K
Date: July 2, 2025
Version: 1.0 - MPC Integration Ready
"""

import numpy as np
import time
import json
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class SOCState:
    """SOC estimation state"""
    soc: float
    voltage: float
    current: float
    temperature: float
    timestamp: float
    coulomb_count: float
    confidence: float

class KalmanFilterSOC:
    """
    Kalman Filter for SOC estimation
    State vector: [SOC, current_bias]
    """
    
    def __init__(self, initial_soc: float = 50.0, capacity_ah: float = 35.0):
        """
        Initialize Kalman Filter for SOC
        
        Args:
            initial_soc: Initial SOC estimate (0-100)
            capacity_ah: Battery capacity in Ah
        """
        self.capacity_ah = capacity_ah
        
        # State vector [SOC, bias]
        self.x = np.array([[initial_soc], [0.0]])  # [SOC%, current_bias_A]
        
        # Covariance matrix
        self.P = np.eye(2) * 100  # Initial uncertainty
        
        # Process noise covariance
        self.Q = np.diag([0.01, 0.001])  # [SOC_noise, bias_noise]
        
        # Measurement noise covariance
        self.R = np.array([[1.0]])  # Voltage-based SOC measurement noise
        
        # Last update time
        self.last_time = time.time()
        
        logger.info(f"Kalman Filter SOC initialized: SOC={initial_soc:.1f}%, Capacity={capacity_ah}Ah")
    
    def predict(self, current_A: float, dt: float):
        """
        Kalman Filter prediction step
        
        Args:
            current_A: Battery current in Amperes (positive = charging)
            dt: Time step in seconds
        """
        try:
            # State transition matrix
            F = np.array([[1, -dt/3600], [0, 1]])  # dt/3600 converts seconds to hours
            
            # Control input matrix
            B = np.array([[dt/3600/self.capacity_ah], [0]])  # Current to SOC conversion
            
            # Predict state
            self.x = F @ self.x + B * current_A
            
            # Predict covariance
            self.P = F @ self.P @ F.T + self.Q
            
            # Constrain SOC to valid range
            self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 100.0)
            
        except Exception as e:
            logger.error(f"Kalman predict error: {e}")
    
    def update(self, soc_voltage: float):
        """
        Kalman Filter update step with voltage-based SOC measurement
        
        Args:
            soc_voltage: SOC estimate from voltage lookup table
        """
        try:
            # Measurement matrix
            H = np.array([[1, 0]])  # We measure SOC directly
            
            # Innovation (measurement residual)
            y = soc_voltage - H @ self.x
            
            # Innovation covariance
            S = H @ self.P @ H.T + self.R
            
            # Kalman gain
            K = self.P @ H.T @ np.linalg.inv(S)
            
            # Update state
            self.x = self.x + K * y
            
            # Update covariance
            I = np.eye(2)
            self.P = (I - K @ H) @ self.P
            
            # Constrain SOC to valid range
            self.x[0, 0] = np.clip(self.x[0, 0], 0.0, 100.0)
            
        except Exception as e:
            logger.error(f"Kalman update error: {e}")
    
    def get_soc(self) -> float:
        """Get current SOC estimate"""
        return float(self.x[0, 0])
    
    def get_bias(self) -> float:
        """Get current bias estimate"""
        return float(self.x[1, 0])
    
    def get_uncertainty(self) -> float:
        """Get SOC uncertainty (standard deviation)"""
        return float(np.sqrt(self.P[0, 0]))
    
    def step(self, current_A: float, voltage_soc: float) -> float:
        """
        Complete Kalman filter step (predict + update)
        
        Args:
            current_A: Battery current in Amperes
            voltage_soc: SOC estimate from voltage
            
        Returns:
            Updated SOC estimate
        """
        current_time = time.time()
        dt = current_time - self.last_time
        self.last_time = current_time
        
        # Prediction step
        self.predict(current_A, dt)
        
        # Update step with voltage-based SOC
        self.update(voltage_soc)
        
        return self.get_soc()

class SOCEstimator:
    """
    Advanced SOC estimator using multiple methods:
    1. Coulomb counting (primary method)
    2. Voltage-based lookup (validation)
    3. Temperature compensation
    4. Kalman filter fusion (optional)
    """
    
    def __init__(self, battery_capacity_ah: float = 35.0, initial_soc: float = 75.0, use_kalman: bool = True):
        """
        Initialize SOC estimator
        
        Args:
            battery_capacity_ah: Battery capacity in Ah
            initial_soc: Initial SOC percentage (0-100)
            use_kalman: Enable Kalman filter for SOC fusion
        """
        self.capacity_ah = battery_capacity_ah
        self.capacity_as = battery_capacity_ah * 3600  # Convert to Ampere-seconds
        
        # SOC state
        self.current_soc = initial_soc
        self.coulomb_counter = 0.0
        self.last_timestamp = time.time()
        
        # Kalman filter integration
        self.use_kalman = use_kalman
        if self.use_kalman:
            self.kalman_filter = KalmanFilterSOC(initial_soc, battery_capacity_ah)
            logger.info("SOC Estimator initialized with Kalman Filter")
        else:
            self.kalman_filter = None
            logger.info("SOC Estimator initialized without Kalman Filter")
        
        # Efficiency parameters
        self.efficiency_charge = 0.95   # Charging efficiency
        self.efficiency_discharge = 0.98  # Discharging efficiency
        
        # Temperature compensation
        self.temp_coefficient = 0.005   # %/°C
        self.reference_temp = 25.0      # Reference temperature
        
        # Voltage-SOC lookup table (12V LiFePO4 4P6S)
        self.voltage_soc_table = {
            # Voltage: SOC (LFP 4P6S 12V system - charging 14.6V, min 12.6V, max 13.5V)
            10.0: 0,    # Empty (emergency cutoff)
            11.5: 5,    # Very low
            12.0: 10,   # Low
            12.6: 20,   # Minimum operational
            12.8: 30,   # Low normal
            13.0: 50,   # Mid range
            13.2: 70,   # Good
            13.4: 85,   # High
            13.5: 95,   # Maximum operational
            14.6: 100   # Full charge voltage
        }
        
        # Data buffers
        self.soc_history = deque(maxlen=1000)
        self.voltage_history = deque(maxlen=100)
        self.current_history = deque(maxlen=100)
        
        # Calibration parameters
        self.last_calibration = time.time()
        self.calibration_interval = 24 * 3600  # 24 hours
        
        # Error tracking
        self.estimation_error = 0.0
        self.confidence_level = 1.0
        
        logger.info(f"SOC Estimator initialized: Capacity={battery_capacity_ah}Ah, Initial SOC={initial_soc}%")
    
    def calculate_soc(self, current_a: float, voltage_v: float, 
                     temperature_c: float, dt_seconds: float = None) -> SOCState:
        """
        Calculate SOC using coulomb counting with temperature compensation
        
        Args:
            current_a: Battery current in Amperes (positive = charging)
            voltage_v: Battery voltage in Volts
            temperature_c: Battery temperature in Celsius
            dt_seconds: Time delta in seconds (auto-calculated if None)
            
        Returns:
            SOCState with updated values
        """
        current_time = time.time()
        
        # Calculate time delta
        if dt_seconds is None:
            dt_seconds = current_time - self.last_timestamp
        
        # Prevent invalid time deltas
        if dt_seconds <= 0 or dt_seconds > 3600:  # Max 1 hour
            dt_seconds = 1.0
        
        # Temperature compensation factor
        temp_factor = 1 + self.temp_coefficient * (temperature_c - self.reference_temp)
        temp_factor = max(0.8, min(1.2, temp_factor))  # Limit to reasonable range
        
        # Efficiency based on current direction
        if current_a > 0:  # Charging
            efficiency = self.efficiency_charge * temp_factor
        else:  # Discharging
            efficiency = self.efficiency_discharge * temp_factor
        
        # Coulomb counting
        charge_delta_as = current_a * dt_seconds * efficiency
        self.coulomb_counter += charge_delta_as
        
        # Calculate SOC from coulomb counting
        soc_coulomb = self.current_soc + (charge_delta_as / self.capacity_as) * 100
        
        # Voltage-based SOC estimation (for validation)
        soc_voltage = self._voltage_to_soc(voltage_v)
        
        # Fusion of coulomb counting and voltage estimation
        if self.use_kalman and self.kalman_filter:
            # Use Kalman filter for optimal fusion
            soc_fused = self.kalman_filter.step(current_a, soc_voltage)
            
            # Log Kalman filter state for debugging
            logger.debug(f"Kalman SOC: {soc_fused:.1f}%, Bias: {self.kalman_filter.get_bias():.3f}A, "
                        f"Uncertainty: {self.kalman_filter.get_uncertainty():.2f}%")
        else:
            # Use simple weighted fusion
            soc_fused = self._fuse_soc_estimates(soc_coulomb, soc_voltage, current_a)
        
        # Apply bounds
        soc_fused = max(0.0, min(100.0, soc_fused))
        
        # Update state
        self.current_soc = soc_fused
        self.last_timestamp = current_time
        
        # Calculate confidence level
        confidence = self._calculate_confidence(current_a, voltage_v, temperature_c)
        
        # Create SOC state
        soc_state = SOCState(
            soc=soc_fused,
            voltage=voltage_v,
            current=current_a,
            temperature=temperature_c,
            timestamp=current_time,
            coulomb_count=self.coulomb_counter,
            confidence=confidence
        )
        
        # Store in history
        self.soc_history.append(soc_state)
        self.voltage_history.append(voltage_v)
        self.current_history.append(current_a)
        
        # Periodic recalibration
        if current_time - self.last_calibration > self.calibration_interval:
            self._recalibrate()
        
        return soc_state
    
    def _voltage_to_soc(self, voltage: float) -> float:
        """Convert voltage to SOC using lookup table with interpolation"""
        voltages = sorted(self.voltage_soc_table.keys())
        
        if voltage <= voltages[0]:
            return self.voltage_soc_table[voltages[0]]
        if voltage >= voltages[-1]:
            return self.voltage_soc_table[voltages[-1]]
        
        # Linear interpolation
        for i in range(len(voltages) - 1):
            v1, v2 = voltages[i], voltages[i + 1]
            if v1 <= voltage <= v2:
                soc1 = self.voltage_soc_table[v1]
                soc2 = self.voltage_soc_table[v2]
                # Interpolate
                ratio = (voltage - v1) / (v2 - v1)
                return soc1 + ratio * (soc2 - soc1)
        
        return 50.0  # Default fallback
    
    def _fuse_soc_estimates(self, soc_coulomb: float, soc_voltage: float, current_a: float) -> float:
        """
        Fuse coulomb counting and voltage-based SOC estimates
        
        Priority:
        - Use coulomb counting during active charge/discharge
        - Use voltage estimation during rest periods
        - Weighted fusion during transition
        """
        current_abs = abs(current_a)
        
        if current_abs > 2.0:  # High current - trust coulomb counting
            weight_coulomb = 0.9
            weight_voltage = 0.1
        elif current_abs < 0.1:  # Rest period - trust voltage more
            weight_coulomb = 0.3
            weight_voltage = 0.7
        else:  # Transition - balanced fusion
            weight_coulomb = 0.6
            weight_voltage = 0.4
        
        # Weighted fusion
        soc_fused = weight_coulomb * soc_coulomb + weight_voltage * soc_voltage
        
        # Drift correction - if voltage and coulomb differ significantly
        error = abs(soc_coulomb - soc_voltage)
        if error > 10.0:  # More than 10% difference
            # Gradually correct coulomb counting towards voltage
            correction_factor = min(0.1, error / 100.0)
            soc_fused = soc_coulomb + correction_factor * (soc_voltage - soc_coulomb)
            
        return soc_fused
    
    def _calculate_confidence(self, current_a: float, voltage_v: float, temperature_c: float) -> float:
        """Calculate confidence level in SOC estimation"""
        confidence = 1.0
        
        # Reduce confidence for extreme conditions
        if temperature_c < 0 or temperature_c > 50:
            confidence *= 0.8
        
        if voltage_v < 44.0 or voltage_v > 56.0:
            confidence *= 0.7
        
        if abs(current_a) > 30.0:
            confidence *= 0.9
        
        # Check estimation consistency
        if len(self.soc_history) > 10:
            recent_socs = [state.soc for state in list(self.soc_history)[-10:]]
            soc_std = np.std(recent_socs)
            if soc_std > 5.0:  # High variation
                confidence *= 0.8
        
        return max(0.1, min(1.0, confidence))
    
    def _recalibrate(self):
        """Periodic recalibration using voltage estimation"""
        if len(self.voltage_history) > 50:
            # Use recent voltage readings during low current periods
            recent_data = list(zip(self.voltage_history, self.current_history))
            
            # Find rest periods (low current)
            rest_voltages = [v for v, i in recent_data[-50:] if abs(i) < 0.5]
            
            if len(rest_voltages) > 5:
                avg_rest_voltage = np.mean(rest_voltages)
                voltage_soc = self._voltage_to_soc(avg_rest_voltage)
                
                # Adjust coulomb counter if significant drift
                soc_error = voltage_soc - self.current_soc
                if abs(soc_error) > 5.0:
                    # Gradual correction
                    correction = soc_error * 0.1  # 10% correction per calibration
                    self.current_soc += correction
                    
                    # Adjust coulomb counter accordingly
                    self.coulomb_counter += (correction / 100.0) * self.capacity_as
                    
                    logger.info(f"SOC recalibrated: {soc_error:.1f}% error corrected")
        
        self.last_calibration = time.time()
    
    def reset_soc(self, new_soc: float):
        """Reset SOC to known value (e.g., after full charge)"""
        self.current_soc = max(0.0, min(100.0, new_soc))
        self.coulomb_counter = (self.current_soc / 100.0) * self.capacity_as
        logger.info(f"SOC reset to {new_soc}%")
    
    def get_soc_statistics(self) -> Dict:
        """Get SOC estimation statistics"""
        if len(self.soc_history) == 0:
            return {}
        
        recent_socs = [state.soc for state in self.soc_history]
        recent_confidences = [state.confidence for state in self.soc_history]
        
        return {
            'current_soc': self.current_soc,
            'soc_std_dev': float(np.std(recent_socs)),
            'avg_confidence': float(np.mean(recent_confidences)),
            'coulomb_counter': self.coulomb_counter,
            'capacity_used': (self.coulomb_counter / self.capacity_as) * 100,
            'data_points': len(self.soc_history),
            'last_calibration_age': time.time() - self.last_calibration
        }
    
    def export_soc_data(self, filename: str):
        """Export SOC history to JSON file"""
        data = {
            'metadata': {
                'capacity_ah': self.capacity_ah,
                'efficiency_charge': self.efficiency_charge,
                'efficiency_discharge': self.efficiency_discharge,
                'export_timestamp': time.time()
            },
            'history': [
                {
                    'timestamp': state.timestamp,
                    'soc': state.soc,
                    'voltage': state.voltage,
                    'current': state.current,
                    'temperature': state.temperature,
                    'coulomb_count': state.coulomb_count,
                    'confidence': state.confidence
                }
                for state in self.soc_history
            ],
            'statistics': self.get_soc_statistics()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"SOC data exported to {filename}")

class SOCValidator:
    """SOC estimation validator for testing accuracy"""
    
    def __init__(self):
        self.reference_soc = None
        self.estimator_soc = None
        self.errors = []
    
    def add_comparison(self, reference_soc: float, estimated_soc: float):
        """Add SOC comparison point"""
        error = abs(reference_soc - estimated_soc)
        self.errors.append(error)
        
        return {
            'reference': reference_soc,
            'estimated': estimated_soc,
            'error': error,
            'error_percentage': (error / reference_soc) * 100 if reference_soc > 0 else 0
        }
    
    def get_accuracy_stats(self) -> Dict:
        """Get accuracy statistics"""
        if not self.errors:
            return {}
        
        return {
            'mean_error': float(np.mean(self.errors)),
            'std_error': float(np.std(self.errors)),
            'max_error': float(np.max(self.errors)),
            'min_error': float(np.min(self.errors)),
            'rmse': float(np.sqrt(np.mean(np.square(self.errors)))),
            'accuracy_5pct': sum(1 for e in self.errors if e <= 5.0) / len(self.errors) * 100
        }

def test_soc_estimation():
    """Test function for SOC estimation"""
    print("🔋 SOC ESTIMATION TEST")
    print("=" * 40)
    
    # Create SOC estimator
    soc_estimator = SOCEstimator(battery_capacity_ah=35.0, initial_soc=50.0)
    
    # Simulate charging scenario
    print("\n📈 Simulating charging scenario...")
    
    for i in range(60):  # 1 hour simulation
        # Simulate charging current (10A)
        current = 10.0
        voltage = 12.8 + (i / 60.0) * 0.5  # Voltage rises during charging (12V system)
        temperature = 25.0 + (i / 60.0) * 5.0  # Temperature rises slightly
        
        soc_state = soc_estimator.calculate_soc(current, voltage, temperature, 60.0)
        
        if i % 10 == 0:  # Print every 10 minutes
            print(f"Time: {i:2d}min | SOC: {soc_state.soc:5.1f}% | "
                  f"V: {voltage:4.1f}V | I: {current:4.1f}A | "
                  f"Conf: {soc_state.confidence:.2f}")
    
    # Print final statistics
    stats = soc_estimator.get_soc_statistics()
    print(f"\n📊 Final Statistics:")
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print(f"\n✅ SOC estimation test completed!")

if __name__ == "__main__":
    test_soc_estimation()
