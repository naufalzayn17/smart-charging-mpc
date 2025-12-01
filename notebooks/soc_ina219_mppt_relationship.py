#!/usr/bin/env python3
"""
PENJELASAN HUBUNGAN SOC, INA219, dan MPPT SCC
=============================================

Menjelaskan bagaimana perhitungan SOC, pembacaan sensor INA219, dan fungsi MPPT SCC
saling terkait dalam sistem smart charging.

Author: Dzaky Naufal K
Date: July 7, 2025
"""

import numpy as np
import time
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class BatteryState:
    """State baterai yang diukur dan dihitung"""
    # Data dari sensor INA219
    voltage: float          # Tegangan baterai (V) - dari INA219
    current: float          # Arus baterai (A) - dari INA219  
    power: float           # Daya baterai (W) - dari INA219
    
    # Data yang dihitung
    soc: float             # State of Charge (%) - DIHITUNG dari data INA219
    capacity_remaining: float  # Kapasitas tersisa (Ah) - DIHITUNG
    
    # Data dari sensor lain
    temperature: float     # Suhu baterai (°C) - dari DS18B20
    timestamp: float

@dataclass
class MPPTState:
    """State MPPT Solar Charge Controller"""
    # Input dari panel surya
    pv_voltage: float      # Tegangan PV (V) - dari INA219 PV
    pv_current: float      # Arus PV (A) - dari INA219 PV
    pv_power: float        # Daya PV (W) - dari INA219 PV
    
    # Output ke baterai
    output_voltage: float  # Tegangan output MPPT (V)
    output_current: float  # Arus output MPPT (A)
    output_power: float    # Daya output MPPT (W)
    
    # Status MPPT
    mppt_mode: str         # "bulk", "absorption", "float", "equalization"
    mppt_efficiency: float # Efisiensi MPPT (%)
    mppt_temperature: float # Suhu MPPT (°C)

class SOCCalculator:
    """
    Kelas untuk menghitung SOC berdasarkan data sensor INA219
    
    SOC TIDAK LANGSUNG DIUKUR oleh INA219, tetapi DIHITUNG berdasarkan:
    1. Tegangan baterai (voltage-based SOC)
    2. Coulomb counting (current integration)
    3. Kombinasi keduanya (hybrid method)
    """
    
    def __init__(self, battery_capacity_ah: float = 35.0):
        self.battery_capacity_ah = battery_capacity_ah
        self.initial_soc = 50.0  # SOC awal (%)
        self.accumulated_charge = 0.0  # Akumulasi charge (Ah)
        self.last_update_time = time.time()
        
        # Lookup table untuk voltage-based SOC (LiFePO4)
        self.voltage_soc_table = {
            # Voltage: SOC%
            14.6: 100.0,
            14.4: 99.0,
            14.2: 95.0,
            13.6: 90.0,
            13.4: 85.0,
            13.2: 80.0,
            13.0: 70.0,
            12.8: 60.0,
            12.6: 50.0,
            12.4: 40.0,
            12.2: 30.0,
            12.0: 20.0,
            11.8: 10.0,
            11.6: 5.0,
            11.0: 0.0
        }
    
    def calculate_soc_from_voltage(self, voltage: float) -> float:
        """
        Menghitung SOC berdasarkan tegangan baterai
        
        CATATAN: Ini adalah perkiraan kasar, akurasi rendah saat baterai sedang di-charge/discharge
        """
        voltages = sorted(self.voltage_soc_table.keys())
        
        if voltage >= voltages[-1]:
            return 100.0
        elif voltage <= voltages[0]:
            return 0.0
        
        # Linear interpolation
        for i in range(len(voltages) - 1):
            if voltages[i] <= voltage <= voltages[i + 1]:
                v1, v2 = voltages[i], voltages[i + 1]
                soc1, soc2 = self.voltage_soc_table[v1], self.voltage_soc_table[v2]
                
                # Linear interpolation
                soc = soc1 + (voltage - v1) * (soc2 - soc1) / (v2 - v1)
                return soc
        
        return 50.0  # Default
    
    def calculate_soc_from_coulomb_counting(self, current: float) -> float:
        """
        Menghitung SOC berdasarkan coulomb counting (integrasi arus)
        
        Ini adalah metode yang lebih akurat untuk tracking perubahan SOC
        """
        current_time = time.time()
        dt = current_time - self.last_update_time  # dalam detik
        
        # Konversi ke hours
        dt_hours = dt / 3600.0
        
        # Akumulasi charge/discharge
        # Positive current = charging, Negative current = discharging
        charge_change = current * dt_hours  # Ah
        
        # Efisiensi charging/discharging
        if current > 0:  # Charging
            efficiency = 0.95  # 95% efisiensi charging
        else:  # Discharging
            efficiency = 0.98  # 98% efisiensi discharging
        
        self.accumulated_charge += charge_change * efficiency
        
        # Hitung SOC berdasarkan perubahan charge
        soc_change = (charge_change * efficiency / self.battery_capacity_ah) * 100.0
        
        # Update SOC
        new_soc = self.initial_soc + (self.accumulated_charge / self.battery_capacity_ah) * 100.0
        
        # Clamp SOC antara 0-100%
        new_soc = max(0.0, min(100.0, new_soc))
        
        self.last_update_time = current_time
        
        return new_soc
    
    def calculate_soc_hybrid(self, voltage: float, current: float) -> float:
        """
        Menghitung SOC dengan metode hybrid (kombinasi voltage + coulomb counting)
        
        Metode ini paling akurat karena menggabungkan kedua metode
        """
        # Hitung SOC dari kedua metode
        soc_voltage = self.calculate_soc_from_voltage(voltage)
        soc_coulomb = self.calculate_soc_from_coulomb_counting(current)
        
        # Jika baterai dalam kondisi rest (tidak ada arus), gunakan voltage-based
        if abs(current) < 0.1:  # Arus sangat kecil (rest condition)
            # Voltage-based lebih akurat saat rest
            return soc_voltage
        else:
            # Saat ada arus, gunakan weighted average
            # Coulomb counting lebih akurat untuk perubahan SOC
            # Voltage-based sebagai referensi
            weight_coulomb = 0.8
            weight_voltage = 0.2
            
            hybrid_soc = (weight_coulomb * soc_coulomb) + (weight_voltage * soc_voltage)
            return hybrid_soc

class INA219_MPPTIntegration:
    """
    Kelas untuk mengintegrasikan pembacaan INA219 dengan control MPPT SCC
    
    HUBUNGAN PENTING:
    1. INA219 mengukur V, I, P dari baterai dan PV
    2. SOC dihitung dari data INA219
    3. MPPT dikontrol berdasarkan SOC dan kondisi PV
    4. Priority system menggunakan SOC untuk decision making
    """
    
    def __init__(self):
        self.soc_calculator = SOCCalculator()
        self.mppt_control_active = False
        
        # MPPT control parameters
        self.mppt_bulk_voltage = 14.4    # Bulk charging voltage
        self.mppt_absorption_voltage = 14.2  # Absorption voltage
        self.mppt_float_voltage = 13.6   # Float voltage
        
        # SOC thresholds for MPPT control
        self.soc_bulk_threshold = 80.0   # Switch to absorption
        self.soc_absorption_threshold = 90.0  # Switch to float
        self.soc_float_threshold = 95.0  # Stop charging
    
    def process_sensor_data(self, ina219_data: Dict) -> Dict:
        """
        Process data dari triple INA219 sensors
        
        Args:
            ina219_data: {
                'battery': {'voltage': V, 'current': A, 'power': W},
                'pv': {'voltage': V, 'current': A, 'power': W},
                'grid': {'voltage': V, 'current': A, 'power': W}
            }
        
        Returns:
            Processed data dengan SOC dan MPPT control
        """
        # Extract data dari INA219
        battery_voltage = ina219_data['battery']['voltage']
        battery_current = ina219_data['battery']['current']
        battery_power = ina219_data['battery']['power']
        
        pv_voltage = ina219_data['pv']['voltage']
        pv_current = ina219_data['pv']['current']
        pv_power = ina219_data['pv']['power']
        
        # Hitung SOC berdasarkan data INA219
        soc = self.soc_calculator.calculate_soc_hybrid(battery_voltage, battery_current)
        
        # Tentukan MPPT control mode berdasarkan SOC
        mppt_mode = self.determine_mppt_mode(soc, battery_voltage, pv_power)
        
        # Generate MPPT control signals
        mppt_control = self.generate_mppt_control(soc, battery_voltage, pv_power, mppt_mode)
        
        return {
            'battery_state': BatteryState(
                voltage=battery_voltage,
                current=battery_current,
                power=battery_power,
                soc=soc,
                capacity_remaining=soc * self.soc_calculator.battery_capacity_ah / 100.0,
                temperature=25.0,  # Dari DS18B20
                timestamp=time.time()
            ),
            'mppt_state': MPPTState(
                pv_voltage=pv_voltage,
                pv_current=pv_current,
                pv_power=pv_power,
                output_voltage=mppt_control['output_voltage'],
                output_current=mppt_control['output_current'],
                output_power=mppt_control['output_power'],
                mppt_mode=mppt_mode,
                mppt_efficiency=mppt_control['efficiency'],
                mppt_temperature=35.0
            ),
            'mppt_control_signals': mppt_control
        }
    
    def determine_mppt_mode(self, soc: float, battery_voltage: float, pv_power: float) -> str:
        """
        Tentukan mode MPPT berdasarkan SOC dan kondisi baterai
        
        MPPT Charging Stages:
        1. Bulk: SOC < 80%, charge dengan maksimum current
        2. Absorption: SOC 80-90%, maintain constant voltage
        3. Float: SOC 90-95%, maintain float voltage
        4. Stop: SOC > 95%, stop charging
        """
        if pv_power < 10.0:  # PV power terlalu rendah
            return "standby"
        
        if soc < self.soc_bulk_threshold:
            return "bulk"
        elif soc < self.soc_absorption_threshold:
            return "absorption"
        elif soc < self.soc_float_threshold:
            return "float"
        else:
            return "stop"
    
    def generate_mppt_control(self, soc: float, battery_voltage: float, 
                             pv_power: float, mppt_mode: str) -> Dict:
        """
        Generate control signals untuk MPPT SCC
        
        CATATAN: Dalam implementasi nyata, ini akan dikirim ke MPPT controller
        via komunikasi digital (RS485, CAN, atau analog control)
        """
        control_signals = {
            'mode': mppt_mode,
            'voltage_setpoint': 13.6,  # Default float voltage
            'current_limit': 30.0,     # Default max current
            'power_limit': 500.0,      # Default max power
            'output_voltage': battery_voltage,
            'output_current': 0.0,
            'output_power': 0.0,
            'efficiency': 0.95,
            'enable': True
        }
        
        if mppt_mode == "bulk":
            control_signals['voltage_setpoint'] = self.mppt_bulk_voltage
            control_signals['current_limit'] = 30.0  # Maximum charging current
            control_signals['output_current'] = min(30.0, pv_power / battery_voltage)
            
        elif mppt_mode == "absorption":
            control_signals['voltage_setpoint'] = self.mppt_absorption_voltage
            control_signals['current_limit'] = 15.0  # Reduced current
            control_signals['output_current'] = min(15.0, pv_power / battery_voltage)
            
        elif mppt_mode == "float":
            control_signals['voltage_setpoint'] = self.mppt_float_voltage
            control_signals['current_limit'] = 5.0   # Minimal current
            control_signals['output_current'] = min(5.0, pv_power / battery_voltage)
            
        elif mppt_mode == "stop":
            control_signals['voltage_setpoint'] = battery_voltage
            control_signals['current_limit'] = 0.0
            control_signals['output_current'] = 0.0
            control_signals['enable'] = False
            
        elif mppt_mode == "standby":
            control_signals['enable'] = False
            control_signals['output_current'] = 0.0
        
        # Hitung output power dan efficiency
        control_signals['output_power'] = control_signals['output_voltage'] * control_signals['output_current']
        
        if pv_power > 0:
            control_signals['efficiency'] = control_signals['output_power'] / pv_power
        else:
            control_signals['efficiency'] = 0.0
        
        return control_signals

def demonstrate_soc_ina219_mppt_relationship():
    """
    Demonstrasi hubungan SOC, INA219, dan MPPT SCC
    """
    print("🔋 HUBUNGAN SOC, INA219, dan MPPT SCC")
    print("=" * 60)
    
    # Inisialisasi integration
    integration = INA219_MPPTIntegration()
    
    # Simulasi data dari Triple INA219
    test_scenarios = [
        {
            "name": "🌅 Pagi - SOC Rendah, PV Mulai",
            "ina219_data": {
                'battery': {'voltage': 12.4, 'current': -2.0, 'power': -24.8},
                'pv': {'voltage': 18.0, 'current': 1.0, 'power': 18.0},
                'grid': {'voltage': 220.0, 'current': 0.0, 'power': 0.0}
            }
        },
        {
            "name": "☀️ Siang - SOC Sedang, PV Tinggi",
            "ina219_data": {
                'battery': {'voltage': 13.2, 'current': 15.0, 'power': 198.0},
                'pv': {'voltage': 22.0, 'current': 10.0, 'power': 220.0},
                'grid': {'voltage': 220.0, 'current': 0.0, 'power': 0.0}
            }
        },
        {
            "name": "🌞 Siang - SOC Tinggi, PV Maksimal",
            "ina219_data": {
                'battery': {'voltage': 13.6, 'current': 8.0, 'power': 108.8},
                'pv': {'voltage': 24.0, 'current': 12.0, 'power': 288.0},
                'grid': {'voltage': 220.0, 'current': 0.0, 'power': 0.0}
            }
        },
        {
            "name": "🌄 Sore - SOC Penuh, PV Menurun",
            "ina219_data": {
                'battery': {'voltage': 13.8, 'current': 2.0, 'power': 27.6},
                'pv': {'voltage': 20.0, 'current': 3.0, 'power': 60.0},
                'grid': {'voltage': 220.0, 'current': 0.0, 'power': 0.0}
            }
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\n{scenario['name']}")
        print("-" * 50)
        
        # Process sensor data
        result = integration.process_sensor_data(scenario['ina219_data'])
        
        battery_state = result['battery_state']
        mppt_state = result['mppt_state']
        mppt_control = result['mppt_control_signals']
        
        print("📊 DATA SENSOR INA219:")
        print(f"   🔋 Battery: {battery_state.voltage:.1f}V, {battery_state.current:.1f}A, {battery_state.power:.1f}W")
        print(f"   ☀️ PV: {mppt_state.pv_voltage:.1f}V, {mppt_state.pv_current:.1f}A, {mppt_state.pv_power:.1f}W")
        
        print("\n🧮 PERHITUNGAN SOC:")
        print(f"   📊 SOC: {battery_state.soc:.1f}%")
        print(f"   🔋 Kapasitas tersisa: {battery_state.capacity_remaining:.1f}Ah")
        
        print("\n⚙️ KONTROL MPPT:")
        print(f"   🎯 Mode: {mppt_control['mode']}")
        print(f"   📐 Voltage setpoint: {mppt_control['voltage_setpoint']:.1f}V")
        print(f"   ⚡ Current limit: {mppt_control['current_limit']:.1f}A")
        print(f"   🔌 Output: {mppt_control['output_voltage']:.1f}V, {mppt_control['output_current']:.1f}A, {mppt_control['output_power']:.1f}W")
        print(f"   📈 Efficiency: {mppt_control['efficiency']:.1%}")
        print(f"   🟢 Enable: {mppt_control['enable']}")
    
    print("\n" + "=" * 60)
    print("📋 KESIMPULAN HUBUNGAN:")
    print("1. 🔌 INA219 mengukur V, I, P dari baterai dan PV")
    print("2. 🧮 SOC DIHITUNG dari data INA219 (bukan diukur langsung)")
    print("3. ⚙️ MPPT dikontrol berdasarkan SOC dan kondisi PV")
    print("4. 🎯 Priority system menggunakan SOC untuk pengambilan keputusan")
    print("5. 🔄 Sistem terintegrasi dalam closed-loop control")

if __name__ == "__main__":
    demonstrate_soc_ina219_mppt_relationship()
