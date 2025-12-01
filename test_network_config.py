#!/usr/bin/env python3
"""
Network Configuration Validation Test
====================================
Test script untuk memvalidasi sinkronisasi konfigurasi jaringan
di seluruh sistem MPC Smart Charging.
"""

import sys
import os
import importlib

def test_network_config():
    """Test network configuration module"""
    print("Testing network configuration module...")
    
    try:
        from config.network_config import get_mqtt_config, get_wifi_config, get_device_config, validate_config
        print("✅ Network config module imported successfully")
        
        # Test configuration loading
        mqtt_config = get_mqtt_config()
        wifi_config = get_wifi_config()
        device_config = get_device_config()
        
        print("✅ Configuration loaded successfully")
        print(f"   MQTT Broker: {mqtt_config['broker']}:{mqtt_config['port']}")
        print(f"   WiFi SSID: {wifi_config['ssid']}")
        print(f"   Device Name: {device_config['name']}")
        
        # Test validation
        validate_config()
        print("✅ Configuration validation passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Network config test failed: {e}")
        return False

def test_module_imports():
    """Test that updated modules can import network config"""
    print("\nTesting module imports...")
    
    modules_to_test = [
        "main_system_orchestrator",
        "data_collection_96hour_headless", 
        "system_status_checker",
        "system_validation"
    ]
    
    success_count = 0
    
    for module_name in modules_to_test:
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            print(f"✅ {module_name} imported successfully")
            success_count += 1
            
        except Exception as e:
            print(f"❌ {module_name} import failed: {e}")
    
    return success_count == len(modules_to_test)

def test_configuration_consistency():
    """Test that all modules use consistent configuration"""
    print("\nTesting configuration consistency...")
    
    try:
        from config.network_config import MQTT_SERVER, MQTT_PORT, WIFI_SSID, DEVICE_NAME
        
        expected_values = {
            "MQTT_SERVER": "192.168.43.232",
            "MQTT_PORT": 1883,
            "WIFI_SSID": "TP-Link_E4-FC", 
            "DEVICE_NAME": "ESP32-MPC-Controller"
        }
        
        actual_values = {
            "MQTT_SERVER": MQTT_SERVER,
            "MQTT_PORT": MQTT_PORT,
            "WIFI_SSID": WIFI_SSID,
            "DEVICE_NAME": DEVICE_NAME
        }
        
        all_correct = True
        for key, expected in expected_values.items():
            actual = actual_values[key]
            if actual == expected:
                print(f"✅ {key}: {actual}")
            else:
                print(f"❌ {key}: expected '{expected}', got '{actual}'")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"❌ Configuration consistency test failed: {e}")
        return False

def test_mqtt_topics():
    """Test MQTT topic structure"""
    print("\nTesting MQTT topic structure...")
    
    try:
        from config.network_config import MQTT_TOPICS, DEVICE_NAME
        
        expected_prefix = f"mpc_smart_charging/{DEVICE_NAME}"
        
        all_correct = True
        for topic_name, topic_path in MQTT_TOPICS.items():
            if topic_path.startswith(expected_prefix):
                print(f"✅ {topic_name}: {topic_path}")
            else:
                print(f"❌ {topic_name}: incorrect prefix - {topic_path}")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"❌ MQTT topics test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("NETWORK CONFIGURATION VALIDATION TEST")
    print("=" * 60)
    
    tests = [
        ("Network Config Module", test_network_config),
        ("Module Imports", test_module_imports),
        ("Configuration Consistency", test_configuration_consistency),
        ("MQTT Topics", test_mqtt_topics)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - Network configuration is synchronized!")
        return True
    else:
        print("⚠️  SOME TESTS FAILED - Check configuration and imports")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
