#!/usr/bin/env python3
"""
Network Connectivity Test
========================
Test konektivitas dengan konfigurasi jaringan yang baru:
- WiFi: TP-Link_E4-FC
- MQTT: 192.168.43.207:1883
"""

import socket
import subprocess
import time
from config.network_config import get_mqtt_config, get_wifi_config

def test_network_connectivity():
    """Test basic network connectivity"""
    print("Testing network connectivity...")
    
    mqtt_config = get_mqtt_config()
    broker_ip = mqtt_config["broker"]
    
    try:
        # Test ping to MQTT broker
        result = subprocess.run(
            ['ping', '-c', '3', broker_ip], 
            capture_output=True, 
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            print(f"✅ Ping to {broker_ip}: SUCCESS")
            return True
        else:
            print(f"❌ Ping to {broker_ip}: FAILED")
            print(f"   Error: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"❌ Ping to {broker_ip}: TIMEOUT")
        return False
    except Exception as e:
        print(f"❌ Ping test failed: {e}")
        return False

def test_mqtt_port():
    """Test MQTT port connectivity"""
    print("Testing MQTT port connectivity...")
    
    mqtt_config = get_mqtt_config()
    broker_ip = mqtt_config["broker"]
    broker_port = mqtt_config["port"]
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((broker_ip, broker_port))
        sock.close()
        
        if result == 0:
            print(f"✅ MQTT port {broker_ip}:{broker_port}: OPEN")
            return True
        else:
            print(f"❌ MQTT port {broker_ip}:{broker_port}: CLOSED")
            return False
            
    except Exception as e:
        print(f"❌ MQTT port test failed: {e}")
        return False

def test_mqtt_client():
    """Test MQTT client connection"""
    print("Testing MQTT client connection...")
    
    try:
        import paho.mqtt.client as mqtt
        from config.network_config import get_mqtt_config
        
        mqtt_config = get_mqtt_config()
        
        connected = False
        
        def on_connect(client, userdata, flags, rc):
            nonlocal connected
            if rc == 0:
                connected = True
                print("✅ MQTT client connection: SUCCESS")
                client.disconnect()
            else:
                print(f"❌ MQTT client connection: FAILED (code {rc})")
        
        client = mqtt.Client()
        client.on_connect = on_connect
        
        client.connect(mqtt_config["broker"], mqtt_config["port"], 60)
        client.loop_start()
        
        # Wait for connection result
        for i in range(10):
            if connected:
                break
            time.sleep(0.5)
        
        client.loop_stop()
        return connected
        
    except ImportError:
        print("❌ MQTT client test: paho-mqtt not installed")
        return False
    except Exception as e:
        print(f"❌ MQTT client test failed: {e}")
        return False

def main():
    """Run all connectivity tests"""
    print("=" * 60)
    print("NETWORK CONNECTIVITY TEST")
    print("=" * 60)
    
    wifi_config = get_wifi_config()
    mqtt_config = get_mqtt_config()
    
    print(f"Testing connection to:")
    print(f"WiFi SSID: {wifi_config['ssid']}")
    print(f"MQTT Broker: {mqtt_config['broker']}:{mqtt_config['port']}")
    print("")
    
    tests = [
        ("Network Connectivity", test_network_connectivity),
        ("MQTT Port Check", test_mqtt_port),
        ("MQTT Client Connection", test_mqtt_client)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"{test_name}:")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
            print("")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            print("")
    
    print("=" * 60)
    print(f"CONNECTIVITY TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL CONNECTIVITY TESTS PASSED!")
        print("✅ System ready for deployment")
    elif passed > 0:
        print("⚠️  PARTIAL CONNECTIVITY - Check network settings")
    else:
        print("❌ NO CONNECTIVITY - Verify network configuration")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
