## ------------------------------------------ packages ------------------------------------------ ##
import time
import psutil
import uuid
import json
import paho.mqtt.client as mqtt


## ------------------------------------- initial variables -------------------------------------- ##
MAC_ADDRESS = hex(uuid.getnode())

MQTT_HOST = 'mqtt.eclipseprojects.io'
MQTT_PORT = 1883
MQTT_TOPIC = 's308563'


## -------------------------------------- mqtt connection --------------------------------------- ##
print('Message broker connection status:')
mqtt_client = mqtt.Client()
mqtt_client.connect(MQTT_HOST, MQTT_PORT)
print(' -- connection result \u2192 True')


## ----------------------------------------- main loop ------------------------------------------ ##
print('Log:')
while True:

    events = []
    for _ in range(10):
        events.append({
            'timestamp': int(1000 * time.time()),
            'battery_level': psutil.sensors_battery().percent,
            'power_plugged': psutil.sensors_battery().power_plugged
        })
        time.sleep(1)
    print('  -- 10 events were acquired')
    
    message = json.dumps({
        'mac_address': MAC_ADDRESS,
        'events': events
    })
    mqtt_client.publish(MQTT_TOPIC, message)
    print('  -- message sent to the broker')