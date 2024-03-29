## ------------------------------------------ packages ------------------------------------------ ##
import argparse
import psutil
import uuid
import time
import redis

## ----------------------------------------- arguments ------------------------------------------ ##
parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, default='redis-15028.c250.eu-central-1-1.ec2.cloud.redislabs.com')
parser.add_argument('--port', type=int, default=15028)
parser.add_argument('--user', type=str, default='default')
parser.add_argument('--password', type=str, default='q2pbNpweUA0LhCre3Ul06TUBInGrMCzn')
args = parser.parse_args()

## ------------------------------------- initial variables -------------------------------------- ##
# general variables
MAC_ADDRESS = hex(uuid.getnode()) # for all time-series
ONE_DAY_IN_MS = 24 * 60 * 60 * 1000 # for battery and power time-series
THIRTY_DAYS_IN_MS = 30*ONE_DAY_IN_MS # for plugged_seconds time-series
BUCKET_SIZE_IN_MS = 60 * 60 * 1000 # for plugged_seconds time-series

# time-series names
BATTERY_TS_NAME = f'{MAC_ADDRESS}:battery'
POWER_TS_NAME = f'{MAC_ADDRESS}:power'
PLUGGED_SECONDS_TS_NAME = f'{MAC_ADDRESS}:plugged_seconds'

## -------------------------------------- redis connection -------------------------------------- ##
try:   
    redis_client = redis.Redis(host=args.host, port=args.port, username=args.user, password=args.password)
    is_connected = redis_client.ping()
    print(f'Redis Connection Status: {is_connected}')
except:
    print(f'Redis Connection Failed!')
    exit()

## ------------------------------------ time-series creation ------------------------------------ ##
# creating/modifying battery time-series
try:
    redis_client.ts().create(BATTERY_TS_NAME, retention_msecs=ONE_DAY_IN_MS)
except:
    redis_client.ts().alter(BATTERY_TS_NAME, retention_msecs=ONE_DAY_IN_MS)

# creating/modifying power time-series
try:
    redis_client.ts().create(POWER_TS_NAME, retention_msecs=ONE_DAY_IN_MS)
except:
    redis_client.ts().alter(POWER_TS_NAME, retention_msecs=ONE_DAY_IN_MS)

# creating/modifying plugged_seconds time-series
try:
    redis_client.ts().create(PLUGGED_SECONDS_TS_NAME, retention_msecs=THIRTY_DAYS_IN_MS)
    redis_client.ts().createrule(POWER_TS_NAME, PLUGGED_SECONDS_TS_NAME, 'sum', bucket_size_msec=BUCKET_SIZE_IN_MS)
except:
    redis_client.ts().alter(PLUGGED_SECONDS_TS_NAME, retention_msecs=THIRTY_DAYS_IN_MS)
    redis_client.ts().deleterule(POWER_TS_NAME, PLUGGED_SECONDS_TS_NAME)
    redis_client.ts().createrule(POWER_TS_NAME, PLUGGED_SECONDS_TS_NAME, 'sum', bucket_size_msec=BUCKET_SIZE_IN_MS)

## ----------------------------------------- main-loop ------------------------------------------ ##
while True:
    # getting data
    timestamp = time.time()
    timestamp_ms = int(timestamp * 1000)
    battery_level = psutil.sensors_battery().percent
    power_plugged = int(psutil.sensors_battery().power_plugged)
    
    # uploading data
    redis_client.ts().add(BATTERY_TS_NAME, timestamp_ms, battery_level)
    redis_client.ts().add(POWER_TS_NAME, timestamp_ms, power_plugged)
    
    # sleeping for 1 second
    time.sleep(1)