import serial
import time
import csv
import re
from datetime import datetime

# -------------------------- CONFIG --------------------------
SERIAL_PORT = '/dev/ttyUSB1'
BAUD_RATE = 3000000
TIMEOUT = 0.1
OUTPUT_FILE = './data/motor_telemetry.csv' 
DURATION_SECONDS = None 

# -------------------------- SETUP --------------------------
pattern = re.compile(r'ERR:(\d+)\s+VREF:(\d+)\s+CURRENT:(\d+)')

with open(OUTPUT_FILE, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([ 'ERROR', 'CURRENT', 'VREF'])

print(f"Zapisuję dane do {OUTPUT_FILE} (Ctrl+C aby zatrzymać)")

ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT)
ser.flushInput()

buffer = ""

start_time = time.time()

try:
    while True:
        if DURATION_SECONDS is not None and (time.time() - start_time) > DURATION_SECONDS:
            break

        if ser.in_waiting > 0:
            raw = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
            buffer += raw

        while '\n' in buffer in buffer:
            line, buffer = re.split(r'[\n]', buffer, 1)
            line = line.strip()
            if not line:
                continue

            match = pattern.fullmatch(line)
            if match:
                err = int(match.group(1))
                vref = int(match.group(2))
                cur = int(match.group(3))
                
                with open(OUTPUT_FILE, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([err, cur, vref])

                print(f"ERROR:{err:5} | CURRENT:{cur:3} | VREF:{vref:4}")

except KeyboardInterrupt:
    print("\nCancelled by the user.")
finally:
    ser.close()
    print(f"Data written to: {OUTPUT_FILE}")