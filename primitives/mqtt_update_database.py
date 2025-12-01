#!/usr/bin/env python3
"""
MQTT Publisher for LEGO Sorting System
Publishes database update messages via MQTT to update the sorting database.

Usage:
    # Insert a Processing record
    python3 primitives/mqtt_update_database.py --action insert --aruco_id 1 --color Red --count 3
    
    # Update Processing to Completed
    python3 primitives/mqtt_update_database.py --action update --aruco_id 1 --color Red
    
    # Insert with custom status
    python3 primitives/mqtt_update_database.py --action insert --aruco_id 2 --color Blue --status Processing --count 5
"""

import argparse
import time
import paho.mqtt.client as mqtt
import json

BROKER = "broker.hivemq.com"   # Public test broker (do NOT send secrets)
PORT = 1883                    # Unencrypted MQTT port
CLIENT_ID = "LEGO_Sorting_Publisher"  # Make unique if many clients run this

# Topic configuration
TOPIC = "lego_sorting/sql_update"

# Global variables for callbacks
publish_success = False
publish_error = None


def on_connect(client, userdata, flags, rc, properties=None):
    """Called when the client connects to the broker.
    rc == 0 means success; non-zero indicates a failure code.
    """
    if rc == 0:
        print(f"✓ Connected to {BROKER} (rc={rc})")
    else:
        print(f"⚠ Connection failed (rc={rc})")


def on_publish(client, userdata, mid, reason_codes=None, properties=None):
    """Called when a publish completes.
    - QoS 0: fired after the packet is sent by the client (no broker ack).
    - QoS 1/2: fired after broker ack (PUBACK/PUBCOMP).
    'mid' is the message ID for the published message.
    """
    global publish_success
    publish_success = True
    print(f"✓ Publish confirmed (mid={mid})")


def publish_message(action, aruco_id, color, status=None, count=None):
    """Publish a single MQTT message to update the database"""
    global publish_success, publish_error
    
    # Reset flags
    publish_success = False
    publish_error = None
    
    # Create client with VERSION2 callback API
    client = mqtt.Client(client_id=CLIENT_ID, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_publish = on_publish
    
    try:
        # Connect to broker
        print(f"Connecting to {BROKER}:{PORT}...")
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()  # Start the background network loop
        print("Loop started. Publishing message...")
        time.sleep(1)  # Wait for connection
        
        # Build message based on action
        if action == "insert":
            if status is None:
                status = "Processing"
            if count is None:
                return False, "count is required for insert action"
            
            message = {
                "action": "insert",
                "aruco_marker_id": aruco_id,
                "color": color,
                "status": status,
                "count": count
            }
        elif action == "update":
            message = {
                "action": "update",
                "aruco_marker_id": aruco_id,
                "color": color
            }
        else:
            return False, f"Invalid action: {action}. Must be 'insert' or 'update'"
        
        # Publish message
        payload = json.dumps(message)
        info = client.publish(TOPIC, payload=payload, qos=1, retain=False)
        print(f"Published {payload} → {TOPIC}")
        
        # Wait for publish confirmation
        timeout = 5.0  # 5 seconds timeout
        start_time = time.time()
        while not publish_success and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if publish_success:
            print("✓ Message published successfully")
            return True, "Message published successfully"
        else:
            return False, "Publish timeout - message may not have been delivered"
            
    except Exception as e:
        publish_error = str(e)
        print(f"⚠ Error: {e}")
        return False, f"Error publishing message: {e}"
    finally:
        # Always stop the loop and disconnect cleanly
        try:
            client.loop_stop()
            client.disconnect()
            print("✓ Disconnected")
        except:
            pass


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Publish MQTT message to update LEGO sorting database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Insert a Processing record
  python3 primitives/mqtt_update_database.py --action insert --aruco_id 1 --color Red --count 3
  
  # Update Processing to Completed
  python3 primitives/mqtt_update_database.py --action update --aruco_id 1 --color Red
  
  # Insert with custom status
  python3 primitives/mqtt_update_database.py --action insert --aruco_id 2 --color Blue --status Processing --count 5
        """
    )
    
    parser.add_argument(
        '--action',
        choices=['insert', 'update'],
        required=True,
        help='Action to perform: insert (add new record) or update (update status to Completed)'
    )
    
    parser.add_argument(
        '--aruco_id',
        type=int,
        required=True,
        help='ArUco marker ID (integer)'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        required=True,
        help='Color of the object (e.g., Red, Blue, Green, Yellow)'
    )
    
    parser.add_argument(
        '--status',
        type=str,
        default='Processing',
        help='Status for insert action (default: Processing)'
    )
    
    parser.add_argument(
        '--count',
        type=int,
        default=None,
        help='Count for insert action (required for insert, ignored for update)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.action == "insert" and args.count is None:
        parser.error("--count is required for insert action")
    
    # Publish message
    success, message = publish_message(
        action=args.action,
        aruco_id=args.aruco_id,
        color=args.color,
        status=args.status if args.action == "insert" else None,
        count=args.count
    )
    
    if success:
        print(f"Success: {message}")
        return 0
    else:
        print(f"Error: {message}")
        return 1


if __name__ == '__main__':
    exit(main())
