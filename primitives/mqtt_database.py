#!/usr/bin/env python3
"""
MQTT Database Client for LEGO Sorting System
Handles both database updates (insert/update) and queries (read) via MQTT.
Publishes update messages and query requests, subscribes to query responses.

Usage:
    # Insert a Processing record
    python3 primitives/mqtt_database.py --action insert --aruco_id 1 --color Red --count 3
    
    # Update Processing to Completed
    python3 primitives/mqtt_database.py --action update --aruco_id 1 --color Red
    
    # Insert with custom status
    python3 primitives/mqtt_database.py --action insert --aruco_id 2 --color Blue --status Processing --count 5
    
    # Query all records
    python3 primitives/mqtt_database.py --action query
    
    # Query by color
    python3 primitives/mqtt_database.py --action query --color Red
    
    # Query by aruco_id
    python3 primitives/mqtt_database.py --action query --aruco_id 1
"""

import argparse
import time
import paho.mqtt.client as mqtt
import json
import sys

BROKER = "broker.hivemq.com"   # Public test broker (do NOT send secrets)
PORT = 1883                    # Unencrypted MQTT port
CLIENT_ID_PUBLISHER = "LEGO_Sorting_Publisher"  # Make unique if many clients run this
CLIENT_ID_SUBSCRIBER = "LEGO_Sorting_Subscriber"  # For query operations


def normalize_color(color):
    """Normalize color string to consistent format: first letter capitalized, rest lowercase"""
    if not color:
        return color
    return color.capitalize()

# Topic configuration
UPDATE_TOPIC = "lego_sorting/sql_update"
QUERY_TOPIC = "lego_sorting/sql_query"
RESPONSE_TOPIC = "lego_sorting/sql_response"

# Global variables for callbacks
publish_success = False
publish_error = None
query_response = None
query_received = False


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


def on_message(client, userdata, msg):
    """Called when a message is received from the broker."""
    global query_response, query_received
    try:
        payload = msg.payload.decode('utf-8')
        query_response = json.loads(payload)
        query_received = True
        print(f"✓ Received query response: {payload}")
    except Exception as e:
        print(f"⚠ Error parsing query response: {e}")
        query_response = {"error": f"Failed to parse response: {e}"}
        query_received = True


def publish_message(action, aruco_id, color, status=None, count=None):
    """Publish a single MQTT message to update the database"""
    global publish_success, publish_error
    
    # Reset flags
    publish_success = False
    publish_error = None
    
    # Normalize color to consistent format
    if color:
        color = normalize_color(color)
    
    # Create client with VERSION2 callback API
    client = mqtt.Client(client_id=CLIENT_ID_PUBLISHER, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
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
        info = client.publish(UPDATE_TOPIC, payload=payload, qos=1, retain=False)
        print(f"Published {payload} → {UPDATE_TOPIC}")
        
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


def query_database(color=None, aruco_id=None):
    """Query the database via MQTT and return results"""
    global query_response, query_received
    
    # Reset flags
    query_response = None
    query_received = False
    
    # Normalize color to consistent format if provided
    if color:
        color = normalize_color(color)
    
    # Create client with VERSION2 callback API
    client = mqtt.Client(client_id=CLIENT_ID_SUBSCRIBER, callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        # Connect to broker
        print(f"Connecting to {BROKER}:{PORT}...")
        client.connect(BROKER, PORT, keepalive=60)
        client.loop_start()  # Start the background network loop
        time.sleep(1)  # Wait for connection
        
        # Subscribe to response topic
        print(f"Subscribing to {RESPONSE_TOPIC}...")
        client.subscribe(RESPONSE_TOPIC, qos=1)
        time.sleep(0.5)  # Wait for subscription to be established
        
        # Build query message
        message = {
            "action": "query"
        }
        
        if color is not None:
            message["color"] = color
        if aruco_id is not None:
            message["aruco_marker_id"] = int(aruco_id)
        
        # Publish query request
        payload = json.dumps(message)
        info = client.publish(QUERY_TOPIC, payload=payload, qos=1, retain=False)
        print(f"Published query {payload} → {QUERY_TOPIC}")
        
        # Wait for response
        timeout = 10.0  # 10 seconds timeout for query
        start_time = time.time()
        while not query_received and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if query_received:
            print("✓ Query response received")
            return True, query_response
        else:
            return False, "Query timeout - no response received"
            
    except Exception as e:
        print(f"⚠ Error: {e}")
        return False, f"Error querying database: {e}"
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
        description='Publish/Query MQTT messages for LEGO sorting database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Insert a Processing record
  python3 primitives/mqtt_database.py --action insert --aruco_id 1 --color Red --count 3
  
  # Update Processing to Completed
  python3 primitives/mqtt_database.py --action update --aruco_id 1 --color Red
  
  # Insert with custom status
  python3 primitives/mqtt_database.py --action insert --aruco_id 2 --color Blue --status Processing --count 5
  
  # Query all records
  python3 primitives/mqtt_database.py --action query
  
  # Query by color
  python3 primitives/mqtt_database.py --action query --color Red
  
  # Query by aruco_id
  python3 primitives/mqtt_database.py --action query --aruco_id 1
        """
    )
    
    parser.add_argument(
        '--action',
        choices=['insert', 'update', 'query'],
        required=True,
        help='Action to perform: insert (add new record), update (update status to Completed), or query (read database)'
    )
    
    parser.add_argument(
        '--aruco_id',
        type=int,
        default=None,
        help='ArUco marker ID (integer). Required for insert/update, optional for query.'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default=None,
        help='Color of the object (e.g., Red, Blue, Green, Yellow). Required for insert/update, optional for query.'
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
        help='Count for insert action (required for insert, ignored for update/query)'
    )
    
    args = parser.parse_args()
    
    # Handle query action
    if args.action == "query":
        success, result = query_database(color=args.color, aruco_id=args.aruco_id)
        if success:
            # Print JSON result to stdout for parsing
            print(json.dumps(result, indent=2))
            return 0
        else:
            print(f"Error: {result}", file=sys.stderr)
            return 1
    
    # Validate arguments for insert/update
    if args.aruco_id is None:
        parser.error("--aruco_id is required for insert/update actions")
    if args.color is None:
        parser.error("--color is required for insert/update actions")
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
