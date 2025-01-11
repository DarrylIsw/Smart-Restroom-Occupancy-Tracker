from flask import Flask, request, jsonify
from waitress import serve
import os
import cv2
import torch
import base64
import pymysql
import traceback
import json
from threading import Lock, Thread, Event
import queue
import uuid
from datetime import datetime
import time
from facenet_pytorch import MTCNN
import numpy as np
import tempfile

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # Increase to 100 MB if needed

# Initialize MTCNN model for face detection
mtcnn = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.eval()  # Set model to evaluation mode

# Database configuration
DB_HOST = '(replace here)'  # Change if your MySQL is hosted elsewhere
DB_USER = '(replace here)'       # Your MySQL username
DB_PASSWORD = '(replace here)'       # Your MySQL password
DB_NAME = '(replace here)'  # Your database name
DB_PORT = 3306         # Default MySQL port

# Threading and Queue for safe database operations
db_lock = Lock()
pending_tasks = queue.Queue()
db_connected_event = Event()

# Directory to save processed images
save_folder = 'processed_images_test_usage_6'
save_dir = 'test_dir'
os.makedirs(save_folder, exist_ok=True)

# Function to establish a database connection
def get_db_connection():
    while True:
        try:
            connection = pymysql.connect(
                host=DB_HOST,
                user=DB_USER,
                password=DB_PASSWORD,
                database=DB_NAME,
                port=DB_PORT,
                charset='utf8mb4'
            )
            db_connected_event.set()  # Signal that the database is connected
            print("Database connected successfully.")
            return connection
        except pymysql.MySQLError as e:
            print(f"Database connection failed: {e}. Retrying in 5 seconds...")
            db_connected_event.clear()  # Signal that the database is disconnected
            time.sleep(5)

# Background thread to process pending tasks
def process_pending_tasks():
    while True:
        db_connected_event.wait()  # Wait until the database is connected
        try:
            # Wait for a task from the queue
            task = pending_tasks.get(block=True)
            
            # Unpack the task
            toilet_id, floor_id, building_id, gender_id, logged_at, usage_count = task
            
            with db_lock:
                connection = get_db_connection()
                with connection.cursor() as cursor:
                    # Insert into logging_counters
                    sql_insert_logs = """
                        INSERT INTO logging_counters (
                            toilet_id, floor_id, building_id, gender_id, logged_at, usage_count, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, %s, %s, NOW(), NOW())
                    """
                    cursor.execute(sql_insert_logs, (
                        toilet_id, floor_id, building_id, gender_id, logged_at, usage_count
                    ))
                    connection.commit()
                    print(f"Inserted into logging_counters: toilet_id={toilet_id}, floor_id={floor_id}, building_id={building_id}, "
                          f"gender_id={gender_id}, logged_at={logged_at}, usage_count={usage_count}.")
                    
                    # Update toilets table usage_count
                    sql_update_toilets = """
                        UPDATE toilets
                        SET usage_count = usage_count + %s, updated_at = NOW()
                        WHERE id = %s AND floor_id = %s AND gender_id = %s
                    """
                    cursor.execute(sql_update_toilets, (usage_count, toilet_id, floor_id, gender_id))
                    connection.commit()
                    print(f"Updated toilets table: Incremented usage_count by {usage_count} for toilet_id={toilet_id}, floor_id={floor_id}, gender_id={gender_id}.")
        except Exception as e:
            print(f"Error processing task: {e}")
            traceback.print_exc()
        finally:
            pending_tasks.task_done()

# Start the background thread
Thread(target=process_pending_tasks, daemon=True).start()

def load_image(image_path):
    """Load and process the image."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img, img_rgb

def increase_brightness(img, value=50):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)  # Add brightness value
    v = np.clip(v, 0, 255)  # Clip values to valid range
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2RGB)  # Convert back to RGB

def rotate_image_upside_down(img_rgb):
    """Rotate the image 180 degrees (upside down)."""
    return cv2.rotate(img_rgb, cv2.ROTATE_180)

def detect_faces(img_rgb):
    """Detect faces and count them using MTCNN."""
    boxes, probs = mtcnn.detect(img_rgb)
    
    # Count the number of faces detected
    usage_count = 0
    if boxes is not None:
        usage_count = len(boxes)  # Number of faces detected
        for box in boxes:
            # Draw bounding box around each face (optional)
            cv2.rectangle(img_rgb, 
                          (int(box[0]), int(box[1])), 
                          (int(box[2]), int(box[3])), 
                          (0, 255, 0), 2)
    return img_rgb, usage_count

def detect_faces_yolov5(img_rgb):
    
    # Run inference
    results = model(img_rgb)  # model does auto resize if not specified
    detections = results.xyxy[0].cpu().numpy()  # Convert to NumPy array
    
    usage_count = 0
    for *box, conf, cls in detections:
        
        # You can draw rectangles on img_rgb if you like:
        cv2.rectangle(img_rgb, 
                      (int(box[0]), int(box[1])),
                      (int(box[2]), int(box[3])),
                      (0, 255, 0), 2)
        usage_count += 1
    
    return img_rgb, usage_count


def sanitize_json(raw_data):
    try:
        # Try loading the JSON directly
        return json.loads(raw_data)
    except json.JSONDecodeError as e:
        print(f"Initial JSON error: {e}")
        
        # Attempt to fix common issues
        fixed_data = raw_data.strip()

        # Ensure it ends with a closing brace
        if not fixed_data.endswith("}"):
            print("Fixing missing closing brace.")
            fixed_data += "}"

        # Remove any trailing commas before the closing brace
        fixed_data = fixed_data.replace(",}", "}").replace(", ]", " ]")

        # Escape unescaped quotes within values
        fixed_data = fixed_data.replace('\\"', '"').replace('"', '\\"')

        # Attempt to parse again
        try:
            return json.loads(fixed_data)
        except json.JSONDecodeError as final_e:
            print(f"Failed to fix JSON: {final_e}")
            raise ValueError("Unable to fix malformed JSON.")

@app.route('/favicon.ico')
def favicon():
    return '', 204  # Respond with "No Content"
@app.route('/')
def home():
    return "Welcome to the Flask App!"

@app.route('/process-image', methods=['POST'])
def process_image():
    try:
        # Get raw data and try to fix JSON if malformed
        raw_data = request.data.decode('utf-8')
        try:
            data = sanitize_json(raw_data)
        except ValueError as fix_e:
            print(f"JSON fixing failed: {fix_e}")
            return jsonify({"message": f"Invalid JSON: {fix_e}"}), 400

        # Validate required fields
        required_fields = ['toilet_id', 'floor_id', 'building_id', 'gender_id', 'timestamp', 'image']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            raise ValueError(f"Missing fields: {missing_fields}")

        # Extract fields
        toilet_id = int(data['toilet_id'])
        floor_id = int(data['floor_id'])
        building_id = int(data['building_id'])
        gender_id = int(data['gender_id'])
        timestamp = data['timestamp']  # This is the `timestamp` from the JSON
        base64_image = data['image']

        # Decode and save the image with a unique filename
        try:
            decoded_image = base64.b64decode(base64_image)

            # Convert binary data to a NumPy array
            nparr = np.frombuffer(decoded_image, np.uint8)

            # Decode the NumPy array to an image
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Rotate the image upside down
            img_rotated = rotate_image_upside_down(img)

            # Save the processed image
            unique_id = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}"
            image_filename = os.path.join(
                save_folder, f"temp_image_{toilet_id}_{floor_id}_{gender_id}_{unique_id}.jpg"
            )
            cv2.imwrite(image_filename, img_rotated)
            print(f"Image saved successfully as {image_filename}")
        except Exception as e:
            print(f"Error decoding or processing image: {e}")
            traceback.print_exc()
            return jsonify({"message": f"Error decoding or processing image: {e}"}), 400

        # Analyze the image to count people
        try:
            img, img_rgb = load_image(image_filename)
            img_rgb_rotated = rotate_image_upside_down(img_rgb)  # Rotate the image upside down
            #img_with_faces, usage_count = detect_faces(img_rgb_rotated)
            img_with_faces, usage_count = detect_faces_yolov5(img_rgb_rotated)

            print(f"People detected: {usage_count}")
        except Exception as e:
            print(f"Error analyzing image: {e}")
            traceback.print_exc()
            return jsonify({"message": f"Error analyzing image: {e}"}), 500
        
        # Skip database insertion if no people are detected
        if usage_count == 0:
            print("No people detected. Skipping database insertion.")
            return jsonify({"message": "No people detected. Skipped database insertion."}), 200
        
        # Insert data into the database
        try:
            with db_lock:
                connection = get_db_connection()
                with connection.cursor() as cursor:
                    # Insert into logging_counters
                    sql_insert = """
                        INSERT INTO logging_counters (
                            toilet_id, floor_id, building_id, gender_id, logged_at, usage_count, created_at, updated_at
                        ) VALUES (%s, %s, %s, %s, NOW(), %s, NOW(), NOW())
                    """  # Use NOW() for created_at and updated_at
                    logged_at = timestamp  # Use the timestamp from the JSON
                    cursor.execute(sql_insert, (
                        toilet_id, floor_id, building_id, gender_id, usage_count
                    ))
                    connection.commit()
                    print(f"Inserted new record: toilet_id={toilet_id}, floor_id={floor_id}, building_id={building_id}, "
                        f"gender_id={gender_id}, logged_at={logged_at}, usage_count={usage_count}.")

                    # Update toilets table usage_count
                    sql_update = """
                        UPDATE toilets
                        SET 
                            usage_count = usage_count + %s,
                            updated_at = NOW(),
                            created_at = CASE 
                                WHEN created_at IS NULL THEN NOW() 
                                ELSE created_at 
                            END
                        WHERE id = %s AND floor_id = %s AND gender_id = %s
                    """
                    cursor.execute(sql_update, (usage_count, toilet_id, floor_id, gender_id))
                    connection.commit()
                    print(f"Updated toilets table: Incremented usage_count by {usage_count} for toilet_id={toilet_id}, floor_id={floor_id}, gender_id={gender_id}.")

        except Exception as e:
            print(f"Database insertion or update failed: {e}")
            traceback.print_exc()
            return jsonify({"message": f"Database insertion or update failed: {e}"}), 500

        #Move the processed image to the processed_image_dir
        try:
            processed_image_path = os.path.join(save_dir, image_filename)
            os.rename(image_filename, processed_image_path)
            print(f"Image moved to {processed_image_path}")
        except Exception as e:
            print(f"Error moving processed image: {e}")
            traceback.print_exc()

        return jsonify({"message": "Image processed successfully", "usage_count": usage_count}), 200

    except Exception as e:
        print("Unhandled server error:")
        traceback.print_exc()
        return jsonify({"message": f"Unhandled server error: {e}"}), 500

if __name__ == '__main__':
    Thread(target=get_db_connection, daemon=True).start()  # Background thread to reconnect database
    app.run(host='0.0.0.0', port=5000)
