import cv2
import numpy as np
import dlib
import random
import time
from scipy.spatial import distance as dist
import tensorflow as tf

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Blink detection function
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear
    
# Lighting check function
def check_lighting(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    brightness = hsv[..., 2].mean()
    if brightness < 50:
        return "Lighting is too low", False
    elif brightness > 200:
        return "Lighting is too bright", False
    return "",True

# Generate a random math problem with single-digit answer
def generate_math_problem():
    operators = ['+', '-', '*']
    while True:
        num1 = random.randint(1, 9)
        num2 = random.randint(1, 9)
        operator = random.choice(operators)
        problem = f"{num1} {operator} {num2}"
        correct_answer = eval(problem)
        if 0 <= correct_answer <= 9:
            return problem, correct_answer

# Save the captured frame and cropped image as 28x28 with black background and white text
def capture_images(frame, roi):
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, thresholded_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY)
    
    # Check if the image is already black background with white text
    white_pixel_count = np.sum(thresholded_roi == 255)
    total_pixels = thresholded_roi.size
    
    if white_pixel_count / total_pixels > 0.5:
        # If more than 50% pixels are white, invert the image
        thresholded_roi = cv2.bitwise_not(thresholded_roi)
    
    resized_roi = cv2.resize(thresholded_roi, (28, 28))

    # Save the full frame and cropped image
    cv2.imwrite("full_frame.jpg", frame)
    cv2.imwrite("cropped_box_28x28_black_bg.jpg", resized_roi)
    
    # Show the processed black background image
    cv2.imshow("Processed Image", resized_roi)
    cv2.waitKey(0)  # Display the processed image until key press
    cv2.destroyAllWindows()

    return resized_roi  # Return the processed image for digit recognition

# Load the digit recognition model
def load_digit_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        print('Loaded saved model.')
    except:
        print("Model not found. Please train the model first.")
        return None
    return model

# Predict digit using model and image
def predict_digit(model, img):
    img = np.array([img]).reshape(1, 28, 28, 1) / 255.0  # Prepare the image for the model
    res = model.predict(img)
    index = np.argmax(res)
    return str(index)

# Main liveness detection and digit recognition pipeline
def run_liveness_detection():
    cap = cv2.VideoCapture(0)

    EYE_AR_THRESH = 0.25
    COUNTER = 0
    BLINKS = 0
    MIN_BLINKS = 3
    MAX_FRAMES_WITHOUT_BLINK = 90
    NO_BLINK_COUNTER = 0

    FACE_DETECT_FRAMES = 5
    face_detect_counter = 0
    face_detected = False

    # Load the digit recognition model
    model = load_digit_model()
    if model is None:
        return

    # Generate a random math problem
    math_problem, correct_answer = generate_math_problem()
    print(f"Math Problem: {math_problem}")
    print("Please solve the problem and write the answer clearly in the circular box.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        height, width = frame.shape[:2]
        center = (int(width / 2), int(height * 0.85))
        radius = int(min(width, height) * 0.15)

        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.putText(frame, "Place your answer here", 
                    (center[0] - 100, center[1] - radius - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        lighting_status, correct_lighting = check_lighting(frame)
        cv2.putText(frame, lighting_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, (0, 255, 0) if correct_lighting else (0, 0, 255), 2)

        if len(faces) == 0:
            face_detect_counter += 1
            if face_detect_counter >= FACE_DETECT_FRAMES:
                cv2.putText(frame, "No face detected", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                face_detected = False
        else:
            face_detect_counter = 0
            face_detected = True

            if len(faces) > 1:
                print("Multiple faces detected!")
                cv2.putText(frame, "Multiple faces detected!", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                time.sleep(5)
                break
            else:
                face = faces[0]
                shape = predictor(gray, face)
                shape = np.array([[p.x, p.y] for p in shape.parts()])

                left_eye = shape[36:42]
                right_eye = shape[42:48]
                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                else:
                    if COUNTER >= 2:
                        BLINKS += 1
                    COUNTER = 0
                    NO_BLINK_COUNTER = 0

                NO_BLINK_COUNTER += 1
                if NO_BLINK_COUNTER > MAX_FRAMES_WITHOUT_BLINK:
                    BLINKS = 0
                    NO_BLINK_COUNTER = 0

                cv2.putText(frame, f"Blink Count: {BLINKS}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        if face_detected and BLINKS >= MIN_BLINKS and correct_lighting:
            cv2.putText(frame, "Press 'c' to capture image", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Liveness Detection & Digit Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # Press 'Esc' to exit
            break
        elif key == ord('c') and face_detected and BLINKS >= MIN_BLINKS and correct_lighting:
            # Capture the region of interest (ROI) inside the circular box
            roi = frame[center[1] - radius:center[1] + radius, 
                        center[0] - radius:center[0] + radius]

            # Preprocess and save the captured image
            processed_image = capture_images(frame, roi)

            # Predict the digit from the captured image
            predicted_digit = predict_digit(model, processed_image)
            print(f"Predicted digit: {predicted_digit}")

            # Verify the answer
            if int(predicted_digit) == correct_answer:
                print("Passed!")
            else:
                print("Wrong Answer!")
            
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the combined liveness detection and digit recognition pipeline
run_liveness_detection()