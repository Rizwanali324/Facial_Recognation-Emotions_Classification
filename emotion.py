'''
 #original model without optimzation

 
import cv2
from keras.models import model_from_json
import numpy as np
from screeninfo import get_monitors
import os

# Ensure the directory exists
save_dir = 'web/images/predicted_emo'
os.makedirs(save_dir, exist_ok=True)

try:
    with open("code/models/emotiondetector.json", "r") as json_file:  # Changed file name extension
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights("code/models/emotiondetector.h5")
except Exception as e:
    print("Failed to load model:", e)
    exit(1)
# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    feature = np.array(image_resized)
    feature = feature.reshape(1, 224, 224, 3)
    return feature / 255.0

# Load and resize emotion images
emotion_images = {
    'angry': cv2.resize(cv2.imread('code/emojies/angery.jpg'), (50, 50)),
    'disgust': cv2.resize(cv2.imread('code/emojies/digusted.png'), (50, 50)),
    'fear': cv2.resize(cv2.imread('code/emojies/fear.jpg'), (50, 50)),
    'happy': cv2.resize(cv2.imread('code/emojies/happy.jpg'), (50, 50)),
    'neutral': cv2.resize(cv2.imread('code/emojies/nuteral.png'), (50, 50)),
    'sad': cv2.resize(cv2.imread('code/emojies/sad.jpg'), (50, 50)),
    'surprise': cv2.resize(cv2.imread('code/emojies/surperized.jpg'), (50, 50))
}

# Map labels to the keys used in emotion_images
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Color dictionary for text and background
emotion_colors = {
    'angry': ((0, 0, 255), (255, 255, 255)),  # Red text on white background
    'disgust': ((0, 100, 0), (255, 255, 255)),  # Dark Green on white
    'fear': ((128, 0, 128), (255, 255, 255)),  # Purple on white
    'happy': ((0, 255, 255), (0, 0, 0)),  # Yellow on black
    'neutral': ((255, 255, 255), (0, 0, 0)),  # White on black
    'sad': ((255, 0, 0), (255, 255, 255)),  # Blue on white
    'surprise': ((0, 255, 0), (0, 0, 0))  # Lime Green on black
}

try:
    # Open the video file or capture device
    webcam = cv2.VideoCapture('code/test_videos/all_emo.mp4')
    if not webcam.isOpened():
        raise ValueError("Unable to open video source")
except Exception as e:
    print("Failed to open video source:", e)
    exit(1)

# Get the first monitor's size
monitor = get_monitors()[0]
window_width = int(monitor.width / 2)
window_height = int(monitor.height / 2)

# Initialize windows
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Detection", window_width, window_height)
saved_snapshots = set()

while True:
    ret, im = webcam.read()
    if not ret:
        break  # Exit the loop if no frame is read (end of video)

    im_display = im.copy()  # Copy of the frame for displaying detected faces
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Initialize prediction_label before the loop
    prediction_label = None
    # Prepare the side window for the score graph
    score_img = np.zeros((im.shape[0], 300, 3), dtype="uint8")

    for (p, q, r, s) in faces:
        face_img = gray[q:q + s, p:p + r]
        cv2.rectangle(im_display, (p, q), (p + r, q + s), (255, 0, 0), 2)
        img_features = extract_features(face_img)
        preds = model.predict(img_features)
        emotion_id = preds.argmax()
        prediction_label = labels[emotion_id]
        text_color, bg_color = emotion_colors[prediction_label]
        emotion_img = emotion_images[prediction_label]

        # Position for emotion image and text
        img_x = p + r + 5
        img_y = q
        text_x = img_x
        text_y = img_y + 60  # Slightly below the emoji image

        # Draw emotion image
        im_display[img_y:img_y + 50, img_x:img_x + 50] = emotion_img

        # Calculate text size for background
        (text_width, text_height), _ = cv2.getTextSize(prediction_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Draw background rectangle for text
        cv2.rectangle(im_display, (text_x, text_y - text_height - 3), (text_x + text_width, text_y + 3), bg_color, -1)
        # Display the emotion name below the image
        cv2.putText(im_display, prediction_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Display score bar graph and labels
        for i, score in enumerate(preds[0]):
            label = labels[i]
            color = (int(score * 255), 200 - int(score * 200), 0)
            cv2.rectangle(score_img, (10, int(i * (score_img.shape[0] / 7) + 10)),
                          (int(score * 290) + 10, int(i * (score_img.shape[0] / 7) + 30)), color, -1)
            cv2.putText(score_img, f'{label}: {score:.2f}', (10, int(i * (score_img.shape[0] / 7) + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combine the main display and the score bar graph side window
    combined_display = np.hstack((im_display, score_img))
    cv2.imshow("Emotion Detection", combined_display)

    # Save a snapshot for each emotion only once
    if prediction_label not in saved_snapshots:
        snapshot_path = os.path.join(save_dir, f'combined_{prediction_label}.jpg')
        cv2.imwrite(snapshot_path, combined_display)
        saved_snapshots.add(prediction_label)
        print(f'Combined snapshot saved for emotion: {prediction_label} at {snapshot_path}')

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
webcam.release()
'''

#>>>>>>>>>>>>>>>>>>>>for quantize model


import cv2
import numpy as np
import tensorflow as tf
from screeninfo import get_monitors
import os

# Ensure the directory exists
save_dir = 'web/images/predicted_emo'
os.makedirs(save_dir, exist_ok=True)

# Load TFLite model and allocate tensors
try:
    interpreter = tf.lite.Interpreter(model_path="code/models/emotiondetector_optimized.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print("Failed to load TFLite model:", e)
    exit(1)

# Load Haar Cascade for face detection
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

def extract_features(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image_resized = cv2.resize(image_rgb, (224, 224))
    feature = np.array(image_resized, dtype=np.float32)  # Ensure the array is of type float32
    feature = feature.reshape(1, 224, 224, 3)
    return feature / 255.0

# Load and resize emotion images
emotion_images = {
    'angry': cv2.resize(cv2.imread('code/emojies/angery.jpg'), (50, 50)),
    'disgust': cv2.resize(cv2.imread('code/emojies/digusted.png'), (50, 50)),
    'fear': cv2.resize(cv2.imread('code/emojies/fear.jpg'), (50, 50)),
    'happy': cv2.resize(cv2.imread('code/emojies/happy.jpg'), (50, 50)),
    'neutral': cv2.resize(cv2.imread('code/emojies/nuteral.png'), (50, 50)),
    'sad': cv2.resize(cv2.imread('code/emojies/sad.jpg'), (50, 50)),
    'surprise': cv2.resize(cv2.imread('code/emojies/surperized.jpg'), (50, 50))
}

# Map labels to the keys used in emotion_images
labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}

# Color dictionary for text and background
emotion_colors = {
    'angry': ((0, 0, 255), (255, 255, 255)),  # Red text on white background
    'disgust': ((0, 100, 0), (255, 255, 255)),  # Dark Green on white
    'fear': ((128, 0, 128), (255, 255, 255)),  # Purple on white
    'happy': ((0, 255, 255), (0, 0, 0)),  # Yellow on black
    'neutral': ((255, 255, 255), (0, 0, 0)),  # White on black
    'sad': ((255, 0, 0), (255, 255, 255)),  # Blue on white
    'surprise': ((0, 255, 0), (0, 0, 0))  # Lime Green on black
}

try:
    # Open the video file or capture device
    webcam = cv2.VideoCapture('code/test_videos/all_emo.mp4')
    if not webcam.isOpened():
        raise ValueError("Unable to open video source")
except Exception as e:
    print("Failed to open video source:", e)
    exit(1)

# Get the first monitor's size
monitor = get_monitors()[0]
window_width = int(monitor.width / 2)
window_height = int(monitor.height / 2)

# Initialize windows
cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Emotion Detection", window_width, window_height)
saved_snapshots = set()

while True:
    ret, im = webcam.read()
    if not ret:
        break  # Exit the loop if no frame is read (end of video)

    im_display = im.copy()  # Copy of the frame for displaying detected faces
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Initialize prediction_label before the loop
    prediction_label = None
    # Prepare the side window for the score graph
    score_img = np.zeros((im.shape[0], 300, 3), dtype="uint8")

    for (p, q, r, s) in faces:
        face_img = gray[q:q + s, p:p + r]
        cv2.rectangle(im_display, (p, q), (p + r, q + s), (255, 0, 0), 2)
        img_features = extract_features(face_img)

        # Set the tensor to point to the input data to be inferred
        interpreter.set_tensor(input_details[0]['index'], img_features)
        interpreter.invoke()
        
        # The function `get_tensor()` returns a copy of the tensor data
        preds = interpreter.get_tensor(output_details[0]['index'])
        emotion_id = preds.argmax()
        prediction_label = labels[emotion_id]
        text_color, bg_color = emotion_colors[prediction_label]
        emotion_img = emotion_images[prediction_label]

        # Position for emotion image and text
        img_x = p + r + 5
        img_y = q
        text_x = img_x
        text_y = img_y + 60  # Slightly below the emoji image

        # Draw emotion image
        im_display[img_y:img_y + 50, img_x:img_x + 50] = emotion_img

        # Calculate text size for background
        (text_width, text_height), _ = cv2.getTextSize(prediction_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        # Draw background rectangle for text
        cv2.rectangle(im_display, (text_x, text_y - text_height - 3), (text_x + text_width, text_y + 3), bg_color, -1)
        # Display the emotion name below the image
        cv2.putText(im_display, prediction_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

        # Display score bar graph and labels
        for i, score in enumerate(preds[0]):
            label = labels[i]
            color = (int(score * 255), 200 - int(score * 200), 0)
            cv2.rectangle(score_img, (10, int(i * (score_img.shape[0] / 7) + 10)),
                          (int(score * 290) + 10, int(i * (score_img.shape[0] / 7) + 30)), color, -1)
            cv2.putText(score_img, f'{label}: {score:.2f}', (10, int(i * (score_img.shape[0] / 7) + 25)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Combine the main display and the score bar graph side window
    combined_display = np.hstack((im_display, score_img))
    cv2.imshow("Emotion Detection", combined_display)

    # Save a snapshot for each emotion only once
    if prediction_label not in saved_snapshots:
        snapshot_path = os.path.join(save_dir, f'combined_{prediction_label}.jpg')
        cv2.imwrite(snapshot_path, combined_display)
        saved_snapshots.add(prediction_label)
        print(f'Combined snapshot saved for emotion: {prediction_label} at {snapshot_path}')

    if cv2.waitKey(1) == 27:  # ESC key to exit
        break

cv2.destroyAllWindows()
webcam.release()
