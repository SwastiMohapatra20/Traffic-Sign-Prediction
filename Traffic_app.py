from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image

# Import TensorFlow and Keras components
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout

app = Flask(__name__)

# --- Function to Define the Model Architecture ---
def create_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=(30, 30, 3)))
    model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(rate=0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(43, activation='softmax'))
    return model

# --- Create the Model and Load the Weights ---
print("Creating Keras model structure...")
model = create_model()

model_weights_path = r'model/TSR.h5' 
print(f"Loading weights from {model_weights_path}...")
try:
    model.load_weights(model_weights_path)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"Error loading model weights: {e}")
    model = None

# --- Configure Upload Folder ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Class Definitions ---
classes = { 0:'Speed limit (20km/h)', 1:'Speed limit (30km/h)', 2:'Speed limit (50km/h)', 3:'Speed limit (60km/h)', 4:'Speed limit (70km/h)', 5:'Speed limit (80km/h)', 6:'End of speed limit (80km/h)', 7:'Speed limit (100km/h)', 8:'Speed limit (120km/h)', 9:'No passing', 10:'No passing veh over 3.5 tons', 11:'Right-of-way at intersection', 12:'Priority road', 13:'Yield', 14:'Stop', 15:'No vehicles', 16:'Vehicle > 3.5 tons prohibited', 17:'No entry', 18:'General caution', 19:'Dangerous curve left', 20:'Dangerous curve right', 21:'Double curve', 22:'Bumpy road', 23:'Slippery road', 24:'Road narrows on the right', 25:'Road work', 26:'Traffic signals', 27:'Pedestrians', 28:'Children crossing', 29:'Bicycles crossing', 30:'Beware of ice/snow', 31:'Wild animals crossing', 32:'End speed + passing limits', 33:'Turn right ahead', 34:'Turn left ahead', 35:'Ahead only', 36:'Go straight or right', 37:'Go straight or left', 38:'Keep right', 39:'Keep left', 40:'Roundabout mandatory', 41:'End of no passing', 42:'End no passing vehicle > 3.5 tons' }

# --- Image Processing Function ---
def image_processing(img_path):
    if model is None: return None
    data = []
    image = Image.open(img_path).convert('RGB').resize((30, 30))
    normalized_image = np.array(image) / 255.0
    data.append(normalized_image)
    X_test = np.array(data)
    predictions_probabilities = model.predict(X_test)
    Y_pred = np.argmax(predictions_probabilities, axis=1)
    return Y_pred

# --- Flask Routes ---
@app.route('/')
def index():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['GET', 'POST']) # Allow both GET and POST
def upload():
    if request.method == 'POST':
        if model is None:
            return render_template('index.html', prediction_text="Model is not loaded. Please check server logs.")
        
        # Check if a file was submitted
        if 'file' not in request.files or request.files['file'].filename == '':
            # If no file is selected, redirect back to the home page
            return redirect(url_for('index'))
        
        f = request.files['file']
        
        if f: # Redundant check, but safe
            filename = secure_filename(f.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            f.save(file_path)
            
            result_array = image_processing(file_path)
            os.remove(file_path) # Clean up the file
            
            if result_array is None:
                return render_template('index.html', prediction_text="Could not process image.")
                
            predicted_class_id = result_array[0]
            result_text = "Predicted Traffic Sign is: " + classes[predicted_class_id]
            return render_template('index.html', prediction_text=result_text)

    # --- THE CRITICAL FIX ---
    # This will catch all GET requests to /predict and any other edge cases
    # and send the user back to the main page.
    return redirect(url_for('index'))

# --- This is the essential block to start the server ---
if __name__ == '__main__':
    app.run(debug=True)