
**How to Use This:**

Create a new file in your main project folder (Traffic_Signs_WebApp-master/) and name it README.md.



**Traffic Sign Recognition Web App**

This project is a deep learning web application that classifies traffic signs from an image. It utilizes a Convolutional Neural Network (CNN) trained on the famous German Traffic Sign Recognition Benchmark (GTSRB) dataset. The model is served through a clean, user-friendly web interface built with Flask.



**Features**

Interactive Web Interface: A simple and modern UI for uploading images.
Deep Learning Model: A powerful CNN built with Keras & TensorFlow for high-accuracy classification.
43 Traffic Sign Classes: The model can recognize 43 different types of traffic signs.
Real-time Prediction: Upload an image (JPG, PNG) and get an instant prediction.
Efficient Backend: The Flask server loads the model once at startup for fast subsequent predictions.

Tech Stack & Dataset
Backend: Python, Flask
Deep Learning: TensorFlow, Keras
Data Processing: NumPy, Pillow
Frontend: HTML, CSS
Dataset: German Traffic Sign Recognition Benchmark (GTSRB)

**Project Structure**

For the application to run correctly, your project must follow this folder structure:

Traffic_Signs_WebApp-master/
├── model/
│   └── TSR.h5              <-- Your trained Keras model file
├── templates/
│   └── index.html          <-- The HTML user interface
├── uploads/                <-- Temporary storage for uploaded images
├── Traffic_app.py          <-- The main Flask application script
└── README.md               <-- You are here!
Setup and Installation

**Follow these steps to get the application running on your local machine.**

**1. Clone the Repository**


git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

**2. Create and Activate a Virtual Environment**

It's highly recommended to use a virtual environment to keep project dependencies isolated.

Windows:


python -m venv .venv
.venv\Scripts\activate

macOS / Linux:



python3 -m venv .venv
source .venv/bin/activate

**3. Install Dependencies**

This project uses a set of Python libraries. You can install them all with a single command using the provided requirements.txt file.
(If you don't have a requirements.txt file, create one by running this command in your activated virtual environment: pip freeze > requirements.txt)


pip install -r requirements.txt
(If you don't have the file, install manually: pip install Flask numpy Pillow tensorflow werkzeug)

**4. Place the Trained Model**

Ensure that your trained Keras model, named TSR.h5, is placed inside the model/ directory.
How to Run the Application
Once the setup is complete, you can start the Flask web server.
1. Run the App
Execute the following command in your terminal from the project's root directory:

python Traffic_app.py

**2. Open in Browser**
You will see output in your terminal indicating that the server is running. The last line will look like this:
* Running on http://127.0.0.1:5000
Open your favorite web browser and navigate to this address: http://127.0.0.1:5000
You should now see the web application and be able to upload an image for prediction!

**Model Architecture**
The heart of this project is a Convolutional Neural Network (CNN) designed for image classification. The architecture is as follows:
Convolutional Layer 1: 32 filters, 5x5 kernel, ReLU activation.
Convolutional Layer 2: 32 filters, 5x5 kernel, ReLU activation.
Max Pooling Layer: 2x2 pool size.
Dropout Layer: 25% rate to prevent overfitting.
Convolutional Layer 3: 64 filters, 3x3 kernel, ReLU activation.
Convolutional Layer 4: 64 filters, 3x3 kernel, ReLU activation.
Max Pooling Layer: 2x2 pool size.
Dropout Layer: 25% rate.
Flatten Layer: To convert the 2D feature maps into a 1D vector.
Dense Layer (Fully Connected): 256 neurons, ReLU activation.
Dropout Layer: 50% rate.
Output Layer (Dense): 43 neurons (one for each class), Softmax activation.

License

This project is licensed under the MIT License. See the LICENSE file for details.
