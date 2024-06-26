from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import requests

app = Flask(__name__)

# Load the saved model
model = load_model('1.h5')  # Replace 'your_model.h5' with the path to your h5 file

# Define a function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize the image as required by your model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define a function to make predictions
def predict_image(img_path):
    img = preprocess_image(img_path)
    predictions = model.predict(img)
    return predictions

# Define the class labels
class_labels = ['Vitamin A', 'Vitamin B', 'Vitamin C', 'Vitamin D', 'Vitamin E', 'Vitamin K']  

# Define a function to process predictions
def process_predictions(predictions, class_labels):
    predicted_index = np.argmax(predictions)
    predicted_vitamin = class_labels[predicted_index]
    confidence_score = predictions[0][predicted_index]
    return predicted_vitamin, confidence_score

@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file uploaded'
    
    file = request.files['file']
    
    if file.filename == '':
        return 'No file selected'
    
    if file:
        # Save the uploaded image temporarily with a unique filename
        img_path = 'uploaded_image.jpg'  
        file.save(img_path)
        
        predictions = predict_image(img_path)
        
        # Process predictions
        predicted_vitamin, confidence_score = process_predictions(predictions, class_labels)
        
        # Display the result in a user-friendly format
        result1 = f'Predicted Vitamin: {predicted_vitamin}'
        result2 = f'Confidence Score: {confidence_score}'

        
        return render_template('index.html', resultp1=result1, resultp2=result2)
        @app.route('/treatments')  # Define route for treatments page
        def treatments():
        return render_template('treatments.html')

        


if __name__ == '__main__':
    app.run(debug=True)
