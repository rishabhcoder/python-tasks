import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import requests

#Function to handle image classification
def classify_image(image):
    url = 'https://api-inference.huggingface.co/models/microsoft/resnet-18'
    headers = {
        'Authorization': 'Bearer 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'
    }
    
    # Read the image as a binary stream (octet stream)
    image_bytes = image.read()

    files = {'image': ('image.jpg', image_bytes, 'image/jpeg')}
    
    try:
        # Make the API call
        response = requests.post(url, headers=headers, files=files)
        
        # Check if response status is OK
        if response.status_code == 200:
            return response.json()
        else:
            # Print the response content for debugging
            st.error(f"API request failed with status code {response.status_code}")
            st.write(response.text)  # Log the full response content for debugging
            return None
    except Exception as e:
        st.error(f"Error during API request: {str(e)}")
        return 
    

    # Streamlit app
st.title('Image Upload and Classification')

# File upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Classify the image when the user clicks the button
    if st.button('Upload and Classify'):
        st.write("Classifying image... Please wait.")
        
        # Classify image and show results
        result = classify_image(uploaded_file)
        
        if result:
            # Check if the expected keys exist in the response
            if 'class' in result and 'confidence' in result:
                prediction = result['class']
                confidence = result['confidence']
                st.success(f"Prediction: {prediction}")
                st.success(f"Confidence: {confidence}%")
            else:
                st.error("Error: Unexpected response structure.")
                st.write(result)  # Show the full response for debugging
        else:
            st.error("Error: Unable to classify the image.")

