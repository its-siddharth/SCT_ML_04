import streamlit as st
import cv2
import numpy as np
import PIL.Image as Image
from datetime import datetime
import io
import base64
import os
import tempfile

# Configure the page - MUST be first Streamlit command
st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤚",
    layout="wide",
    initial_sidebar_state="expanded"
)

class GestureRecognizer:
    def __init__(self):
        self.model = None
        self.model_loaded = False
        self.model_input_shape = None
        self.auto_detected_config = None
        
        # Default gesture classes - will be updated when model is loaded
        self.gesture_classes = [
            'thumbs_up', 'thumbs_down', 'peace_sign', 'ok_sign',
            'fist', 'open_palm', 'point_up', 'rock_on',
            'call_me', 'stop_sign'
        ]
        
    def analyze_model_input(self):
        """Analyze model input shape and suggest configuration"""
        if self.model is None:
            return None
            
        try:
            input_shape = self.model.input_shape
            self.model_input_shape = input_shape
            
            config = {
                'target_size': (224, 224),
                'flatten': False,
                'grayscale': False,
                'channels': 3
            }
            
            if len(input_shape) == 2:  # Flattened input (batch_size, features)
                features = input_shape[1]
                config['flatten'] = True
                
                # Try to determine image dimensions
                import math
                
                # Check for grayscale square images
                sqrt_features = int(math.sqrt(features))
                if sqrt_features * sqrt_features == features:
                    config['target_size'] = (sqrt_features, sqrt_features)
                    config['grayscale'] = True
                    config['channels'] = 1
                
                # Check for RGB square images
                rgb_side = int(math.sqrt(features / 3))
                if rgb_side * rgb_side * 3 == features:
                    config['target_size'] = (rgb_side, rgb_side)
                    config['grayscale'] = False
                    config['channels'] = 3
                    
            elif len(input_shape) == 4:  # Image input (batch, height, width, channels)
                _, h, w, c = input_shape
                config['target_size'] = (w, h)
                config['flatten'] = False
                config['grayscale'] = (c == 1)
                config['channels'] = c
                
            elif len(input_shape) == 3:  # Image without batch dim or with different order
                if input_shape[0] == 1 or input_shape[2] == 1:  # Grayscale
                    h, w = input_shape[1], input_shape[2] if input_shape[0] == 1 else input_shape[0], input_shape[1]
                    config['target_size'] = (w, h)
                    config['grayscale'] = True
                    config['channels'] = 1
                else:  # RGB
                    h, w, c = input_shape
                    config['target_size'] = (w, h)
                    config['grayscale'] = False
                    config['channels'] = c
            
            self.auto_detected_config = config
            return config
            
        except Exception as e:
            st.error(f"Error analyzing model: {e}")
            return None
        
    def load_model_from_file(self, model_file):
        """Load H5 model from uploaded file"""
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.h5') as tmp_file:
                tmp_file.write(model_file.getvalue())
                tmp_path = tmp_file.name
            
            # Try to load the model
            try:
                import tensorflow as tf
                self.model = tf.keras.models.load_model(tmp_path)
                self.model_loaded = True
                
                # Analyze model input shape
                self.analyze_model_input()
                
                # Clean up temporary file
                os.unlink(tmp_path)
                
                return True, "Model loaded successfully!"
                
            except Exception as e:
                # Clean up temporary file
                os.unlink(tmp_path)
                return False, f"Error loading model: {str(e)}"
                
        except Exception as e:
            return False, f"Error processing file: {str(e)}"
    
    def update_gesture_classes(self, classes_text):
        """Update gesture classes from user input"""
        try:
            # Split by comma or newline and clean up
            classes = [cls.strip() for cls in classes_text.replace('\n', ',').split(',')]
            classes = [cls for cls in classes if cls]  # Remove empty strings
            
            if classes:
                self.gesture_classes = classes
                return True, f"Updated with {len(classes)} gesture classes"
            else:
                return False, "No valid gesture classes found"
                
        except Exception as e:
            return False, f"Error updating classes: {str(e)}"
    
    def preprocess_image(self, image, target_size=None, flatten=False, grayscale=False):
        """Preprocess the image for the model"""
        try:
            # Use auto-detected config if available
            if self.auto_detected_config and target_size is None:
                target_size = self.auto_detected_config['target_size']
                flatten = self.auto_detected_config['flatten']
                grayscale = self.auto_detected_config['grayscale']
            
            # Use default size if not specified
            if target_size is None:
                target_size = (224, 224)
            
            # Convert PIL to numpy array if needed
            if isinstance(image, Image.Image):
                image = np.array(image)
            
            # Handle different input formats
            if len(image.shape) == 3 and image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) == 3 and image.shape[2] == 3:  # RGB
                pass  # Already in correct format
            
            # Convert to grayscale if needed
            if grayscale and len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif not grayscale and len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Resize image
            image_resized = cv2.resize(image, target_size)
            
            # Normalize pixel values to [0, 1]
            image_normalized = image_resized.astype(np.float32) / 255.0
            
            # Handle different output formats based on model requirements
            if flatten:
                # Flatten for dense layers
                image_batch = image_normalized.reshape(1, -1)
            else:
                # Keep spatial dimensions for CNN
                if len(image_normalized.shape) == 2:  # Grayscale
                    image_batch = np.expand_dims(image_normalized, axis=(0, -1))  # Add batch and channel dims
                else:  # RGB
                    image_batch = np.expand_dims(image_normalized, axis=0)  # Add batch dim only
            
            return image_batch
            
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            return None
    
    def predict_gesture(self, image, target_size=None, flatten=False, grayscale=False):
        """Predict hand gesture from image"""
        try:
            if not self.model_loaded or self.model is None:
                return None, 0.0, "No model loaded. Please upload a model first."
            
            # Preprocess the image
            processed_image = self.preprocess_image(image, target_size, flatten, grayscale)
            
            if processed_image is None:
                return None, 0.0, "Preprocessing failed"
            
            # Debug: Show actual input shape vs expected
            st.write(f"**Debug Info:**")
            st.write(f"- Model expects: {self.model_input_shape}")
            st.write(f"- Providing: {processed_image.shape}")
            
            # Make prediction with the loaded model
            prediction = self.model.predict(processed_image, verbose=0)
            predicted_class_idx = np.argmax(prediction[0])
            confidence = float(prediction[0][predicted_class_idx])
            
            # Make sure we don't go out of bounds
            if predicted_class_idx >= len(self.gesture_classes):
                predicted_gesture = f"class_{predicted_class_idx}"
            else:
                predicted_gesture = self.gesture_classes[predicted_class_idx]
            
            # Get all predictions for visualization
            all_predictions = {}
            for i, gesture in enumerate(self.gesture_classes):
                if i < len(prediction[0]):
                    all_predictions[gesture] = float(prediction[0][i])
                else:
                    all_predictions[gesture] = 0.0
            
            return predicted_gesture, confidence, all_predictions
            
        except Exception as e:
            return None, 0.0, f"Prediction error: {str(e)}"

# Initialize the gesture recognizer
@st.cache_resource
def get_gesture_recognizer():
    return GestureRecognizer()

gesture_recognizer = get_gesture_recognizer()

def main():
    # Title and description
    st.title("🤚 Hand Gesture Recognition")
    st.markdown("Upload your H5 model and recognize hand gestures!")
    
    # Sidebar for model configuration
    with st.sidebar:
        st.header("🔧 Model Configuration")
        
        # Model upload section
        st.subheader("1. Upload Model")
        uploaded_model = st.file_uploader(
            "Upload your H5 model file",
            type=['h5'],
            help="Upload your trained Keras/TensorFlow model (.h5 file)"
        )
        
        if uploaded_model is not None:
            if st.button("Load Model", type="primary"):
                with st.spinner("Loading model..."):
                    success, message = gesture_recognizer.load_model_from_file(uploaded_model)
                    if success:
                        st.success(message)
                        st.session_state['model_loaded'] = True
                    else:
                        st.error(message)
        
        # Model status and auto-detected configuration
        if gesture_recognizer.model_loaded:
            st.success("✅ Model Loaded")
            
            # Show model info
            if gesture_recognizer.model is not None:
                st.write("**Model Summary:**")
                try:
                    input_shape = gesture_recognizer.model.input_shape
                    output_shape = gesture_recognizer.model.output_shape
                    st.write(f"Input: {input_shape}")
                    st.write(f"Output: {output_shape}")
                    
                    # Show auto-detected configuration
                    if gesture_recognizer.auto_detected_config:
                        st.success("🤖 **Auto-Detected Configuration:**")
                        config = gesture_recognizer.auto_detected_config
                        st.write(f"- Image Size: {config['target_size']}")
                        st.write(f"- Grayscale: {config['grayscale']}")
                        st.write(f"- Flatten: {config['flatten']}")
                        st.write(f"- Channels: {config['channels']}")
                        
                        # Quick apply button
                        if st.button("🚀 Apply Auto-Config", type="primary"):
                            st.session_state['target_size'] = config['target_size']
                            st.session_state['flatten_input'] = config['flatten']
                            st.session_state['grayscale_input'] = config['grayscale']
                            st.success("✅ Configuration applied!")
                            st.rerun()
                            
                except Exception as e:
                    st.write(f"Model analysis error: {e}")
        else:
            st.warning("⚠️ No Model Loaded")
        
        st.markdown("---")
        
        # Gesture classes configuration
        st.subheader("2. Configure Gesture Classes")
        
        current_classes = "\n".join(gesture_recognizer.gesture_classes)
        gesture_classes_input = st.text_area(
            "Gesture Classes (one per line)",
            value=current_classes,
            height=200,
            help="Enter your gesture class names, one per line"
        )
        
        if st.button("Update Classes"):
            success, message = gesture_recognizer.update_gesture_classes(gesture_classes_input)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        st.write(f"**Current Classes:** {len(gesture_recognizer.gesture_classes)}")
        for i, gesture in enumerate(gesture_recognizer.gesture_classes):
            st.write(f"{i}. {gesture}")
        
        st.markdown("---")
        
        # Manual preprocessing settings (only if needed)
        with st.expander("🔧 Manual Settings (Optional)"):
            st.write("*Only use if auto-detection doesn't work*")
            
            # Input size configuration
            col1, col2 = st.columns(2)
            with col1:
                img_width = st.number_input("Image Width", min_value=32, max_value=512, value=200, step=32)
            with col2:
                img_height = st.number_input("Image Height", min_value=32, max_value=512, value=200, step=32)
            
            target_size = (img_width, img_height)
            st.session_state['manual_target_size'] = target_size
            
            # Model input format options
            st.markdown("**Input Format:**")
            flatten_input = st.checkbox("Flatten input (for Dense layers)", value=False)
            grayscale_input = st.checkbox("Convert to Grayscale", value=True)
            
            st.session_state['manual_flatten_input'] = flatten_input
            st.session_state['manual_grayscale_input'] = grayscale_input
            
            if st.button("Apply Manual Settings"):
                st.session_state['target_size'] = target_size
                st.session_state['flatten_input'] = flatten_input
                st.session_state['grayscale_input'] = grayscale_input
                st.success("Manual settings applied!")
                st.rerun()
        
        # Common model configurations
        with st.expander("📚 Quick Fixes"):
            st.markdown("**For your specific error:**")
            if st.button("Fix: 200x200 Grayscale CNN", type="secondary"):
                st.session_state['target_size'] = (200, 200)
                st.session_state['flatten_input'] = False
                st.session_state['grayscale_input'] = True
                st.success("✅ Applied: 200x200 grayscale, not flattened")
                st.rerun()
                
            st.markdown("**Other common formats:**")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("96x96 Gray Flat"):
                    st.session_state['target_size'] = (96, 96)
                    st.session_state['flatten_input'] = True
                    st.session_state['grayscale_input'] = True
                    st.rerun()
            with col2:
                if st.button("224x224 RGB CNN"):
                    st.session_state['target_size'] = (224, 224)
                    st.session_state['flatten_input'] = False
                    st.session_state['grayscale_input'] = False
                    st.rerun()
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📷 Image Input")
        
        # Check if model is loaded
        if not gesture_recognizer.model_loaded:
            st.warning("⚠️ Please upload and load your H5 model first in the sidebar!")
            st.info("Steps: 1) Upload .h5 file → 2) Click 'Load Model' → 3) Use auto-config or manual settings")
        
        # Tab layout for different input methods
        tab1, tab2 = st.tabs(["📤 Upload Image", "📷 Camera"])
        
        with tab1:
            uploaded_file = st.file_uploader(
                "Choose an image file", 
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear image showing a hand gesture"
            )
            
            if uploaded_file is not None:
                # Display uploaded image
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)
                
                # Predict button
                predict_button = st.button("🔍 Predict Gesture", type="primary", disabled=not gesture_recognizer.model_loaded)
                
                if predict_button and gesture_recognizer.model_loaded:
                    # Use auto-detected config or manual settings
                    if gesture_recognizer.auto_detected_config:
                        config = gesture_recognizer.auto_detected_config
                        target_size = config['target_size']
                        flatten = config['flatten']
                        grayscale = config['grayscale']
                        st.info(f"🤖 Using auto-detected config: {target_size}, grayscale={grayscale}, flatten={flatten}")
                    else:
                        target_size = st.session_state.get('target_size', (200, 200))
                        flatten = st.session_state.get('flatten_input', False)
                        grayscale = st.session_state.get('grayscale_input', True)
                        st.info(f"🔧 Using manual config: {target_size}, grayscale={grayscale}, flatten={flatten}")
                    
                    with st.spinner("Analyzing gesture..."):
                        gesture, confidence, predictions = gesture_recognizer.predict_gesture(
                            image, target_size, flatten, grayscale
                        )
                        
                        if gesture and not isinstance(predictions, str):
                            # Store results in session state
                            st.session_state['last_prediction'] = {
                                'gesture': gesture,
                                'confidence': confidence,
                                'predictions': predictions,
                                'timestamp': datetime.now(),
                                'image': image
                            }
                            st.success("✅ Prediction completed!")
                        else:
                            st.error(f"❌ Prediction failed: {predictions}")
        
        with tab2:
            st.write("📸 Camera Input")
            
            # Camera input
            camera_image = st.camera_input("Take a picture of your hand gesture")
            
            if camera_image is not None:
                # Convert camera input to PIL Image
                image = Image.open(camera_image)
                
                # Predict button for camera
                predict_camera_button = st.button("🔍 Predict from Camera", type="primary", disabled=not gesture_recognizer.model_loaded)
                
                if predict_camera_button and gesture_recognizer.model_loaded:
                    # Use auto-detected config or manual settings
                    if gesture_recognizer.auto_detected_config:
                        config = gesture_recognizer.auto_detected_config
                        target_size = config['target_size']
                        flatten = config['flatten']
                        grayscale = config['grayscale']
                        st.info(f"🤖 Using auto-detected config: {target_size}, grayscale={grayscale}, flatten={flatten}")
                    else:
                        target_size = st.session_state.get('target_size', (200, 200))
                        flatten = st.session_state.get('flatten_input', False)
                        grayscale = st.session_state.get('grayscale_input', True)
                        st.info(f"🔧 Using manual config: {target_size}, grayscale={grayscale}, flatten={flatten}")
                    
                    with st.spinner("Analyzing gesture..."):
                        gesture, confidence, predictions = gesture_recognizer.predict_gesture(
                            image, target_size, flatten, grayscale
                        )
                        
                        if gesture and not isinstance(predictions, str):
                            # Store results in session state
                            st.session_state['last_prediction'] = {
                                'gesture': gesture,
                                'confidence': confidence,
                                'predictions': predictions,
                                'timestamp': datetime.now(),
                                'image': image
                            }
                            st.success("✅ Prediction completed!")
                        else:
                            st.error(f"❌ Prediction failed: {predictions}")
    
    with col2:
        st.header("📊 Results")
        
        # Display results if available
        if 'last_prediction' in st.session_state:
            result = st.session_state['last_prediction']
            
            # Main prediction result
            st.markdown("### 🎯 Prediction")
            gesture_name = result['gesture'].replace('_', ' ').title()
            confidence_pct = result['confidence'] * 100
            
            # Display with colored background based on confidence
            if confidence_pct >= 80:
                st.success(f"**{gesture_name}**")
            elif confidence_pct >= 60:
                st.warning(f"**{gesture_name}**")
            else:
                st.error(f"**{gesture_name}**")
            
            # Confidence meter
            st.markdown("### 📈 Confidence")
            st.progress(result['confidence'])
            st.write(f"**{confidence_pct:.1f}%**")
            
            # All predictions chart
            if isinstance(result['predictions'], dict):
                st.markdown("### 📊 All Predictions")
                
                # Create DataFrame for chart
                import pandas as pd
                df = pd.DataFrame([
                    {'Gesture': k.replace('_', ' ').title(), 'Score': v} 
                    for k, v in result['predictions'].items()
                ])
                df = df.sort_values('Score', ascending=True)
                
                # Bar chart
                st.bar_chart(df.set_index('Gesture'))
                
                # Top 3 predictions
                top_3 = df.tail(3).iloc[::-1]  # Reverse to show highest first
                st.markdown("**Top 3 Predictions:**")
                for idx, row in top_3.iterrows():
                    st.write(f"{row['Gesture']}: {row['Score']*100:.1f}%")
            
            # Timestamp
            st.markdown("### ⏰ Analysis Time")
            st.write(result['timestamp'].strftime("%H:%M:%S"))
            
        else:
            if gesture_recognizer.model_loaded:
                st.info("👆 Upload an image or take a photo to see predictions!")
            else:
                st.warning("🔧 Configure your model first!")
            
            # Show tips
            st.markdown("### 💡 Tips for better results:")
            st.write("• Ensure good lighting")
            st.write("• Clear background")
            st.write("• Hand should be clearly visible")
            st.write("• Match your training data style")
    
    # Prediction history
    st.markdown("---")
    st.header("📋 Prediction History")
    
    # Initialize history in session state
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # Add current prediction to history
    if 'last_prediction' in st.session_state:
        last_pred = st.session_state['last_prediction']
        # Check if this prediction is already in history
        if not st.session_state['prediction_history'] or \
           st.session_state['prediction_history'][-1]['timestamp'] != last_pred['timestamp']:
            st.session_state['prediction_history'].append(last_pred)
            # Keep only last 10 predictions
            if len(st.session_state['prediction_history']) > 10:
                st.session_state['prediction_history'] = st.session_state['prediction_history'][-10:]
    
    # Display history
    if st.session_state['prediction_history']:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state['prediction_history'] = []
                st.rerun()
        
        with col1:
            st.write(f"Showing last {len(st.session_state['prediction_history'])} predictions:")
        
        # Show history in expandable format
        for i, pred in enumerate(reversed(st.session_state['prediction_history'])):
            with st.expander(
                f"#{len(st.session_state['prediction_history'])-i}: "
                f"{pred['gesture'].replace('_', ' ').title()} - "
                f"{pred['confidence']*100:.1f}% - "
                f"{pred['timestamp'].strftime('%H:%M:%S')}"
            ):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(pred['image'], caption="Input Image", width=150)
                with col2:
                    st.write(f"**Gesture:** {pred['gesture'].replace('_', ' ').title()}")
                    st.write(f"**Confidence:** {pred['confidence']*100:.1f}%")
                    st.write(f"**Time:** {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    else:
        st.info("📝 No predictions yet. Make a prediction to see history!")
    
    # Error troubleshooting guide
    with st.expander("🆘 Troubleshooting Guide"):
        st.markdown("""
        ### Common Errors and Solutions:
        
        **Error: "Expected shape (None, 200, 200, 1)"**
        - Your model expects 200x200 grayscale images
        - ✅ Click "Fix: 200x200 Grayscale CNN" button
        - Or manually set: Width=200, Height=200, Grayscale=Yes, Flatten=No
        
        **Error: "Expected shape (None, 9216)"**  
        - Your model expects flattened 96x96 grayscale images
        - ✅ Click "96x96 Gray Flat" button
        - 96 × 96 = 9,216 features
        
        **Error: "Expected shape (None, 224, 224, 3)"**
        - Your model expects 224x224 RGB images  
        - ✅ Click "224x224 RGB CNN" button
        
        ### Auto-Detection:
        - The app automatically analyzes your model
        - Look for "🤖 Auto-Detected Configuration" 
        - Click "🚀 Apply Auto-Config" to use detected settings
        
        ### Manual Override:
        - Use "🔧 Manual Settings" if auto-detection fails
        - Match your training data preprocessing exactly
        """)

if __name__ == "__main__":
    main()