import os
import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
import pandas as pd

# set up webpage and title
st.set_page_config(page_title="bird id", page_icon=":camera:", layout="centered")

class bird_id_app:
    def __init__(self, model_path="app/model/bird_id.onnx", metadata_path="data/processed/val.csv"):
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            self.session = None
            self.class_names = ["demo mode: run src/predict.py to generate the model"]
            st.error(f"model file missing at {model_path}!")
            return
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        
        self.session = ort.InferenceSession(
            model_path, 
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        
        # load class names from metadata
        if os.path.exists(metadata_path):
            
            df = pd.read_csv(metadata_path)
            class_mapping = df[["label", "class_name"]].drop_duplicates().sort_values("label")  
            self.class_names = class_mapping["class_name"].tolist()
        else: 
            self.class_names = [f"species index {i}" for i in range(200)]
        
    def preprocess_image(self, image_input):
        
        file_bytes = np.asarray(bytearray(image_input.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        img = cv2.resize(img, (224, 224))
        img = img.astype(np.float32)
 
        img = np.expand_dims(img, axis=0)
        return img
    
@st.cache_resource
def get_app_instance():
    return bird_id_app()

app = get_app_instance()

st.title("what yo name is?")
st.write("upload an image of a bird baddie to identify its species")
    
uploaded_file = st.file_uploader("choose an image", type=["jpg", "jpeg", "png"])
    
if uploaded_file is not None:
    uploaded_file.seek(0)
    st.image(uploaded_file, caption="uploaded image", use_container_width=True)
        
    with st.spinner("analyzing features..."):
        if app.session is None:
            st.warning("model not found. please run src/predict.py to generate the model.")
        else:
            try: 
                processed_tensor = app.preprocess_image(uploaded_file)
                    
                input_name = app.session.get_inputs()[0].name
                
                raw_outputs = app.session.run(None, {input_name: processed_tensor})
                outputs = raw_outputs[0].flatten()
                
                if np.isclose(np.sum(outputs), 1.0, atol=1e-3):
                    probabilities = outputs
                else:    
                    exp_scores = np.exp(outputs - np.max(outputs))
                    probabilities = exp_scores / np.sum(exp_scores)
                                    
                top_indices = np.argmax(probabilities)
                confidence_score = probabilities[top_indices]
                
                raw_class = app.class_names[top_indices]
                    
                if "." in raw_class:
                    predicted_class = raw_class.split(".", 1)[-1].replace("_", " ")
                else:
                    predicted_class = raw_class.replace("_", " ")
                    
                st.success(f"species: **{predicted_class}**")
                st.metric(label="we're this sure", value=f"{confidence_score * 100:.1f}%")
                              
            except Exception as e:
                st.error(f"error processing image: {str(e)}")