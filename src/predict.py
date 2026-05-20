import os
import tensorflow as tf
import shutil

def convert_to_onnx(model_path="models/best_bird_model.keras", output_onnx_path="app/model/bird_id.onnx"):
    print("staring keras folder to onnx...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model file not found at {model_path}. please run src/train.py first.")
    
    model = tf.keras.models.load_model(model_path)
    
    tspecs = [tf.TensorSpec(i.shape, dtype=i.dtype, name="input") for i in model.inputs]
    
    temp_export_dir = "models/temp_exported_tf"
    print("exporting production graph to temporary saved_model directory...")
    if os.path.exists(temp_export_dir):
        shutil.rmtree(temp_export_dir)
        
    model.export(temp_export_dir)
    
    os.makedirs(os.path.dirname(output_onnx_path), exist_ok=True)
    
    print("encoding neural weights layout into ONNX format...")
    terminal_command = f"python -m tf2onnx.convert --saved-model {temp_export_dir} --opset 18 --output {output_onnx_path}"
    
    exit_status = os.system(terminal_command)
    
    if os.path.exists(temp_export_dir):
        shutil.rmtree(temp_export_dir)
        
    if exit_status == 0:
        print(f"\n success!")
    else:
        print(f"\n oopsies!")
        
if __name__ == "__main__":
    convert_to_onnx()