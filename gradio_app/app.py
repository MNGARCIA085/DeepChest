import gradio as gr
from pathlib import Path
from PIL import Image
import json

# Import your existing code
from deep_chest.inference.builder import (
    build_pipeline_from_export,  # or build_pipeline_from_run
    InferencePipeline
)


# Load your pipeline (choose one method)
export_path = Path("test_model")
pipeline = build_pipeline_from_export(export_path)


class_names = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']



def predict_image(img):
    preds = pipeline.predict(img)
    probs = preds.squeeze().tolist()
    
    table = [[cls, float(p)] for cls, p in zip(class_names, probs)]
    
    threshold = 0.5
    positives = [
        cls for cls, p in zip(class_names, probs)
        if p > threshold
    ]
    
    findings_text = ", ".join(positives) if positives else "No findings above threshold."
    
    return table, findings_text


    
# Build Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs = [
        gr.Dataframe(headers=["Condition", "Probability"]),
        gr.Textbox(label="Findings Above Threshold")
    ],
    title="Image Prediction",
    description="Upload an image and get predictions"
)

if __name__ == "__main__":
    iface.launch(debug=True)

# python gradio_app/app.py
# http://127.0.0.1:7860/
