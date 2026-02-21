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



# Gradio prediction function
def predict_imagev0(img: Image.Image):
    """
    img: PIL image from Gradio
    """
    # Call your pipeline
    preds = pipeline.predict(img)
    
    # Format output nicely (assume dict of label:prob)
    if isinstance(preds, dict):
        return {str(k): float(v) for k, v in preds.items()}
    else:
        return str(preds)






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



"""
[[0.25179872 0.5920954  0.17985962 0.63353497 0.17521143 0.47014755
  0.6151251  0.29562163 0.5354593  0.23120105 0.30441943 0.701831
  0.1022056  0.2491951 ]]
"""




def predict_imagev1(img: Image.Image):
    """
    img: PIL image from Gradio
    """
    # Call your pipeline
    preds = pipeline.predict(img)
    print(preds)
    probs = preds.squeeze().tolist()

    return {cls: float(p) for cls, p in zip(class_names, probs)}



def predict_image(img: Image.Image):
    """
    img: PIL image from Gradio
    """
    # Call your pipeline
    preds = pipeline.predict(img)
    print(preds)
    probs = preds.squeeze().tolist()

    return [[cls, float(p)] for cls, p in zip(class_names, probs)]


    
    
# Build Gradio interface
iface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    #outputs=gr.Label(), -> with v1
    outputs = gr.Dataframe(
        headers=["Condition", "Probability"],
        datatype=["str", "number"]
    ),
    title="Image Prediction",
    description="Upload an image and get predictions"
)

if __name__ == "__main__":
    iface.launch(debug=True)

# python gradio_app/app.py
# http://127.0.0.1:7860/




"""
def predict(image):
    probs = model(image).squeeze().tolist()
    
    threshold = 0.5
    results = {
        cls: float(p)
        for cls, p in zip(class_names, probs)
        if p > threshold
    }
    
    return results if results else {"No finding above threshold": 1.0}
"""