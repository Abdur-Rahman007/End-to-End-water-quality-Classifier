import gradio as gr
import pandas as pd
import numpy as np
import pickle


# load the model

with open("water_rf_pipeline.pkl", "rb") as file:
    model = pickle.load(file)

# main logic
def predict_water_quality(ph, Hardness, Solids, Chloramines,
                          Sulfate, Conductivity, Organic_carbon,
                          Trihalomethanes, Turbidity):
    input_df = pd.DataFrame([[
        ph, Hardness, Solids, Chloramines,
        Sulfate, Conductivity, Organic_carbon,
        Trihalomethanes, Turbidity
    ]],
      columns=[
        'ph', 'Hardness', 'Solids', 'Chloramines',
        'Sulfate', 'Conductivity', 'Organic_carbon',
        'Trihalomethanes', 'Turbidity'
    ])
    
    # prediction
    prediction = model.predict(input_df)[0]
    
    return f"Predicted Water quality: {np.clip(prediction, 0, 1):.2f}"


inputs = [
    gr.Slider(0.0, 14.0, value=7.0, step=0.01, label="ph"),
    gr.Slider(0.0, 350.0, value=200.0, step=0.1, label="Hardness"),
    gr.Number(value=22000.0, label="Solids"),
    gr.Slider(0.0, 15.0, value=7.0, step=0.01, label="Chloramines"),
    gr.Number(value=330.0, label="Sulfate"),
    gr.Number(value=420.0, label="Conductivity"),
    gr.Slider(0.0, 30.0, value=14.0, step=0.01, label="Organic_carbon"),
    gr.Number(value=66.0, label="Trihalomethanes"),
    gr.Slider(0.0, 10.0, value=4.0, step=0.01, label="Turbidity")
]
# interface

app = gr.Interface(
    fn=predict_water_quality,
    inputs=inputs,
    outputs="text", 
    title="Water Quality Predictor")

# launch
app.launch(share=True)