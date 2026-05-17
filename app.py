import streamlit as st
import torch
from PIL import Image
from pathlib import Path

from model.model_loader import load_model
from src.preprocessing import get_inference_transform
from src.predict import predict_image
from src.gradcam import generate_gradcam


st.set_page_config(
    page_title="Age & Gender Prediction Demo",
    page_icon="🧠",
    layout="wide"
)


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "model" / "multitask_resnet34.pth"


@st.cache_resource
def load_cached_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, gender_names, age_group_names = load_model(MODEL_PATH, device)
    transform = get_inference_transform()

    return model, gender_names, age_group_names, transform, device


def format_confidence(confidence):
    return f"{confidence * 100:.2f}%"


st.title("Explainable Age and Gender Prediction")

st.write(
    "Upload a face image to predict gender and age group using a multi-task ResNet34 model. "
    "Grad-CAM heatmaps show which regions influenced the model's prediction."
)

model, gender_names, age_group_names, transform, device = load_cached_model()

st.sidebar.header("Model Information")
st.sidebar.write("Architecture: Multi-task ResNet34")
st.sidebar.write("Framework: PyTorch")
st.sidebar.write("Explainability: Grad-CAM")
st.sidebar.write(f"Device: {device}")

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, width=350)

    if st.button("Run Prediction"):
        with st.spinner("Running prediction..."):
            result = predict_image(
                image=image,
                model=model,
                transform=transform,
                device=device,
                gender_names=gender_names,
                age_group_names=age_group_names
            )

        st.subheader("Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                label="Predicted Gender",
                value=result["predicted_gender"],
                delta=format_confidence(result["gender_confidence"])
            )

        with col2:
            st.metric(
                label="Predicted Age Group",
                value=result["predicted_age_group"],
                delta=format_confidence(result["age_confidence"])
            )

        st.subheader("Grad-CAM Explainability")

        with st.spinner("Generating Grad-CAM heatmaps..."):
            gender_overlay, age_overlay = generate_gradcam(
                image=image,
                model=model,
                transform=transform,
                device=device
            )

        cam_col1, cam_col2 = st.columns(2)

        with cam_col1:
            st.image(
                gender_overlay,
                caption="Gender Grad-CAM Heatmap",
                use_container_width=True
            )

        with cam_col2:
            st.image(
                age_overlay,
                caption="Age Group Grad-CAM Heatmap",
                use_container_width=True
            )

        st.info(
            "Grad-CAM highlights the image regions that influenced the model prediction. "
            "It does not prove correctness; it helps inspect model attention."
        )

else:
    st.warning("Upload an image to start prediction.")
