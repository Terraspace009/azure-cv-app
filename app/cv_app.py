import streamlit as st
import io
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image

# Set page config (must be first Streamlit command)
st.set_page_config(page_title="Azure Computer Vision", layout="centered")

# Azure credentials
subscription_key = "7VlwoGXzVBcBkzAwC6Xbzz9tpclZ0z8cq9KtTc3zP4j42FaMU4O9JQQJ99BEACYeBjFXJ3w3AAAFACOGQf0T"
endpoint = "https://my-computervision-app.cognitiveservices.azure.com/"
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# App title and upload UI
st.title("🧠 Azure Computer Vision App")
st.write("Upload an image to get description, tags, and object detection from Azure's AI.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image bytes once, and reuse
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image_stream = io.BytesIO(image_bytes)
    image_stream.seek(0)

    with st.spinner("Analyzing image with Azure..."):
        try:
            result = client.analyze_image_in_stream(
                image_stream,
                visual_features=[
                    VisualFeatureTypes.description,
                    VisualFeatureTypes.tags,
                    VisualFeatureTypes.objects
                ]
            )
        except Exception as e:
            st.error(f"❌ Azure API failed: {str(e)}")
        else:
            # Description
            st.subheader("📝 Description:")
            if result.description and result.description.captions:
                for caption in result.description.captions:
                    st.write(f"**{caption.text}** (confidence: {caption.confidence:.2f})")
            else:
                st.warning("No description generated.")

            # Tags
            st.subheader("🏷️ Tags:")
            if result.tags:
                tags_list = [f"{tag.name} ({tag.confidence:.2f})" for tag in result.tags]
                st.write(", ".join(tags_list))
            else:
                st.write("No tags found.")

            # Objects
            st.subheader("📦 Detected Objects:")
            if result.objects:
                for obj in result.objects:
                    rect = obj.rectangle
                    st.write(f"**{obj.object_property}** at "
                             f"({rect.x}, {rect.y}, {rect.w}, {rect.h}) — "
                             f"confidence: {obj.confidence:.2f}")
            else:
                st.write("No objects detected.")
