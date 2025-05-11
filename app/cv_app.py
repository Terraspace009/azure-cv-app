import streamlit as st
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import VisualFeatureTypes
from msrest.authentication import CognitiveServicesCredentials
from PIL import Image
import io

st.set_page_config(page_title="Dreamy Azure Vision", layout="centered")

# Azure credentials
subscription_key = "7VlwoGXzVBcBkzAwC6Xbzz9tpclZ0z8cq9KtTc3zP4j42FaMU4O9JQQJ99BEACYeBjFXJ3w3AAAFACOGQf0T"
endpoint = "https://my-computervision-app.cognitiveservices.azure.com/"
client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(subscription_key))

# 🌸 Page setup
st.set_page_config(page_title="Dreamy Azure Vision", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #A78BFA;'>🖼️ Azure Computer Vision App</h1>
    <p style='text-align: center; color: #71717A; font-size: 18px;'>
        Upload an image and explore what Azure sees — captions, tags, objects & even text ✨
    </p>
    <hr style='border-top: 1px solid #ccc;'/>
""", unsafe_allow_html=True)

# 📤 Image uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, caption="🌸 Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image... 🧠"):
        # Perform analysis with all features
        analysis = client.analyze_image_in_stream(
            io.BytesIO(image_bytes),
            visual_features=[
                VisualFeatureTypes.description,
                VisualFeatureTypes.tags,
                VisualFeatureTypes.objects
            ]
        )

        # OCR (text detection)
        ocr_result = client.read_in_stream(io.BytesIO(image_bytes), raw=True)
        operation_location = ocr_result.headers["Operation-Location"]
        operation_id = operation_location.split("/")[-1]
        import time
        while True:
            result = client.get_read_result(operation_id)
            if result.status not in ['notStarted', 'running']:
                break
            time.sleep(0.5)

    # 🗂 Tabs for features
    tabs = st.tabs(["📝 Description", "🏷️ Tags", "📦 Objects", "🔤 Text (OCR)"])

    # Description
    with tabs[0]:
        st.subheader("📝 Image Description")
        if analysis.description and analysis.description.captions:
            for caption in analysis.description.captions:
                st.markdown(f"**{caption.text.capitalize()}** ({caption.confidence:.2f} confidence)")
        else:
            st.warning("No description found.")

    # Tags
    with tabs[1]:
        st.subheader("🏷️ Tags")
        if analysis.tags:
            tag_text = ", ".join(f"`{tag.name}` ({tag.confidence:.2f})" for tag in analysis.tags)
            st.markdown(tag_text)
        else:
            st.info("No tags detected.")

    # Objects
    with tabs[2]:
        st.subheader("📦 Detected Objects")
        if analysis.objects:
            for obj in analysis.objects:
                r = obj.rectangle
                st.markdown(f"**{obj.object_property}** — ({r.x}, {r.y}, {r.w}, {r.h}) "
                            f"– confidence: {obj.confidence:.2f}")
        else:
            st.info("No objects detected.")

    # OCR
    with tabs[3]:
        st.subheader("🔤 Text (OCR)")
        if result.status == 'succeeded':
            for page in result.analyze_result.read_results:
                for line in page.lines:
                    st.markdown(f"📝 {line.text}")
        else:
            st.warning("No text found.")

# Footer
st.markdown("""
    <hr style='border-top: 1px solid #eee;'/>
    <p style='text-align: center; font-size: 14px; color: #bbb;'>
        Built with 💜 by <a href='https://github.com/Terraspace009' target='_blank'>Aishwarya Shukla</a>
    </p>
""", unsafe_allow_html=True)
