import streamlit as st
from algorithms.gpt_vision import get_analysis_messages


api_key = st.secrets["OPEN_AI_KEY"]

# Streamlit app
st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)
st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Cause of Accident 🚨]')

st.write("Observe the photo snapshots of the accident and analyze the possible causes to obtain recommendations on what to improve")

st.divider()

uploaded_files = st.file_uploader("Upload images", accept_multiple_files=True, type=["png", "jpg", "jpeg"])

if uploaded_files:
    image_paths = []
    for uploaded_file in uploaded_files:
        with open(f"temp_{uploaded_file.name}", "wb") as f:
            f.write(uploaded_file.getbuffer())
            image_paths.append(f"temp_{uploaded_file.name}")

    messages = get_analysis_messages(image_paths, api_key)
    for i, message in enumerate(messages, 1):
        st.subheader(f"Message {i}")
        st.write(message)

st.divider()
st.subheader(':orange[Ask questions about the accident]')
chat_input = st.text_area("Type your question here")
if st.button("Send"):
    respuesta = get_analysis_messages([image_paths[0]], api_key)[0]
    st.subheader("Anser:")
    st.write(respuesta)
    