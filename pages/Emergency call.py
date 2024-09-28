import streamlit as st

st.set_page_config(
    page_title="EmergencyAct",
    page_icon="🚨",
)
st.image("assets/images/emergency.png", use_column_width=True)

st.title(':orange[Emergency call 🚨]')

st.write("Fetch the list of emergency calls, their status, description and location")

st.divider()

st.subheader(":orange[Some graphs to visualize the data]")