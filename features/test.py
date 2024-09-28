from twilio.rest import Client
import streamlit as st

# Define la ubicación de prueba
location = 'Street Consentida 256, Nuevo Leon, Mexico'

# Configura la clave de acceso y el token de autenticación usando st.secrets
account_sid = st.secrets["twilio"]["AS_KEY"]
auth_token = st.secrets["twilio"]["AT_KEY"]

# Uso de las credenciales para acceder a Twilio
client = Client(account_sid, auth_token)

# Mensaje de prueba
test_message = (
    f"ALERT: A person is lying down at {location}. "
    f"Check immediately for potential loss of consciousness. The location is {location}. "
    "This could indicate an emergency or health issue. Please respond promptly."
)

# Envío de un mensaje SMS de prueba
message = client.messages.create(
    messaging_service_sid='MG43c9a0f105c733f74b287754cf9ad978',
    body=test_message,
    to='+528124363149'
)

# Muestra en pantalla que el mensaje ha sido enviado con éxito
st.write(f"Test Message sent with SID: {message.sid}")
