from twilio.rest import Client
import streamlit as st

def sms(type_of_movement){
    # The location will be get by the camera information
    location = 'Street Consentida 256, Nuevo Leon, Mexico'
    account_sid = st.secrets["twilio"]["AS_KEY"]
    auth_token = st.secrets["twilio"]["AT_KEY"]

    # Using the credentials to access twilio
    client = Client(account_sid, auth_token)

    # Sending the message depending o the case
    if type_of_movement == 'fell':{
        message = client.messages.create(
            messaging_service_sid = 'MG43c9a0f105c733f74b287754cf9ad978',
            body = f'Someone fell at {location}, needs assistance ASAP',
            to = '+528124363149'
            # Message sended!
            st.write(f"Message sent with SID: {message.sid}")
        )

    }elif type_of_movement == 'lying down':{
        message = client.messages.create(
            messaging_service_sid = 'MG43c9a0f105c733f74b287754cf9ad978',
            body = f'Someone is lying down at {location}, needs assistance ASAP',
            to = '+528124363149'
            # Message sended!
            st.write(f"Message sent with SID: {message.sid}")  
        )
    
    }
}