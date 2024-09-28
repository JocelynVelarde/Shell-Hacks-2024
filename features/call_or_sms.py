from twilio.rest import Client
import streamlit as st

def sms(type_of_movement){
    account_sid = st.secrets["twilio"]["AS_KEY"]
    auth_token = st.secrets["twilio"]["AT_KEY"]

    # Using the credentials to access twilio
    client = Client(account_sid, auth_token)

   # dictionary for the messsages
    movement_messages = {
        'fell': (
            f"URGENT: A person has fallen at {location}. Please attend immediately. "
            f"The location is {location}. Possible risk of injury or unconsciousness. Needs urgent assistance!"
        ),
        'lying down': (
            f"ALERT: A person is lying down at {location}. "
            f"Check immediately for potential loss of consciousness. The location is {location}. "
            "This could indicate an emergency or health issue. Please respond promptly."
        )
    }

    # Veryfing the type of movement detected
    if type_of_movement in movement_messages:
        # Generate the message based on the type of movement detected
        message_body = movement_messages[type_of_movement]
        message = client.messages.create(
            messaging_service_sid='MG43c9a0f105c733f74b287754cf9ad978',
            body=message_body,
            to='+528124363149'
        )
        st.write(f"Message sent with SID: {message.sid}")

    # En caso de que se detecte sangre, se realiza una llamada con un mensaje crítico
    elif person_in_danger == 'blood detected':
        call_message = (
            f"EMERGENCY: Blood detected around the person at {location}. This is not a drill. "
            f"Immediate assistance is required at {location}. Repeat, blood detected. "
            "Act immediately. This message is critical, not a joke."
        )
        call = client.calls.create(
            twiml=f'<Response><Say>{call_message}</Say></Response>',
            to='+528124363149',
            from_='+15017122661'
        )
        st.write(f"Call initiated with SID: {call.sid}")

    else:
        st.write("No critical movement or danger detected.")
}