from twilio.rest import Client
import streamlit as st
import sqlite3

def sms(type_of_movement){
    # The location will be get by the camera information
    def get_location_from_db():
        # Connect to the database
        conn = sqlite3.connect('your_database.db')
        cursor = conn.cursor()
        
        # Execute a query to fetch the location
        cursor.execute("SELECT location FROM your_table WHERE condition")
        location = cursor.fetchone()[0]
        
        # Close the connection
        conn.close()
        
        return location
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
    
    elif person_in_danger == 'blood detected':
        call = client.calls.create(
            twiml='<Response><Say>Blood detected around the person, immediate assistance required.</Say></Response>',
            to='+528124363149',
            from_='+15017122661'
        )
        st.write(f"Call initiated with SID: {call.sid}")
}