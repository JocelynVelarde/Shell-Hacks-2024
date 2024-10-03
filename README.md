
# EmergencyAct: Creating a Safer World for All 

Created for ShellHacks 2024 🦖🌴

**1st Place Winners in @Waymo challenge**

## Authors

- [@JocelynVelarde](https://github.com/JocelynVelarde)
- [@htovarm7](https://github.com/htovarm7)
- [@JLDominguezM](https://github.com/JLDominguezM)
- [@morozovdd](https://github.com/morozovdd)

# Features
EmergencyAct uses surveillance cameras to detect accidents and emergencies in real-time, providing essential insights and support.

### Main Infrastructure Features
#### Computer Vision Tools
- **Object Detection:** Detects a variety of people in an area.
- **Pose Estimation:** Identifies if a person is in a dangerous position.
- **Privacy Preservation:** Maintains privacy by using tags without saving biometric data.
- **Pose Coordinate Change:** Utilizes LLM to determine the position of individuals.
- **Accident Prevention Recommendations:** Provides suggestions to enhance safety.

#### Generative AI Tools
- **Timestamp Detection:** Uses Vision AI to detect the time of a fall.
- **Relative Position Analysis:** Determines the accident location relative to the individual.
- **Cause Analysis:** Analyzes potential causes based on timestamps.
- **Emergency Call Prompting:** Crafts messages to call emergency services using text-to-speech.
- **Dashboard Summary:** Displays a summary of accidents and emergencies.

We combine various tools to enhance safety in your spaces.

## Tools Used
- 📌 **Computer Vision:** OpenCV
- 📌 **Natural Language Processing:** GPT-3.5 Turbo, prompt classification
- 📌 **Generative AI:** GPT-4-o, base64 image encoding
- 📌 **Data Visualization:** Plotly
- 📌 **Model Vision** YOLOv8mpose, ultralytics, pytorch
- 📌 **Machine Learning** Random Forest (97% test set accuracy), MLP (97% test set accuracy)
- 📌 **libraries** streamlit, CV2, ultralytics, numpy, base64, os, requests, openai, collections, datetime, torch, pickle, sklearn, joblib, pandas, numpy, fastapi, shutil, pymongo, urllib, aiohttp, json, PIL, gridfs, BASE64.

## Structure
```bash
streamlit_app 
├─ home.py
├─ .streamlit
│   └─ secrets.toml
├─ algorithms
│  └─ BB_prompt.py
│  └─ gpt_vision.py
├─ api
│  └─ main.py
├─ assets
│  └─ images
├─ features
│  └─ call_or_sms.py
│  └─ prompt.py
├─ input
│  └─ input_video.avi
│  └─ input1.avi
├─ pages
│  └─ Accident location.py
│  └─ Cause of accident.py
│  └─ Emergency call.py
│  └─ instructions.py
│  └─ Video analysis.py
```

Deployed with: Streamlit Cloud

  ## Docs

[DevPost](https://devpost.com/software/emergencyact)


## License

[MIT](https://choosealicense.com/licenses/mit/)
