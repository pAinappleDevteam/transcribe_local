import os
import time

import streamlit as st
from streamlit import session_state as state

from audiorecorder import audiorecorder
import whisper



#### SETUP
model = 'large-v3'
device = 'cuda'  # or `cpu`


@st.cache_resource
def get_whispermodel(model, device):
    return whisper.load_model(model, device = device)


if 'session' not in state:
    state['session'] = int(time.time() * 1e7)

session_dir = f'data/transcribe/sessions/{state["session"]}'
os.makedirs(session_dir, exist_ok = True)


#### APP
st.title('pAinapple Transcribe :microphone:')
st.markdown('')
lang = st.text_input('Language', value='nl', help='2 letter language codes, like `en` or `nl`')

#### RECORD USER AUDIO
audio = audiorecorder("Start", "Stop")

#### DISPLAY STATUS TEXT
status = st.empty()

#### CHAT INPUT
audio_input_file  = f'''{session_dir}/message.mp3'''

#### STOP FLOW
if not len(audio) > 0:
    st.stop()

start_time = time.time()

#### PROCESS USER AUDIO
status.warning('determining what you meant...')
audio.export(audio_input_file, format="mp3")

whispermodel = get_whispermodel(model, device)
result = whispermodel.transcribe(audio_input_file, language = lang)
transcript = result['text']

st.header('Transcript :keyboard:')
st.markdown(transcript)

# saving transcript
with open(f'{session_dir}/transcript.md', 'w', encoding = 'utf-8') as f:
    f.write(transcript)

end_time = time.time()

st.info(f'it took {round(end_time-start_time, 2)} seconds to compute {round(audio.duration_seconds, 2)} seconds of audio')

#### Cleanup
status.warning('cleaning up audio file, keeping transcript')
status.empty()
