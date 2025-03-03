import streamlit as st
from streamlit_option_menu import option_menu
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
import tempfile
import os
import whisper
import torch
import joblib
import numpy as np
import librosa
import pandas as pd

# Load the dataset
df = pd.read_excel('dataset/quran.xlsx',engine='openpyxl')
# Drop rows with missing values
df.dropna(inplace=True)
# Checking for duplicates and removing them
df.drop_duplicates(inplace=True)
# Ensure the columns are correctly named
df.columns = ['juzno', 'surahno', 'qurantext']


# Initialize Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['qurantext'])
# Convert text to sequences
X = tokenizer.texts_to_sequences(df['qurantext'])
# Pad sequences to ensure uniform length
X = tf.keras.preprocessing.sequence.pad_sequences(X)


# Initialize LabelEncoder
label_encoder_juzno = LabelEncoder()
label_encoder_surahno = LabelEncoder()
# Convert juzno and surahno to numerical values
y_juzno = label_encoder_juzno.fit_transform(df['juzno'])
y_surahno = label_encoder_surahno.fit_transform(df['surahno'])


#load  models,tokenizer, and label econder 
model=joblib.load("model/reciter_model")
content_model=joblib.load("model/quran_model")



# Function to extract audio from video
def extract_audio_from_video(video_bytes):
    # Create a temporary file to store the video
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_video_file:
        temp_video_file.write(video_bytes)
        video_path = temp_video_file.name

    # Load the video from the temporary file and extract audio
    audio_path = video_path.replace(".mp4", ".wav")
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_path)  # Save audio to a temporary file
    
    return audio_path


# Function to load or save the model locally
def load_or_save_model(model_name="small", save_path="whisper_models"):
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)
    # Define the model path
    model_path = os.path.join(save_path, f"{model_name}.pt")
    
    # Check if the model is already saved locally
    if not os.path.exists(model_path):
        print("Downloading model...")
        model = whisper.load_model(model_name)
        # Save the model's state dictionary
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at {model_path}")
    else:
        print(f"Loading model from {model_path}")
        # Load the model
        model = whisper.load_model(model_name)
        model.load_state_dict(torch.load(model_path))
    
    return model

# Transcribe function
def convert_text(audio_path, model_name="small", save_path="whisper_models"):
    model = load_or_save_model(model_name=model_name, save_path=save_path)
    result = model.transcribe(audio_path, language='ar')  # Specify Arabic language
    transcription=result['text']
    return transcription


def text_con(audio):
    model=joblib.load("whisper_models/small")
    audio=whisper.load_audio(audio)
    audio=whisper.pad_or_trim(audio)
    var=model.transcribe(audio,language='ar',fp16=False)
    text=var['text']
    return text
#prediction function
def predict_reciter(audio_path):
    y, sr = librosa.load(audio_path,sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0).reshape(1, -1)
    
    prediction = model.predict(mfcc_scaled)
    pred_reciter=np.argmax(prediction)
    return pred_reciter

# Function to handle uploaded files and process them
def handle_file_upload(file):
    # Get the video file as bytes
    video_bytes = file.getbuffer()
    audio_path = extract_audio_from_video(video_bytes)
    text=text_con(audio_path)
    # Clean up temporary files
    #os.remove(audio_path)
    return   text,audio_path


 # Juz Names Dictionary in Arabic
juz_names = { 
    1: "الم", 2: "سيقول",  3: "تلك الرسل",  4: "لن تنالوا",   5: "والمحصنات", 6: "لا يحب الله", 
    7: "وإذا سمعوا", 8: "ولو أننا", 9: "قال الملا", 10: "واعلموا",  11: "يعتذرون", 12: "ومامن دابة",       
    13: "وما أبريء", 14: "ربما", 15: "سبحان الذي", 16: "قال ألم",17: "اقترب للناس",  18: "قد أفلح",          
    19: "وقال الذين", 20: "أمن خلق", 21: "اتل ما أوحي",  22: "ومن يقنت", 23: "وماأنزلنا", 24: "فمن أظلم",         
    25: "إليه يرد",  26: "حم",  27: "قال فما خطبكم", 28: "قد أفلح من",  29: "تبارك", 30: "عم يتساءلون"       
    }

# Surah Names Dictionary in Arabic
surah_names = {
    1: "الفاتحة",2: "البقرة",3: "آل عمران",4: "النساء",5: "المائدة",6: "الأنعام",7: "الأعراف",8: "الأنفال",
    9: "التوبة",10: "يونس",11: "هود",12: "يوسف",13: "الرعد",14: "إبراهيم",15: "الحجر",16: "النحل",17: "الإسراء",
    18: "الكهف",19: "مريم",20: "طه",21: "الأنبياء",22: "الحج",23: "المؤمنون",24: "النور",25: "الفرقان",26: "الشعراء",
    27: "النمل",28: "القصص",29: "العنكبوت",30: "الروم",31: "لقمان",32: "السجدة",33: "الأحزاب",34: "سبأ",35: "فاطر",
    36: "يس",37: "الصافات",38: "ص",39: "الزمر",40: "غافر",41: "فصلت",42: "الشورى",43: "الزخرف",44: "الدخان",
    45: "الجاثية",46: "الأحقاف",47: "محمد",48: "الفتح",49: "الحجرات",50: "ق",51: "الذاريات",52: "الطور",53: "النجم",
    54: "القمر",55: "الرحمن",56: "الواقعة",57: "الحديد",58: "المجادلة",59: "الحشر",60: "الممتحنة",61: "الصف",
    62: "الجمعة",63: "المنافقون",64: "التغابن",65: "الطلاق",66: "التحريم",67: "الملك",68: "القلم",69: "الحاقة",
    70: "المعارج",71: "نوح",72: "الجن",73: "المزمل",74: "المدثر",75: "القيامة",76: "الإنسان",77: "المرسلات",78: "النبأ",
    79: "النازعات",80: "عبس",81: "التكوير",82: "الإنفطار",83: "المطففين",84: "الإنشقاق",85: "البروج",86: "الطارق",
    87: "الأعلى",88: "الغاشية",89: "الفجر",90: "البلد",91: "الشمس",92: "الليل",93: "الضحى",94: "الشرح",95: "التين",
    96: "العلق",97: "القدر",98: "البينة",99: "الزلزلة",100: "العاديات",101: "القارعة",102: "التكاثر",103: "العصر",
    104: "الهمزة",105:"الفيل",106: "قريش",107: "الماعون",108: "الكوثر",109: "الكافرون",110: "النصر",111: "المسد",
    112: "الإخلاص", 113: "الفلق", 114: "الناس"
}



def  prediction_juz_surah(text):
     # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=X.shape[1])
    # Predict using the trained model
    predictions = content_model.predict(padded_sequence)
    # Extract and decode predictions for Juz No and Surah No
    juzno_prediction = np.argmax(predictions[0], axis=1)
    surahno_prediction = np.argmax(predictions[1], axis=1)
   # Decode the predicted classes back to labels
    predicted_juzno = label_encoder_juzno.inverse_transform(juzno_prediction)
    predicted_surahno = label_encoder_surahno.inverse_transform(surahno_prediction)
    juz=predicted_juzno[0]
    surah=predicted_surahno[0]
    return juz,surah



def surah_juz(juzno,surahno):
    juz_name=juz_names.get(juzno,"unkonwn surah")
    surah_name=surah_names.get(surahno,"unkonwn surah")
    return  juz_name,surah_name




# Streamlit app layout
st.title("Quran content and Reciter Identification App")
with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["Home","About App","About Us"],
        menu_icon="cast",
        icons=["house","book","people"],
       
    )
    
if selected=='Home':
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
       
    # Check if the file is a video
        if uploaded_file.type.startswith("video"):
            st.write("Video uploaded...")
            text ,audio_path= handle_file_upload(uploaded_file)
            st.subheader("Reciter Name")
            reciter=predict_reciter(audio_path)
            if reciter==0:
                st.info("Maher Al-Quealy")
            elif reciter==1:
                st.info("Saud Al-Shuraim")
            elif reciter==2:
                st.info("Yasser Al Dossari")
        else:
            st.error("Invalid file type. Please upload a video.")
      
    
        text= "إِذَا جَاءَ نَصْرُ اللَّهِ وَالْفَتْحُ"
        juz,surah= prediction_juz_surah(text)

        juz_name,surah_name=surah_juz(juz,surah)

        juzno,juzname,=st.columns(2)
        juzno.subheader("Juz-No (رقم الجزء)")
        juzno.info(juz)
        juzname.subheader("Juz-Name (اسم الجزء)")
        juzname.info(f"{juz_name}")

        surahno,surahname=st.columns(2)
        surahno.subheader("Surah-No (رقم السورة)")
        surahno.info(surah)
        surahname.subheader("Surah-Name (اسم السورة)")
        surahname.info(f"{surah_name}")


    
elif selected=="About App":
    st.success("This app is an intelligent tool designed to analyze Quran recitation videos and extract meaningful details using deep learning models. When you upload a video, the app first extracts the audio and processes it through a trained Deep Neural Network (DNN) model to identify the reciter based on their unique voice features. Simultaneously, the app converts the audio into text and uses another trained model to analyze the text and determine details such as the Juz number, Juz name, Surah number, and Surah name. By leveraging advanced AI models for both audio and text processing, this app provides accurate and detailed information about the recitation, making it highly useful for Quranic research, learning, and study purposes.")

elif selected=="About Us":
     st.info("muhib")
     st.info("5678945678")
