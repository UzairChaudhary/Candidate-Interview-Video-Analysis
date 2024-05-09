import os
import numpy as np
import subprocess
from predictemt import pred, removeout, vidframe, ssimscore1
from flask import Flask, request, render_template, jsonify
import speech_recognition as sr
from werkzeug.utils import secure_filename
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt
import io
import base64
import urllib
import tkinter as tk
from moviepy.editor import VideoFileClip
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from flask_cors import CORS

import spacy, fitz
from ResumeAnalysis.utility.downloadFile import download_file
from ResumeAnalysis.utility.extractTextFromPdf import extractTextFromPdf
from ResumeAnalysis.calculateSimilarity import calculate_similarity

import whisper

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')   #load face detection cascade file

app = Flask(__name__)
CORS(app)
#CORS(app, origins="http://localhost:3001")
app.secret_key = 'some secret key'



def vedio_to_text(path):
    # Extracting file name and extension
    file_name, extension = os.path.splitext(path)
    
    # Constructing output file path
    audio_output_path = f"{file_name}.wav"
    
    # Constructing the ffmpeg command
    command2mp3 = f"ffmpeg -i {path} {audio_output_path}"
    
    # Execute the command
    os.system(command2mp3)
    
    model = whisper.load_model("medium")
    result = model.transcribe(audio_output_path,fp16=False)
    print(result["text"])
    try:
        #text=r.recognize_google(audio)
        # Save the text to a .txt file in the root directory
        with open("uploads/speech_text.txt", "w") as file:
            file.write(result["text"])
        
        # Printing speech
        print('Speech Detected: ')
        
    except:
        print('Could not hear anything!')
        with open("uploads/speech_text.txt", "w") as file:
            file.write("")
    
        

# def video_to_text(video_path):
#     #----------------------------------Speech Detection Part-----------------------------------#
#     # Fix video metadata
#     #fixed_video_path = fix_video_metadata(video_path)
#     video = VideoFileClip(video_path)
#     audio_path = video_path[:-4] + ".wav"
#     audio = video.audio
#     audio.write_audiofile(audio_path)
    
#     r = sr.Recognizer()

#     with sr.AudioFile(audio_path) as source:
#         audio = r.listen(source)
#         try:
#             # Google speech recognition (You can select from other options)
#             text = r.recognize_google(audio)
#             # Save the text to a .txt file in the root directory
#             with open("uploads/speech_text.txt", "w") as file:
#                 file.write(text)
            
#             # Printing speech
#             print('Speech Detected:')
#             print(text)
        
#         except:
#             print('Could not hear anything!')

#-------------------------------------------------------------------------------------------#
overall_scores = {'neg': 0, 'neu': 0, 'pos': 0, 'compound': 0}
def text_sentiment_analysis():
    # Download VADER lexicon
    #nltk.download('vader_lexicon')
    #nltk.download('punkt')
    # Initialize the VADER sentiment analyzer
    sid = SentimentIntensityAnalyzer()

    # Sample paragraph for sentiment analysis
    #paragraph = "I love spending time with my friends. The weather was perfect for a picnic in the park. However, I had a terrible experience at the restaurant we went to afterward. The food was cold and the service was extremely slow. Despite that, we managed to have a good time. Overall, it was a mixed day."
    # Read the text input from a .txt file
    file_path = "uploads/speech_text.txt"  # Change this to the path of your .txt file
    with open(file_path, "r") as file:
        paragraph = file.read()
    # Tokenize the paragraph into sentences
    sentences = nltk.sent_tokenize(paragraph)
    
    if not sentences:
        print("No sentences found in the input text.")
        return "Neutral"

    # Perform sentiment analysis on each sentence
    
    for sentence in sentences:
        scores = sid.polarity_scores(sentence)
        
        overall_scores['neg'] += scores['neg']
        overall_scores['neu'] += scores['neu']
        overall_scores['pos'] += scores['pos']
        overall_scores['compound'] += scores['compound']
        
        print(sentence)
        scores = sid.polarity_scores(sentence)
        print("Sentiment Score:", scores)
        if scores['compound'] > 0 and scores['pos']>scores['neu']:
            print("Overall Sentiment: Positive")
        elif scores['compound'] < 0:
            print("Overall Sentiment: Negative")
        else:
            print("Overall Sentiment: Neutral")
        print()
    
    # Calculate the average sentiment scores
    num_sentences = len(sentences)
    if num_sentences != 0:
        for key in overall_scores:
            overall_scores[key] /= num_sentences

        # Print the combined sentiment analysis of the whole paragraph
        print("Combining the Sentiment Analysis result...")
        print("Negative Sentiment:", overall_scores['neg'])
        print("Neutral Sentiment:", overall_scores['neu'])
        print("Positive Sentiment:", overall_scores['pos'])
        print("Compound Sentiment Score:", overall_scores['compound'])

        # Determine overall sentiment
        if overall_scores['compound'] > 0  and overall_scores['pos' ]> overall_scores['neu']:
            print("Overall Sentiment: Positive")
            return "Positive"
        elif overall_scores['compound'] < 0:
            print("Overall Sentiment: Negative")
            return "Negative"
        else:
            print("Overall Sentiment: Neutral")
            return "Neutral"
    else:
        print("Cannot calculate average sentiment scores due to zero sentences.")
        return "Neutral"


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        print("Request: ")
        print(request)
        if 'file' in request.files:

            f = request.files['file']  #getting uploaded video 
            basepath = os.path.dirname(__file__)
            file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
            f.save(file_path)  #saving uploaded video

            result, face = vidframe(file_path) #running vidframe with the uploaded video
            vedio_to_text(file_path)
            overall_sentiment=text_sentiment_analysis()
            #os.remove(file_path)  #removing the video as we dont need it anymore
            #os.remove("Audio.wav")
            #os.remove("uploads/speech_text.txt")
        else:
            result, face = vidframe(0)
        try:
            smileindex=result.count('Happy')/len(result)  #smileIndex
            smileindex=round(smileindex,2)

        except:
            smileindex=0

        ssimscore=[ssimscore1(i,j) for i, j in zip(face[: -1],face[1 :])]  # calculating similarityscore for images
        if np.mean(ssimscore)<0.6:
        	posture="Not Good"
        else:
        	posture="Good"
        # fig = plt.figure()     #matplotlib plot
        # ax = fig.add_axes([0,0,1,1])
        # ax.axis('equal')
        #emotion = ['angry','disgust','fear', 'happy', 'sad']
        #emotion = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        #counts = [result.count('angry'),result.count('disgust'),result.count('fear'),result.count('happy'),result.count('sad')]
        emotion_counts = [
            result.count('Angry'),
            result.count('Disgust'),
            result.count('Fear'),
            result.count('Happy'),
            result.count('Sad'),
            result.count('Surprise'),
            result.count('Neutral')
        ]

        # Calculate the total counts
        total_emotions = sum(emotion_counts)

        # Calculate the percentage of each emotion
        counts = [count / total_emotions * 100 for count in emotion_counts]

        #ax.pie(counts, labels=emotion, autopct='%1.2f%%')  # adding pie chart
        #img = io.BytesIO()
        #plt.savefig(img, format='png')  # saving pie chart
        #img.seek(0)
        #plot_data = urllib.parse.quote(base64.b64encode(img.read()).decode())  # pie chart object that can be returned to the html

        # Calculate the scores for nervousness and confidence
        nervousness_score = (counts[2] + counts[0] + counts[1]) / total_emotions  # Fear + Angry + Disgust
        confidence_score = (counts[3] + counts[6]) / total_emotions  # Happy + Neutral

        # Determine the emotional state based on scores
        nervousness_state = "High" if nervousness_score > 0.5 else "Low"
        confidence_state = "High" if confidence_score > 0.5 else "Low"
        
        # Define weights for each parameter
        
        weight_smile_index = 0.2
        weight_nervousness = 0.4
        weight_confidence = 0.4
        # Calculate overall score
        overall_score = (
            
            weight_smile_index * smileindex +
            weight_nervousness * nervousness_score +
            weight_confidence * confidence_score
        )
        # Round the overall score to 2 decimal places
        print("overall Score:", overall_score)
        if (overall_score<10) :
            overall_score *= 100
        
        
        overall_score = round(overall_score,2)
        
        
        return jsonify({
            'Posture': posture,
            'Sentiment': overall_sentiment,
            'PositiveSentiment':overall_scores['pos'],
            'NegativeSentiment':overall_scores['neg'],
            'NeutralSentiment':overall_scores['neu'],
            'SmileIndex': smileindex,
            'Angry': counts[0],
            'Disgust': counts[1],
            'Fear': counts[2],
            'Happy': counts[3],
            'LackOfEnthusiasm': counts[4],#sad
            'Surprise': counts[5],
            'Neutral': counts[6],
            'NervousnessScore': nervousness_score,
            'ConfidenceScore': confidence_score,
            'NervousnessState': nervousness_state,
            'ConfidenceState': confidence_state,
            'OverallScore':overall_score,
            
            
        }), 200

    return None

download_folder = 'ResumeAnalysis/Downloads'  # Update this to the desired folder path

# API endpoint for calculating similarity
@app.route('/api/calculate_similarity', methods=['POST'])
def api_calculate_similarity():
    try:
        data = request.json
        print(data)
        # Extract resume and job description URLs from the request data
        resume_url = data.get('resumeUrl')
        job_description_url = data.get('jobDescriptionUrl')
        # Download Resume File from URL
        resume_file = download_file(resume_url, download_folder)
        
        with fitz.open(resume_file) as doc:
            print("Resume taken as input")
            # Load the spacy model
            print("Loading Resume Parser model...")
            resume_nlp = spacy.load('ResumeAnalysis/assets/ResumeModel/output/model-best')
            print("Resume Parser Model Loaded")
            resume_dic = extractTextFromPdf(doc, resume_nlp)
            print("Resume Parser Model work done")
            # Resume Dictionary
            print(resume_dic)

        # Download job description file from URL
        job_description_file = download_file(job_description_url, download_folder)
        with fitz.open(job_description_file) as jd_doc:
            # Spacy model for Job Description
            print("Loading Jd Parser model...")
            jd_nlp = spacy.load('ResumeAnalysis/assets/JdModel/output/model-best')
            print("Jd Parser model loaded")
            jd_dic = extractTextFromPdf(jd_doc, jd_nlp)
            print("JD Parser Model work done")
            # Job Description Dictionary
            print(jd_dic)
        
        similarity_percentage = calculate_similarity(resume_dic, jd_dic)

          
        
        return jsonify({'similarity_percentage': similarity_percentage})
        

    except Exception as e:
        print(e)
        return jsonify({'error': 'Internal Server Error'}), 500
    


if __name__ == '__main__':
    app.run(debug=True)
    app.secret_key = 'some secret key'
