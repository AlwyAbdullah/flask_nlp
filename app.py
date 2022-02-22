from flask import Flask, render_template, request, redirect
import speech_recognition as sr
from werkzeug.utils import secure_filename
import librosa
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        os.makedirs(os.path.join(app.instance_path, 'uploads'), exist_ok=True)
        file.save(os.path.join(app.instance_path, 'uploads', secure_filename(file.filename)))
        if file.filename == "":
            return redirect(request.url)

        a = librosa.get_duration(filename='instance/uploads/'+file.filename)
        print(int(a/180))
        print(int(a))
        b = int(a/180) + 1

        list_of_num = [0] * b
        print(list_of_num)

        text = [''] * (len(list_of_num))
        print(text)

        c = 0 # For Indexing
        d = 0 # For Editing Array based on duration

        while(c < b):
            if(list_of_num[c] < int(a)):
                list_of_num[c] = d
                c+=1
                d = d + 180

        if file:
            r = sr.Recognizer()

            for idx, val in enumerate(list_of_num):  
                with sr.AudioFile('instance/uploads/'+file.filename) as source:
                    if(idx == len(list_of_num) - 1):
                        c = int(a) - list_of_num[idx]
                        c = c - 10
                        urutan = 'akhir'
                    else:
                        c = 180
                        urutan = int((val/60)+3)
                    try:
                        audio = r.record(source, offset=val, duration=c)
                        text[idx] = r.recognize_google(audio, language="id-ID")
                        print('index: ', idx, 'Menit ke: ', int(val/60), '-' , urutan, ' : ', text[idx])
                    except LookupError:
                            print("Sorry your voice is ugly")
        for idx, val in enumerate(text):
            text[idx] += '.'

        transcript = ' '.join(text)

    return render_template('text_summarization.html', transcript=transcript)


if __name__ == "__main__":
    app.run(debug=True, threaded=True)