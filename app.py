from datetime import datetime
from flask import Flask, render_template, request, redirect, make_response
from matplotlib.pyplot import text
import speech_recognition as sr
from werkzeug.utils import secure_filename
import os
import re
import numpy as np
import pandas as pd
from pprint import pprint
from fpdf import FPDF
from rouge import Rouge
import locale


#gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.summarization import summarize
from gensim.models import CoherenceModel

import nltk

nltk.download('stopwords')
# nltk.download('punkt')

app = Flask(__name__, static_url_path='/static')

@app.route("/", methods=["GET", "POST"])
def index():
    transcript = ""
    listTopic = ""
    score= ""
    coherenceScore = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        radio = request.form["radio"]
        
        directory = "audio"
        parent_dir = os.path.join(app.root_path+'/static', directory)
        os.makedirs(parent_dir, exist_ok=True)
        if file.filename == "":
            return redirect(request.url)
        else:
            file.save(os.path.join(app.root_path+'/static', directory, secure_filename(file.filename)))

        import wave
        import contextlib
        with contextlib.closing(wave.open(app.root_path+'/static/audio/'+secure_filename(file.filename),'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            a = int(frames / rate)
            print(a)

        # a = librosa.get_duration(filename='instance/uploads/'+secure_filename(file.filename))
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
                with sr.AudioFile(app.root_path+'/static/audio/'+secure_filename(file.filename)) as source:
                    if(idx == len(list_of_num) - 1):
                        c = int(a) - list_of_num[idx]
                        c = c - 10
                        urutan = 'akhir'
                    else:
                        c = 180
                        urutan = int((val/60)+3)
                    audio = r.record(source, offset=val, duration=c)
                    try:
                        text[idx] = r.recognize_google(audio, language="id-ID")
                        print('index: ', idx, 'Menit ke: ', int(val/60), '-' , urutan, ' : ', text[idx])
                    except LookupError:
                            print("Sorry... there is error in the process")
        for idx, val in enumerate(text):
            text[idx] += '.'

        joinedText = ' '.join(text)
        print(joinedText)

        if radio == "Text Summarization":
            summarizedText = summarize(joinedText, ratio=0.2)
            stringSummarize = ''.join(map(str, summarizedText))
            stringSummarize = stringSummarize.replace('.', '.\n')
            transcript = stringSummarize
            print('')
            print(transcript)

            rouge = Rouge()
            score = rouge.get_scores(summarizedText, joinedText, avg=True)
            score = str(score)
            score = score.replace('[', '').replace(']', '').replace("'", "").replace('{', '').replace('}', '').replace('r:', 'recall:').replace('p', 'precision').replace('f', 'Score')
            print(score)
        else:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory #Library for removing stopword
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

            from nltk.stem import PorterStemmer


            factory = StopWordRemoverFactory()
            stopword = factory.create_stop_word_remover()

            # create stemmer
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            porter = PorterStemmer()

            # NLTK Stop words
            from nltk.corpus import stopwords
            stop_words = stopwords.words('indonesian')
            stop_words.extend(['ya', 'gua', 'an', 'ancang', 'anggar', 'anti', 'arti', 'aston', 'arah', 'atus', 'bales', 'banget', 'bas', 'basic', 'bayang', 'bayi', 'be', 'belanja','bibi','biding', 'bill', 'bos', 'bunda',
                            'cek', 'isi', 'iya', 'jagain', 'jaga', 'jual', 'juang', 'jurus', 'jam', 'kastem', 'kali', 'hendra', 'henti', 'kain' 'febby', 'flanel', 'ganti', 'guling', 'habis', 'hadis', 'halo', 'han',
                            'hantam', 'harap', 'harga' 'cakap', 'coc', 'cocok', 'diarsi', 'dika', 'eko', 'elemen', 'enak', 'assalamualaikum', 'warahmatullahi', 'abang', 'waalaikumsalam',	'wakil', 'usul', 'utama',
                            'utara',	'warahmatullah',	'wawancara',	'whatsapp',	'wi',	'word	','zaman', 'acara', 'adab', 'adik', 'yuk', 'agam', 'wajib', 'mbak', 'pak', 'nggih', 'wabarakatuh', 'selamat', 'siang', 'pagi',
                            'malam', 'pijit','pikir','perangkat','perhati','perilaku', 'permata', 'persen', 'peter', 'pokok', 'gitu', 'nggak', 'sih', 'bener', 'ajak', 'anak', 'ajar', 'nama', 'kayak', 'pakai', 'masuk', 'mata',
                            'orang', 'nya', 'aja', 'kasih', 'kait', 'tulis', 'web', 'gin', 'temen', 'bu', 'sesuai', 'kenal', 'beda', 'bikin', 'salah', 'laku', 'terima', 'kalah', 'kaya', 'kedip', 'kelulus', 'kemarin', 'kembang', 'kerjain',
                            'kuning', 'kuningan', 'lari', 'latar', 'latih', 'lempar', 'lunak', 'album', 'maju', 'makan', 'makasih', 'maksud', 'mas', 'mateng', 'mentingin', 'menteri',
                            'merah', 'mi', 'mobil', 'monggo', 'mpasi', 'mulut', 'muncul', 'naksir', 'ngerjain', 'ngerti', 'ngomong', 'ngomongnya', 'nih', 'nomor', 'nota', 'nyetting', 'objek', 'nyata',
                            'of', 'oke', 'on', 'operasi', 'operasional', 'oriented', 'pemograman', 'pendi', 'penuh', 'peran', 'pilih', 'pimpin', 'pisah', 'pln', 'poin', 'pusing', 'pribadi',
                            'presales', 'pritil', 'pusing', 'putus', 'raba', 'ranaya', 'ranah', 'ranting', 'read', 'real', 'rendah', 'rendi','ritme', 'rp', 'rpl', 'rumah', 'ruang', 'rumpun', 'rumput',
                            'salam', 'sanding', 'saran', 'sebentar', 'selebriti', 'semprot', 'sesi', 'sit', 'siti', 'situ', 'sopir', 'sorry', 'standart', 'sulam', 'sulis', 'susu', 'susun', 'tablu',
                            'takut', 'tambal', 'tanda', 'tangan', 'tantang', 'tarik', 'tato', 'tau', 'tdd', 'tembok', 'temu', 'tenda', 'tik', 'terap', 'titik', 'tok', 'tolong', 'tuh', 'tuju', 'tunda',
                            'udah', 'up', 'upda', 'urus' 'usernya', 'analis', 'analisa', 'analisis', 'bagus', 'bahas', 'bappenas', 'barangkali', 'batas', 'bekal', 'benah', 'benang', 'biar', 'brainly', 'cari', 'contoh',
                            'cuman', 'batang', 'beliau', 'bentuk', 'db', 'dbms', 'diassign', 'ah', 'ahli', 'ajarin', 'alya', 'analitik', 'analyst', 'and', 'andi', 'sila', 'aren', 'bk', 'bpk', 'buah', 'ci',
                            'codingannya', 'crv', 'dateng', 'dede', 'detil', 'dijadiin', 'dinas', 'disimpen', 'dorong', 'dot', 'edi', 'entar', 'etl', 'febby', 'fi', 'gang', 'hadits', 'hit', 'hilang', 'if', 'ikan', 'ikhtiar',
                            'in', 'intan', 'ips', 'istri', 'ji', 'kadang', 'kaget', 'kain', 'kena', 'ketemu', 'ketes', 'lho', 'letak', 'life', 'lukman', 'lupa', 'mc', 'millions', 'miss', 'nambah', 'nembak', 'ngajar', 'nikah', 'nokia',
                            'obeng', 'nyanyi', 'paku', 'palu', 'papi', 'pms', 'pentest', 'primbon', 'pulsa', 'rezeki', 'riak', 'rian', 'roti', 'sali', 'sambut', 'sebenernya', 'segitu', 'sederhana', 'sekian', 'sisa', 'slimenya', 'socket', 'soft',
                            'sydney', 'stringstream', 'tablo', 'tampil', 'tanah', 'teteh', 'tv', 'urus', 'lu', 'papa'])


            # Define functions for stopwords, bigrams, trigrams and lemmatization
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

            # Stemming indonesia dan inggris
            eng_stem_text1_topic = porter.stem(joinedText)
            text1_topic_stem = stemmer.stem(eng_stem_text1_topic)
            text1_topic_stopwords_sastrawi = stopword.remove(text1_topic_stem)
            print(len(text1_topic_stem))
            print(len(text1_topic_stopwords_sastrawi))

            # Change to list
            list_word_topic = [(text1_topic_stopwords_sastrawi.strip())]
            print(list_word_topic)

            # Remove Stopwords
            data_words_nostops_text1_topic = remove_stopwords(list_word_topic)
            print(data_words_nostops_text1_topic)

            # Build the bigram and trigram models (Bigram adalah 2 kata sedangkan Trigram adalah 3 kata)
            bigram = gensim.models.Phrases(data_words_nostops_text1_topic, min_count=5, threshold=100) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words_nostops_text1_topic], threshold=100)

            # Faster way to get a sentence clubbed as a trigram/bigram
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            # See trigram example
            # print(trigram_mod[bigram_mod[data_words[0]]])
            print(trigram_mod[bigram_mod[data_words_nostops_text1_topic[0]]])

            # Create Dictionary
            id2word = corpora.Dictionary(data_words_nostops_text1_topic)

            # Create Corpus
            texts = data_words_nostops_text1_topic

            # Term Document Frequency
            corpus = [id2word.doc2bow(text1) for text1 in texts]

            # View
            print(corpus[:1])

            [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=3500,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=2000,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

            # Print the Keyword in the topics
            # pprint(lda_model.print_topics(1))
            # doc_lda = lda_model[corpus]

            # topic = lda_model.print_topics(1)
            # stringTopic = ' '.join(map(str, topic))
            # stringTopic = ''.join(i for i in stringTopic if not i.isdigit())
            # stringTopic = stringTopic.replace("'", '').replace(",", '').replace("*", '').replace(".", '').replace('"', '').replace("(", '').replace(")", '').replace("+", ",")
            # print(stringTopic)

            pprint(lda_model.print_topics(10))
            doc_lda = lda_model[corpus]

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_nostops_text1_topic, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            coherenceScore = coherence_lda

            listTopic = lda_model.print_topics(10)
            changeOrder = [9,8,7,6,5,4,3,2,1,0]
            listTopic = [listTopic[i] for i in changeOrder]

            for n in range(10):
                print('Topic ', n+1, ' :', listTopic[n])
                listTopic[n] =listTopic[n]

            print(listTopic)
            stringTopic = ' '.join(map(str, listTopic))
            stringTopic = ''.join(i for i in stringTopic if not i.isdigit())
            stringTopic = stringTopic.replace("'", '').replace(",", '').replace("*", '').replace(".", '').replace('"', '').replace("(", '').replace(")", '\n').replace(" ", "").replace("+", ', ')
            print(stringTopic)
            arrayTopic = stringTopic.split("\n")

            listTopic = arrayTopic

        path = os.path.join(app.root_path+'/static/audio/'+secure_filename(file.filename))
        # path = os.path.join(app.instance_path, 'uploads/'+secure_filename(file.filename))
        os.remove(path)
    return render_template('text_summarization.html', transcript=transcript, listTopic=listTopic, score=score, coherenceScore=coherenceScore)


@app.route('/test')
def test():
    return render_template('summarizePDF.html')

@app.route('/liveMic', methods=["GET", "POST"])
def liveMic():
    transcript = ''
    listTopic=''
    coherenceScore=''
    score=''
    if request.method == "POST":
        print("FORM DATA RECEIVED")
        textarea = request.form["formtextarea"]
        radio = request.form["radio"]
        if radio == "Text Summarization":
            summarized_text = summarize(textarea, ratio=0.4)
            print(summarized_text)
            transcript = summarized_text

            rouge = Rouge()
            score = rouge.get_scores(summarized_text, textarea, avg=True)
            score = str(score)
            score = score.replace('[', '').replace(']', '').replace("'", "").replace('{', '').replace('}', '').replace('r:', 'recall:').replace('p', 'precision').replace('f', 'Score')
            print(score)
        else:
            from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory #Library for removing stopword
            from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

            from nltk.stem import PorterStemmer


            factory = StopWordRemoverFactory()
            stopword = factory.create_stop_word_remover()

            # create stemmer
            factory = StemmerFactory()
            stemmer = factory.create_stemmer()
            porter = PorterStemmer()

            # NLTK Stop words
            from nltk.corpus import stopwords
            stop_words = stopwords.words('indonesian')
            stop_words.extend(['ya', 'gua', 'an', 'ancang', 'anggar', 'anti', 'arti', 'aston', 'arah', 'atus', 'bales', 'banget', 'bas', 'basic', 'bayang', 'bayi', 'be', 'belanja','bibi','biding', 'bill', 'bos', 'bunda',
                            'cek', 'isi', 'iya', 'jagain', 'jaga', 'jual', 'juang', 'jurus', 'jam', 'kastem', 'kali', 'hendra', 'henti', 'kain' 'febby', 'flanel', 'ganti', 'guling', 'habis', 'hadis', 'halo', 'han',
                            'hantam', 'harap', 'harga' 'cakap', 'coc', 'cocok', 'diarsi', 'dika', 'eko', 'elemen', 'enak', 'assalamualaikum', 'warahmatullahi', 'abang', 'waalaikumsalam',	'wakil', 'usul', 'utama',
                            'utara',	'warahmatullah',	'wawancara',	'whatsapp',	'wi',	'word	','zaman', 'acara', 'adab', 'adik', 'yuk', 'agam', 'wajib', 'mbak', 'pak', 'nggih', 'wabarakatuh', 'selamat', 'siang', 'pagi',
                            'malam', 'pijit','pikir','perangkat','perhati','perilaku', 'permata', 'persen', 'peter', 'pokok', 'gitu', 'nggak', 'sih', 'bener', 'ajak', 'anak', 'ajar', 'nama', 'kayak', 'pakai', 'masuk', 'mata',
                            'orang', 'nya', 'aja', 'kasih', 'kait', 'tulis', 'web', 'gin', 'temen', 'bu', 'sesuai', 'kenal', 'beda', 'bikin', 'salah', 'laku', 'terima', 'kalah', 'kaya', 'kedip', 'kelulus', 'kemarin', 'kembang', 'kerjain',
                            'kuning', 'kuningan', 'lari', 'latar', 'latih', 'lempar', 'lunak', 'album', 'maju', 'makan', 'makasih', 'maksud', 'mas', 'mateng', 'mentingin', 'menteri',
                            'merah', 'mi', 'mobil', 'monggo', 'mpasi', 'mulut', 'muncul', 'naksir', 'ngerjain', 'ngerti', 'ngomong', 'ngomongnya', 'nih', 'nomor', 'nota', 'nyetting', 'objek', 'nyata',
                            'of', 'oke', 'on', 'operasi', 'operasional', 'oriented', 'pemograman', 'pendi', 'penuh', 'peran', 'pilih', 'pimpin', 'pisah', 'pln', 'poin', 'pusing', 'pribadi',
                            'presales', 'pritil', 'pusing', 'putus', 'raba', 'ranaya', 'ranah', 'ranting', 'read', 'real', 'rendah', 'rendi','ritme', 'rp', 'rpl', 'rumah', 'ruang', 'rumpun', 'rumput',
                            'salam', 'sanding', 'saran', 'sebentar', 'selebriti', 'semprot', 'sesi', 'sit', 'siti', 'situ', 'sopir', 'sorry', 'standart', 'sulam', 'sulis', 'susu', 'susun', 'tablu',
                            'takut', 'tambal', 'tanda', 'tangan', 'tantang', 'tarik', 'tato', 'tau', 'tdd', 'tembok', 'temu', 'tenda', 'tik', 'terap', 'titik', 'tok', 'tolong', 'tuh', 'tuju', 'tunda',
                            'udah', 'up', 'upda', 'urus' 'usernya', 'analis', 'analisa', 'analisis', 'bagus', 'bahas', 'bappenas', 'barangkali', 'batas', 'bekal', 'benah', 'benang', 'biar', 'brainly', 'cari', 'contoh',
                            'cuman', 'batang', 'beliau', 'bentuk', 'db', 'dbms', 'diassign', 'ah', 'ahli', 'ajarin', 'alya', 'analitik', 'analyst', 'and', 'andi', 'sila', 'aren', 'bk', 'bpk', 'buah', 'ci',
                            'codingannya', 'crv', 'dateng', 'dede', 'detil', 'dijadiin', 'dinas', 'disimpen', 'dorong', 'dot', 'edi', 'entar', 'etl', 'febby', 'fi', 'gang', 'hadits', 'hit', 'hilang', 'if', 'ikan', 'ikhtiar',
                            'in', 'intan', 'ips', 'istri', 'ji', 'kadang', 'kaget', 'kain', 'kena', 'ketemu', 'ketes', 'lho', 'letak', 'life', 'lukman', 'lupa', 'mc', 'millions', 'miss', 'nambah', 'nembak', 'ngajar', 'nikah', 'nokia',
                            'obeng', 'nyanyi', 'paku', 'palu', 'papi', 'pms', 'pentest', 'primbon', 'pulsa', 'rezeki', 'riak', 'rian', 'roti', 'sali', 'sambut', 'sebenernya', 'segitu', 'sederhana', 'sekian', 'sisa', 'slimenya', 'socket', 'soft',
                            'sydney', 'stringstream', 'tablo', 'tampil', 'tanah', 'teteh', 'tv', 'urus'])


            # Define functions for stopwords, bigrams, trigrams and lemmatization
            def remove_stopwords(texts):
                return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

            def make_bigrams(texts):
                return [bigram_mod[doc] for doc in texts]

            def make_trigrams(texts):
                return [trigram_mod[bigram_mod[doc]] for doc in texts]

            # Stemming indonesia dan inggris
            eng_stem_text1_topic = porter.stem(textarea)
            text1_topic_stem = stemmer.stem(eng_stem_text1_topic)
            text1_topic_stopwords_sastrawi = stopword.remove(text1_topic_stem)
            print(len(text1_topic_stem))
            print(len(text1_topic_stopwords_sastrawi))

            # Change to list
            list_word_topic = [(text1_topic_stopwords_sastrawi.strip())]
            print(list_word_topic)

            # Remove Stopwords
            data_words_nostops_text1_topic = remove_stopwords(list_word_topic)
            print(data_words_nostops_text1_topic)

            # Build the bigram and trigram models (Bigram adalah 2 kata sedangkan Trigram adalah 3 kata)
            bigram = gensim.models.Phrases(data_words_nostops_text1_topic, min_count=5, threshold=100) # higher threshold fewer phrases.
            trigram = gensim.models.Phrases(bigram[data_words_nostops_text1_topic], threshold=100)

            # Faster way to get a sentence clubbed as a trigram/bigram
            bigram_mod = gensim.models.phrases.Phraser(bigram)
            trigram_mod = gensim.models.phrases.Phraser(trigram)

            # See trigram example
            # print(trigram_mod[bigram_mod[data_words[0]]])
            print(trigram_mod[bigram_mod[data_words_nostops_text1_topic[0]]])

            # Create Dictionary
            id2word = corpora.Dictionary(data_words_nostops_text1_topic)

            # Create Corpus
            texts = data_words_nostops_text1_topic

            # Term Document Frequency
            corpus = [id2word.doc2bow(text1) for text1 in texts]

            # View
            print(corpus[:1])

            [[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]]

            # Build LDA model
            lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                    id2word=id2word,
                                                    num_topics=100,
                                                    random_state=100,
                                                    update_every=1,
                                                    chunksize=2000,
                                                    passes=10,
                                                    alpha='auto',
                                                    per_word_topics=True)

            # Print the Keyword in the topics
            # pprint(lda_model.print_topics(1))
            # doc_lda = lda_model[corpus]

            # topic = lda_model.print_topics(1)
            # stringTopic = ' '.join(map(str, topic))
            # stringTopic = ''.join(i for i in stringTopic if not i.isdigit())
            # stringTopic = stringTopic.replace("'", '').replace(",", '').replace("*", '').replace(".", '').replace('"', '').replace("(", '').replace(")", '').replace("+", ",")
            # print(stringTopic)

            pprint(lda_model.print_topics(5))
            doc_lda = lda_model[corpus]

            # Compute Coherence Score
            coherence_model_lda = CoherenceModel(model=lda_model, texts=data_words_nostops_text1_topic, dictionary=id2word, coherence='c_v')
            coherence_lda = coherence_model_lda.get_coherence()
            coherenceScore = coherence_lda

            listTopic = lda_model.print_topics(5)
            changeOrder = [4,3,2,1,0]
            listTopic = [listTopic[i] for i in changeOrder]

            for n in range(5):
                print('Topic ', n+1, ' :', listTopic[n])
                listTopic[n] =listTopic[n]

            print(listTopic)
            stringTopic = ' '.join(map(str, listTopic))
            stringTopic = ''.join(i for i in stringTopic if not i.isdigit())
            stringTopic = stringTopic.replace("'", '').replace(",", '').replace("*", '').replace(".", '').replace('"', '').replace("(", '').replace(")", '\n').replace(" ", "").replace("+", ', ')
            print(stringTopic)
            arrayTopic = stringTopic.split("\n")

            listTopic = arrayTopic

    return render_template('liveMicTranscription.html', transcript=transcript, listTopic=listTopic, coherenceScore=coherenceScore, score=score)

@app.route('/downloadPDF', methods=["POST", "GET"])
def downloadPDF():
    date_time_obj = ""
    if request.method == "POST":
        print("FORM DATA RECEIVED")

        participantNumber = request.form["member"]
        name = request.form.getlist("memberName")
        print(name)
        print(participantNumber)
        date = request.form["fordate"]
        # select = request.form["formselect"]
        textarea = request.form["formtextarea"]
        tempat = request.form["tempat"]

        date_time_obj = datetime.strptime(date, "%Y-%m-%d")
        date_time_obj = date_time_obj.strftime("%d %B %Y")

        pdf = FPDF()
        pdf.add_page()
        # pdf.set_right_margin(20)

        for x in range(int(participantNumber)):
            print(name[x])

        # pdf.set_margins(10,0,50)

        pdf.set_font("Times", size=12)

        pdf.image("static/image/HeaderPDF.png",w=190, h=50)

        pdf.cell(195, 10, txt = tempat + ", " +  date_time_obj + "     ", ln = 10, align = 'R')

        pdf.cell(200, 10, txt = "Participant:", ln = 10, align = 'L')

        for x in range(int(participantNumber)):
            if x == 0:
                pdf.set_font("Times", "B", size=12)
            else:
                pdf.set_font("Times", size=12)
            pdf.cell(200, 10, txt = name[x], ln = 10, align = 'L')

        # pdf.cell(200, 10, txt = name[0], ln = 10, align = 'L')

        pdf.set_font("Times", 'B', size=20)
        pdf.multi_cell(200, 10, txt = "Kesimpulan", align = 'C')

        pdf.set_font("Times", size=12)
        pdf.multi_cell(190, 10, txt = textarea, align = 'J')

        pdf.cell(195, 10, txt = tempat + ", " +  date_time_obj + "     ", ln = 10, align = 'R')
        pdf.cell(180, 10, txt = "Ketua Rapat", ln=10, align='R')
        pdf.ln()
        pdf.ln()
        pdf.ln()

        pdf.cell(183, 10, txt = name[0], ln=40, align='R')

        response = make_response(pdf.output(dest='S').encode('latin-1'))
        response.headers.set('Content-Disposition', 'attachment', filename="Notulensi" + '.pdf')
        response.headers.set('Content-Type', 'application/pdf')
    return response

if __name__ == "__main__":
    app.run(debug=True, threaded=True)