from flask import Flask, render_template, request, send_from_directory
import os
import certifi
import whisper
import openai
import json
import requests

# Establecer la variable de entorno SSL_CERT_FILE
os.environ["SSL_CERT_FILE"] = certifi.where()

app = Flask(__name__, static_folder="templates/static")

# Configurar el directorio de carga
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo de Whisper
model = whisper.load_model("large")

# Configurar la API Key de OpenAI
API_KEY = "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"
API_ENDPOINT = "https://api.openai.com/v1/chat/completions"

def generate_chat_completion(messages, model="gpt-4", temperature=0.5, max_tokens=None):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }

    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if max_tokens is not None:
        data["max_tokens"] = max_tokens

    response = requests.post(API_ENDPOINT, headers=headers, data=json.dumps(data))

    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        raise Exception(f"Error {response.status_code}: {response.text}")

def improve_transcription(text):
    messages = [
        {"role": "system", "content": "Eres mi asistente para la redacción de textos."},
        {"role": "user", "content": f"Corrige la redacción de este texto. Cada vez que cambie el interlocutor, inicia un nuevo párrafo con un guión: {text}"}
    ]

    improved_text = generate_chat_completion(messages, model="gpt-4", temperature=1)
    return improved_text.strip()

def summarize_text(text):
    messages = [
        {"role": "system", "content": "Eres mi asistente para resumir textos."},
        {"role": "user", "content": f"Por favor, resume este texto: {text}"}
    ]

    summary = generate_chat_completion(messages, model="gpt-4", temperature=0.5)
    return summary.strip()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    return process_upload(False)

@app.route('/upload_literal', methods=['POST'])
def upload_literal():
    return process_upload(True)

def process_upload(literal_transcription):
    if request.method == 'POST':
        # Obtener el archivo de audio de la solicitud
        audio_file = request.files['file']
        
        # Guardar el archivo en el directorio de carga
        audio_file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        audio_file.save(audio_file_path)

        # Transcribir el archivo y guardar la transcripción en un archivo .txt
        result = model.transcribe(audio_file_path, language="Spanish")
        transcription_file = os.path.splitext(audio_file.filename)[0] + ".txt"
        transcription_path = os.path.join(app.config['UPLOAD_FOLDER'], transcription_file)

        with open(transcription_path, "w") as f:
            if literal_transcription:
                f.write(result["text"])
                output_text = result["text"]
            else:
                improved_text = improve_transcription(result["text"])
                f.write(improved_text)
                output_text = improved_text

        # Eliminar el archivo de audio original
        os.remove(audio_file_path)
    return output_text

@app.route('/summarize', methods=['POST'])
def summarize():
    if request.method == 'POST':
        transcription = request.data.decode("utf-8")
        summary = summarize_text(transcription)
        return summary

if __name__ == '__main__':
    app.run(debug=True)