import os
import uuid
import time
from flask import Flask, request, jsonify, render_template, send_from_directory
from core.processing import process_vintage_film

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB max for RAW files
app.config['MAX_AGE_MINUTES'] = 15 # Images older than 15 min will be deleted space

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

def cleanup_old_files():
    """Löscht alle zuvor hochgeladenen und verarbeiteten Dateien."""
    try:
        for folder in [app.config['UPLOAD_FOLDER'], app.config['PROCESSED_FOLDER']]:
            if not os.path.exists(folder):
                continue
            for filename in os.listdir(folder):
                filepath = os.path.join(folder, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
                    print(f"[Cleanup] Gelöscht: {filepath}")
    except Exception as e:
        print(f"[Cleanup Error] {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # 1. Alte Dateien aufräumen, bevor das neue Bild verarbeitet wird
    cleanup_old_files()

    if 'image' not in request.files:
        return jsonify({'error': 'Kein Bild hochgeladen.'}), 400
        
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Keine Datei ausgewählt.'}), 400
        
    if file:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ['.jpg', '.jpeg', '.png', '.webp', '.dng', '.cr2', '.nef', '.arw']:
             return jsonify({'error': 'Ungültiges Dateiformat. Bitte JPG/PNG/RAW hochladen.'}), 400
             
        filename = str(uuid.uuid4()) + file_ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        # Always save processed as .jpg since browsers can't display RAW files natively
        processed_filename = "processed_" + str(uuid.uuid4()) + ".jpg"
        processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
        
        file.save(filepath)
        
        try:
            # Gewählten Filter aus dem Formular auslesen (Standard: filter1)
            filter_type = request.form.get('filter', 'filter1')
            # Bildverarbeitungspipeline aufrufen (immer 100% für live client-side blending)
            process_vintage_film(filepath, processed_filepath,
                                 filter_type=filter_type, intensity=1.0)
            
            return jsonify({
                'success': True,
                'original_url': f'/uploads/{filename}',
                'processed_url': f'/processed/{processed_filename}'
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/processed/<filename>')
def serve_processed(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    print("Starte Vintage Camera App auf http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=False)
