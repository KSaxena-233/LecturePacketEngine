import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file
import os
from PIL import Image
import numpy as np
import requests
from lecture_packet_engine import LecturePacketEngine
from eeg_ml_pipeline import start_eeg_monitoring, stop_eeg_monitoring, get_current_state
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
UPLOAD_FOLDER = 'generated_videos'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
engine = LecturePacketEngine()

# Start EEG monitoring when the app starts
start_eeg_monitoring()

def create_placeholder_image(path, size=(800, 600)):
    """Create a simple placeholder image if the original is not available"""
    try:
        # Try to create a simple gradient image
        img = Image.new('RGB', size, color='white')
        # Add some text or pattern if needed
        img.save(path, 'PNG')
        return True
    except Exception as e:
        print(f"Error creating placeholder image: {e}")
        return False

# Helper to fetch an image from Unsplash (or fallback to placeholder)
def fetch_image_for_slide(topic, fallback_path):
    try:
        url = f"https://source.unsplash.com/800x600/?{topic.replace(' ', ',')}"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img_path = os.path.join(UPLOAD_FOLDER, f"{topic.replace(' ', '_')}_img.jpg")
            with open(img_path, 'wb') as f:
                f.write(response.content)
            print(f"[INFO] Downloaded image for topic: {topic}")
            return img_path
    except Exception as e:
        print(f"[WARN] Could not fetch image for {topic}: {e}")
    return fallback_path

# --- Video Generation Pipeline ---
def generate_video(prompt):
    slides = [
        {"title": prompt, "text": f"Brought to you by HEALLY", "image": None, "is_intro": True},
        {"title": "What is " + prompt + "?", "text": f"{prompt} is a fundamental concept. This slide will explain it in detail.", "image": None},
        {"title": "How does " + prompt + " work?", "text": f"A step-by-step breakdown of {prompt} with examples and diagrams.", "image": None},
        {"title": "Applications of " + prompt, "text": f"Where is {prompt} used in the real world?", "image": None},
        {"title": "Advanced Insights", "text": f"Expert-level tips, common mistakes, and best practices for {prompt}.", "image": None},
        {"title": "Summary & Next Steps", "text": f"- Mastery of {prompt} opens doors.\n- Practice, review, and explore more!\nThank you for watching!", "image": None, "is_outro": True},
    ]

    # Use a visually interesting placeholder
    placeholder_img = os.path.join(os.path.dirname(__file__), 'templates', 'heally_placeholder.png')
    try:
        Image.open(placeholder_img)
    except (FileNotFoundError, Image.UnidentifiedImageError):
        create_placeholder_image(placeholder_img)

    for slide in slides:
        # Try to fetch a relevant image for each slide
        slide['image'] = fetch_image_for_slide(slide['title'], placeholder_img)
        print(f"[DEBUG] Slide '{slide['title']}' uses image: {slide['image']}")
        time.sleep(1)  # Simulate rate limit

    video_filename = f"{prompt.replace(' ', '_')}_heally.mp4"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    render_slides_with_manim(slides, video_path)
    return video_filename

# --- Manim Rendering ---
def render_slides_with_manim(slides, video_path):
    from manim import tempconfig, Scene, Text, ImageMobject, VGroup, FadeIn, FadeOut, WHITE, BLACK, Rectangle, RIGHT, LEFT, ORIGIN
    class HeallyVideo(Scene):
        def construct(self):
            self.camera.background_color = WHITE
            for slide in slides:
                print(f"[MANIM] Rendering slide: {slide['title']}")
                txt = Text(slide['title'] + "\n\n" + slide['text'], color=BLACK, font="Arial", font_size=36, line_spacing=1.2)
                txt.wrap_width = 8
                try:
                    img = ImageMobject(slide['image'])
                    img.set_width(5)
                    img_vmobject = img.to_edge(RIGHT)
                except Exception as e:
                    print(f"[MANIM] Error loading image: {e}")
                    img_vmobject = Rectangle(width=5, height=3, fill_opacity=0.3, stroke_width=2)
                    img_vmobject.to_edge(RIGHT)
                txt.to_edge(LEFT)
                self.play(FadeIn(txt), FadeIn(img_vmobject))
                self.wait(5 if slide.get('is_intro') or slide.get('is_outro') else 12)
                self.play(FadeOut(txt), FadeOut(img_vmobject))
    with tempconfig({"output_file": video_path, "media_dir": "."}):
        HeallyVideo().render()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        video_filename = generate_video(prompt)
        return redirect(url_for('result', filename=video_filename))
    return render_template('index.html')

@app.route('/result/<filename>')
def result(filename):
    return render_template('result.html', filename=filename)

@app.route('/download/<filename>')
def download(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)

@app.route('/generate', methods=['POST'])
def generate_packet():
    try:
        data = request.get_json()
        if not data or 'topic' not in data:
            return jsonify({'error': 'Topic is required'}), 400
            
        topic = data['topic']
        logger.info(f"Generating packet for topic: {topic}")
        
        # Generate the packet
        packet_path = engine.generate_packet(topic)
        
        if not packet_path:
            return jsonify({'error': 'Failed to generate packet'}), 500
            
        return jsonify({
            'message': 'Packet generated successfully',
            'path': packet_path,
            'cognitive_state': get_current_state()
        })
        
    except Exception as e:
        logger.error(f"Error generating packet: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/cognitive_state', methods=['GET'])
def get_cognitive_state():
    """Get the current cognitive state and focus level."""
    try:
        state = get_current_state()
        
        # Calculate focus level based on attention and meditation
        readings = global_state.buffer.get_window()
        if readings:
            attention = np.mean([r['attention'] for r in readings])
            meditation = np.mean([r['meditation'] for r in readings])
            focus_level = (attention + meditation) / 2
        else:
            focus_level = 0
            
        return jsonify({
            'state': state,
            'focus_level': focus_level,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting cognitive state: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/cognitive_history', methods=['GET'])
def get_cognitive_history():
    """Get the cognitive state history."""
    try:
        readings = global_state.buffer.get_window()
        history = []
        
        for reading in readings:
            attention = reading['attention']
            meditation = reading['meditation']
            focus_level = (attention + meditation) / 2
            
            history.append({
                'timestamp': reading['timestamp'],
                'state': global_state.analyzer.analyze_state([reading]),
                'focus_level': focus_level
            })
            
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error getting cognitive history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Download a generated PDF file."""
    try:
        return send_file(
            filename,
            as_attachment=True,
            download_name=os.path.basename(filename)
        )
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.teardown_appcontext
def cleanup(exception=None):
    """Cleanup when the app context is torn down."""
    stop_eeg_monitoring()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True) 