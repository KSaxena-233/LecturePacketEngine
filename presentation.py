from manim import *
import requests
import json
from pathlib import Path
from pdf_generator import create_notes_pdf
from flask import Flask, send_file, render_template_string
import os

class EducationalSlide(Scene):
    def construct(self):
        # Title slide
        title = Text("Educational Presentation", font_size=72)
        subtitle = Text("Created with Manim", font_size=36)
        subtitle.next_to(title, DOWN, buff=0.5)
        
        self.play(Write(title))
        self.play(FadeIn(subtitle))
        self.wait(2)
        self.play(FadeOut(title), FadeOut(subtitle))
        
        # Content slide example
        content_title = Text("Key Concepts", font_size=48)
        bullet_points = VGroup(
            Text("• First important point", font_size=36),
            Text("• Second important point", font_size=36),
            Text("• Third important point", font_size=36)
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.3)
        bullet_points.next_to(content_title, DOWN, buff=0.5)
        
        self.play(Write(content_title))
        for point in bullet_points:
            self.play(Write(point))
        self.wait(2)
        
        # Graph example
        axes = Axes(
            x_range=[-3, 3, 1],
            y_range=[-2, 2, 1],
            axis_config={"color": BLUE},
        )
        graph = axes.plot(lambda x: np.sin(x), color=YELLOW)
        
        self.play(FadeOut(content_title), FadeOut(bullet_points))
        self.play(Create(axes), Create(graph))
        self.wait(2)

def generate_presentation(prompt):
    # This function will be expanded to use Gemini API for content generation
    pass

if __name__ == "__main__":
    # Example usage
    config.media_width = "75%"
    config.media_dir = "media"
    config.output_file = "presentation.mp4"
    
    # Create the presentation
    scene = EducationalSlide()
    scene.render()
    
    # Generate PDF notes
    content = {
        "title": "Educational Presentation",
        "key_points": ["First important point", "Second important point", "Third important point"],
        "visual_elements": ["Graph of sin(x)"],
        "summary": "A brief summary of the presentation."
    }
    create_notes_pdf(content, "notes.pdf")
    
    # Start a simple Flask web server to serve the video and PDF
    app = Flask(__name__)
    
    @app.route('/')
    def index():
        return render_template_string('''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Educational Presentation</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    video { width: 100%; max-width: 800px; }
                    .download-btn { display: inline-block; margin: 10px 0; padding: 10px 20px; background-color: #4CAF50; color: white; text-decoration: none; border-radius: 5px; }
                </style>
            </head>
            <body>
                <h1>Educational Presentation</h1>
                <video controls>
                    <source src="/video" type="video/mp4">
                    Your browser does not support the video tag.
                </video>
                <br>
                <a href="/download/video" class="download-btn">Download Video</a>
                <a href="/download/pdf" class="download-btn">Download PDF Notes</a>
            </body>
            </html>
        ''')
    
    @app.route('/video')
    def video():
        return send_file('media/videos/1080p60/presentation.mp4', mimetype='video/mp4')
    
    @app.route('/download/<file_type>')
    def download(file_type):
        if file_type == 'video':
            return send_file('media/videos/1080p60/presentation.mp4', as_attachment=True, download_name='presentation.mp4')
        elif file_type == 'pdf':
            return send_file('notes.pdf', as_attachment=True, download_name='notes.pdf')
        return "File not found", 404
    
    app.run(debug=True, port=5002) 