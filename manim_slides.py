from manim import *
import os
from datetime import datetime

class PresentationScene(Scene):
    def __init__(self, title, content):
        super().__init__()
        self.title = title
        self.content = content

    def construct(self):
        # Title slide
        title_text = Text(self.title, font_size=72)
        title_text.to_edge(UP, buff=1)
        self.play(Write(title_text))
        self.wait(1)

        # Content slides
        for i, slide_content in enumerate(self.content):
            # Clear previous content
            self.clear()
            
            # Create slide title
            slide_title = Text(f"Slide {i+1}", font_size=48)
            slide_title.to_edge(UP)
            self.play(Write(slide_title))
            
            # Create content
            content_text = Text(slide_content, font_size=36)
            content_text.next_to(slide_title, DOWN, buff=0.5)
            self.play(Write(content_text))
            
            # Add some visual elements
            if i % 2 == 0:
                circle = Circle(radius=1)
                circle.to_edge(RIGHT)
                self.play(Create(circle))
            else:
                square = Square(side_length=2)
                square.to_edge(RIGHT)
                self.play(Create(square))
            
            self.wait(2)

def generate_presentation(title, content):
    # Create output directory if it doesn't exist
    output_dir = "generated_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/presentation_{timestamp}.mp4"
    
    # Configure Manim
    config.media_dir = output_dir
    config.output_file = filename
    
    # Create and render the scene
    scene = PresentationScene(title, content)
    scene.render()
    
    return filename

if __name__ == "__main__":
    # Example usage
    title = "My Presentation"
    content = [
        "First slide content goes here",
        "Second slide with more information",
        "Third slide with additional details",
        "Final slide with conclusions"
    ]
    
    video_path = generate_presentation(title, content)
    print(f"Presentation generated and saved to: {video_path}") 