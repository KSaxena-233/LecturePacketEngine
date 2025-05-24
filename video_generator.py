from manim import *
import os
from datetime import datetime

class PromptBasedScene(Scene):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt

    def construct(self):
        # Create title
        title = Text(self.prompt, font_size=48)
        title.to_edge(UP)
        self.play(Write(title))

        # Create content based on prompt
        if "mechanical" in self.prompt.lower() or "engineering" in self.prompt.lower():
            # Create a simple mechanical diagram
            circle = Circle(radius=2)
            square = Square(side_length=2)
            self.play(Create(circle))
            self.play(Transform(circle, square))
            
            # Add some engineering elements
            arrow = Arrow(LEFT * 3, RIGHT * 3)
            self.play(Create(arrow))
            
            # Add some text
            text = Text("Mechanical Systems", font_size=36)
            text.next_to(arrow, DOWN)
            self.play(Write(text))
            
            # Hold the scene
            self.wait(2)

def generate_video(prompt):
    # Create output directory if it doesn't exist
    output_dir = "generated_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/video_{timestamp}.mp4"
    
    # Configure Manim
    config.media_dir = output_dir
    config.output_file = filename
    
    # Create and render the scene
    scene = PromptBasedScene(prompt)
    scene.render()
    
    return filename

if __name__ == "__main__":
    # Example usage
    prompt = "Mechanical Engineering"
    video_path = generate_video(prompt)
    print(f"Video generated and saved to: {video_path}") 