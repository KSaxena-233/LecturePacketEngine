from manim import *
import requests
from PIL import Image
from io import BytesIO
import os

# Utility: Download and add image from URL (SVG or raster)
def image_from_url(url, width=4):
    ext = url.split(".")[-1].lower()
    filename = "temp_img." + ext
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
    try:
        if ext == "svg":
            img_mob = SVGMobject(filename).set_width(width)
        else:
            # Try to open with PIL to check validity
            from PIL import Image as PILImage
            PILImage.open(filename)
            img_mob = ImageMobject(filename).set_width(width)
    except Exception as e:
        print(f"[Image Error] Could not open {filename}: {e}")
        # Fallback: use a white square as placeholder
        img_mob = Square(side_length=width, color=BLACK, fill_opacity=0.1)
    # Optionally, clean up file after loading
    # os.remove(filename)
    return img_mob

# Base Slide class for easy extension
class SlideBase(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        self.show_slide()

    def show_slide(self):
        pass  # To be implemented by subclasses

# Slide 1: Math (Derivatives)
class MathSlide(SlideBase):
    def show_slide(self):
        title = Text("Unit 1: Derivatives", color=BLACK, weight=BOLD).scale(1.1).to_edge(UP)
        subtitle = Text("What is a derivative?", color=BLACK).scale(0.7).next_to(title, DOWN, buff=0.3)
        bullets = VGroup(
            Text("• Geometric interpretation", color=BLACK, font_size=32),
            Text("• Physical interpretation", color=BLACK, font_size=32),
            Text("• Important for all measurements", color=BLACK, font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(subtitle, DOWN, buff=0.5, aligned_edge=LEFT)
        formula = MathTex(r"\\frac{d}{dx} x^n = n x^{n-1}", color=BLACK).scale(1).next_to(bullets, DOWN, buff=0.7)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(LaggedStart(*[FadeIn(b) for b in bullets], lag_ratio=0.2))
        self.play(Write(formula))
        self.wait(1)

# Slide 2: Science (Physics Example)
class ScienceSlide(SlideBase):
    def show_slide(self):
        title = Text("Physical Interpretation of Derivatives", color=BLACK, weight=BOLD).scale(1.1).to_edge(UP)
        subtitle = Text("Velocity as a Derivative", color=BLACK).scale(0.7).next_to(title, DOWN, buff=0.3)
        bullets = VGroup(
            Text("• Derivative = rate of change", color=BLACK, font_size=32),
            Text("• Example: Dropping a pumpkin", color=BLACK, font_size=32),
            Text("• Height: y = 400 - 16t^2", color=BLACK, font_size=32)
        ).arrange(DOWN, aligned_edge=LEFT).next_to(subtitle, DOWN, buff=0.5, aligned_edge=LEFT)
        formula = MathTex(r"v = \\frac{dy}{dt} = -32t", color=BLACK).scale(1).next_to(bullets, DOWN, buff=0.7)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(LaggedStart(*[FadeIn(b) for b in bullets], lag_ratio=0.2))
        self.play(Write(formula))
        self.wait(1)

# Slide 3: Code Example
class CodeSlide(SlideBase):
    def show_slide(self):
        title = Text("Code Example: Python Derivative", color=BLACK, weight=BOLD).scale(1.1).to_edge(UP)
        subtitle = Text("Numerical Derivative in Python", color=BLACK).scale(0.7).next_to(title, DOWN, buff=0.3)
        code = '''def derivative(f, x, h=1e-5):\n    return (f(x + h) - f(x)) / h\n\nprint(derivative(lambda x: x**2, 3))  # Output: ~6'''
        code_mob = Code(code=code, tab_width=4, background="window", language="Python", font="Monospace", style="monokai", font_size=24)
        code_mob.set_color(BLACK)
        code_mob.next_to(subtitle, DOWN, buff=0.7)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(FadeIn(code_mob))
        self.wait(1)

# Slide 4: Image from the Web (simple PNG for debug)
class ImageSlide(SlideBase):
    def show_slide(self):
        title = Text("Visual: Example Image", color=BLACK, weight=BOLD).scale(1.1).to_edge(UP)
        subtitle = Text("Image from the Web (PNG)", color=BLACK).scale(0.7).next_to(title, DOWN, buff=0.3)
        # Use a simple, reliable PNG (Python logo)
        img_url = "https://www.python.org/static/community_logos/python-logo.png"
        image = image_from_url(img_url, width=5)
        image.next_to(subtitle, DOWN, buff=0.7)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.play(FadeIn(image))
        self.wait(1)

# To render all slides in sequence, use a SceneGroup or render individually
class AllSlides(Scene):
    def construct(self):
        for slide_cls in [MathSlide, ScienceSlide, CodeSlide, ImageSlide]:
            slide = slide_cls()
            slide.construct()
            self.clear()

# ---
# To add more slides: create a new class inheriting from SlideBase and add it to AllSlides
# --- 