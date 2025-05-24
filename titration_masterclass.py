from manim import *
import requests
import os
from PIL import Image as PILImage
from io import BytesIO

# Helper to download image if not present
IMG_DIR = "titration_images"
os.makedirs(IMG_DIR, exist_ok=True)
def get_image(url, name):
    path = os.path.join(IMG_DIR, name)
    if not os.path.exists(path):
        r = requests.get(url)
        with open(path, "wb") as f:
            f.write(r.content)
    # Validate image
    try:
        PILImage.open(path)
    except Exception as e:
        print(f"[Image Error] {name}: {e}")
        return None
    return path

# Slide data: (text, image_url, image_name)
slides = [
    ("""Titration is a quantitative analytical technique used to determine the unknown concentration of a solute in solution. By gradually adding a titrant of known concentration to a solution containing the analyte, and monitoring the reaction's progress, chemists can precisely calculate the analyte's concentration. Titrations are foundational in analytical chemistry, clinical diagnostics, environmental science, and industrial quality control.""",
     "https://upload.wikimedia.org/wikipedia/commons/7/7e/Titration_setup.jpg", "slide1.jpg"),
    ("""At the heart of titration is a well-defined chemical reaction, often a neutralization, redox, or complexation. The titrant and analyte must react in a known stoichiometric ratio. For example, in acid-base titrations:\nHCl(aq) + NaOH(aq) → NaCl(aq) + H₂O(l)\nThe endpoint is detected by a physical change, such as a color shift or a voltage jump.""",
     "https://upload.wikimedia.org/wikipedia/commons/2/2b/Acid-base_titration_curve.png", "slide2.png"),
    ("""Types of titrations:\n- Acid-Base: Involve proton transfer; use pH indicators or meters.\n- Redox: Involve electron transfer; use redox indicators or potentiometry.\n- Complexometric: Involve formation of a complex; often use EDTA.\n- Precipitation: Involve formation of a precipitate; use indicators like chromate.""",
     "https://upload.wikimedia.org/wikipedia/commons/2/2a/Titration_types.png", "slide3.png"),
    ("""Key apparatus includes a burette (for precise titrant delivery), Erlenmeyer flask (for analyte), pipette (for accurate sample transfer), and indicator or sensor. Modern titrations may use automated burettes and digital pH meters for higher precision.""",
     "https://upload.wikimedia.org/wikipedia/commons/6/6e/Titration_apparatus_annotated.jpg", "slide4.jpg"),
    ("""A titration curve plots pH versus volume of titrant added. The curve's shape reveals the acid/base strength and the equivalence point. For strong acid-strong base titrations, the pH rises sharply at the equivalence point. For weak acid-strong base, the curve is more gradual and includes a buffer region.""",
     "https://upload.wikimedia.org/wikipedia/commons/6/6b/Titration_curve.png", "slide5.png"),
    ("""Indicators are chosen based on the expected pH at the equivalence point. Phenolphthalein (colorless to pink, pH 8.2–10) is common for strong acid-strong base titrations. For weak acid-strong base, methyl orange or bromothymol blue may be used. The endpoint (indicator color change) should closely match the equivalence point.""",
     "https://upload.wikimedia.org/wikipedia/commons/2/2c/Phenolphthalein_color_change.jpg", "slide6.jpg"),
    ("""Step-by-step titration procedure:\n1. Rinse and fill the burette with titrant.\n2. Pipette a measured volume of analyte into the flask.\n3. Add a few drops of suitable indicator.\n4. Record initial burette reading.\n5. Add titrant slowly, swirling constantly.\n6. Near endpoint, add titrant dropwise.\n7. Record final burette reading at color change.\n8. Repeat for concordant results.""",
     "https://upload.wikimedia.org/wikipedia/commons/7/7e/Titration_steps.png", "slide7.png"),
    ("""The titration formula is:\nC₁V₁ = C₂V₂\nwhere C₁ and V₁ are the concentration and volume of the titrant, and C₂ and V₂ are those of the analyte. For polyprotic acids or redox titrations, include stoichiometric coefficients:\n(C₁V₁)/n₁ = (C₂V₂)/n₂\nwhere n₁ and n₂ are the number of moles of reactive species.""",
     "https://upload.wikimedia.org/wikipedia/commons/3/3b/Titration_calculation_example.png", "slide8.png"),
    ("""Sources of error and best practices:\n- Rinse all glassware with solutions to be used.\n- Avoid parallax error by reading burette at eye level.\n- Swirl flask continuously for uniform mixing.\n- Use white tile under flask for clear endpoint detection.\n- Record multiple concordant titres for accuracy.\n- Common errors: overshooting endpoint, air bubbles in burette, misreading meniscus.""",
     "https://upload.wikimedia.org/wikipedia/commons/2/2d/Burette_reading.jpg", "slide9.jpg"),
    ("""Advanced applications:\n- Pharmaceutical analysis (drug purity, content uniformity)\n- Environmental monitoring (water hardness, pollution)\n- Food industry (acidity in wine, dairy)\n- Clinical labs (blood serum analysis)\n- Automated titrators for high-throughput labs""",
     "https://upload.wikimedia.org/wikipedia/commons/8/8e/Automated_titrator.jpg", "slide10.jpg"),
    ("""Real-world example: Vitamin C in juice. To determine vitamin C content, titrate juice with iodine solution. The endpoint is detected by starch indicator turning blue-black. Calculations reveal mg of vitamin C per 100 mL.""",
     "https://upload.wikimedia.org/wikipedia/commons/3/3e/Vitamin_C_titration.jpg", "slide11.jpg"),
    ("""Mastering titration requires understanding chemical theory, precise technique, and careful calculation. It is a gateway to advanced analytical chemistry and essential for scientific careers.""",
     "https://upload.wikimedia.org/wikipedia/commons/7/7e/Titration_setup.jpg", "slide12.jpg"),
]

class TitrationMasterclass(Scene):
    def construct(self):
        self.camera.background_color = WHITE
        for idx, (text, url, name) in enumerate(slides):
            img_path = get_image(url, name)
            # Text block
            txt = Text(text, color=BLACK, font="Arial", font_size=32, line_spacing=1.2, t2c={"\n": BLACK})
            txt.stretch_to_fit_width(6)
            # Image block
            if img_path:
                img = ImageMobject(img_path).set_width(5)
            else:
                img = Square(side_length=5, color=BLACK, fill_opacity=0.1)
            # Layout: text left, image right
            group = VGroup(txt, img).arrange(RIGHT, buff=1.0).move_to(ORIGIN)
            self.play(FadeIn(group))
            self.wait(15)
            self.play(FadeOut(group)) 