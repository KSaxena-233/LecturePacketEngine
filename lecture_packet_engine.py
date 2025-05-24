import os
import json
import requests
from typing import Dict, List, Optional
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage
import io
import tempfile
from duckduckgo_search import ddg_images
import time
from io import BytesIO

# Constants
GEMINI_API_KEY = "AIzaSyDcfF_sos6xfCfgiIyokWGEVOqYTfsgLgk"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
OUTPUT_DIR = "output"

class LecturePacketEngine:
    def __init__(self):
        self.styles = getSampleStyleSheet()
        self._setup_styles()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
    def _setup_styles(self):
        """Setup custom styles for the PDF"""
        # MIT colors
        MIT_RED = colors.HexColor('#A31F34')  # MIT Cardinal Red
        MIT_GRAY = colors.HexColor('#8A8B8C')  # MIT Gray
        
        # Create custom styles dictionary
        custom_styles = {
            'CustomTitle': ParagraphStyle(
                name='CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1,  # Center alignment
                textColor=MIT_RED,
                fontName='Helvetica-Bold'
            ),
            'SectionHeader': ParagraphStyle(
                name='SectionHeader',
                parent=self.styles['Heading2'],
                fontSize=18,
                spaceAfter=12,
                textColor=MIT_RED,
                fontName='Helvetica-Bold'
            ),
            'ConceptTitle': ParagraphStyle(
                name='ConceptTitle',
                parent=self.styles['Heading3'],
                fontSize=16,
                spaceAfter=8,
                textColor=MIT_GRAY,
                fontName='Helvetica-Bold'
            ),
            'CustomBodyText': ParagraphStyle(
                name='CustomBodyText',
                parent=self.styles['Normal'],
                fontSize=12,
                spaceAfter=6,
                leading=14,
                fontName='Helvetica'
            ),
            'EquationBox': ParagraphStyle(
                name='EquationBox',
                parent=self.styles['Normal'],
                fontSize=12,
                backColor=colors.HexColor('#F8F9FA'),
                borderWidth=1,
                borderColor=MIT_GRAY,
                borderPadding=5,
                fontName='Helvetica'
            )
        }
        
        # Add styles to stylesheet
        for style_name, style in custom_styles.items():
            if style_name not in self.styles:
                self.styles.add(style)

    def generate_content(self, prompt: str) -> Optional[Dict]:
        """Generate educational content using Gemini API"""
        headers = {'Content-Type': 'application/json'}
        data = {
            "contents": [{"parts": [{"text": self._create_prompt(prompt)}]}]
        }
        
        try:
            print("Sending request to Gemini API...")
            response = requests.post(
                f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            response_data = response.json()
            print("\nRaw API Response:")
            print(json.dumps(response_data, indent=2))
            
            if 'candidates' in response_data and response_data['candidates']:
                content_text = response_data['candidates'][0]['content']['parts'][0]['text']
                print("\nExtracted Content Text:")
                print(content_text)
                
                # Extract JSON from the response text
                json_match = re.search(r'\{.*\}', content_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group()
                    print("\nExtracted JSON:")
                    print(json_str)
                    return json.loads(json_str)
                else:
                    print("\nNo JSON found in response text")
            else:
                print("\nNo candidates in response")
            return None
        except Exception as e:
            print(f"Error generating content: {e}")
            return None

    def _create_prompt(self, topic: str) -> str:
        """Create a detailed prompt for Gemini"""
        return f"""
        Create a comprehensive educational lecture packet about {topic}. Structure the content as follows:

        1. ABSTRACT (3-5 sentences):
        - Introduce the topic
        - Explain its significance
        - Mention key applications

        2. MAIN CONTENT (4-6 sections):
        For each section:
        - Clear concept title
        - Detailed explanation (2-3 paragraphs)
        - Mathematical examples (if applicable)
        - Real-world applications
        - Visual element suggestions

        3. EQUATIONS (if applicable):
        - List all major formulas
        - Include brief explanations

        4. PRACTICE QUESTIONS:
        - 3-5 synthesis-level questions
        - Space for answers

        Format the response as a JSON object with the following structure:
        {{
            "title": "string",
            "abstract": "string",
            "sections": [
                {{
                    "title": "string",
                    "explanation": "string",
                    "examples": ["string"],
                    "applications": ["string"],
                    "visual_elements": ["string"]
                }}
            ],
            "equations": [
                {{
                    "name": "string",
                    "formula": "string",
                    "explanation": "string"
                }}
            ],
            "practice_questions": ["string"]
        }}

        Ensure the content is:
        - Expert-level but accessible
        - Pedagogically sound
        - Visually oriented
        - Academically rigorous
        """

    def _render_latex_to_image(self, latex: str, fontsize: int = 16) -> str:
        """Render LaTeX string to an image file and return the file path."""
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = True
        fig, ax = plt.subplots(figsize=(0.01, 0.01))
        ax.axis('off')
        fig.patch.set_visible(False)
        plt.text(0.5, 0.5, f'${latex}$', fontsize=fontsize, ha='center', va='center')
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0.2, dpi=200, transparent=True)
        plt.close(fig)
        return tmpfile.name

    def _unicode_to_latex(self, s: str) -> str:
        """Convert common Unicode math symbols and superscripts/subscripts to LaTeX."""
        # Map Unicode math symbols to LaTeX
        replacements = {
            '∑': r'\sum',
            '∞': r'\infty',
            'π': r'\pi',
            '√': r'\sqrt{}',
            'Δ': r'\Delta',
            'θ': r'\theta',
            '≈': r'\approx',
            '≤': r'\leq',
            '≥': r'\geq',
            '≠': r'\neq',
            '±': r'\pm',
            '·': r'\cdot',
            '→': r'\to',
            '←': r'\leftarrow',
            '×': r'\times',
            '÷': r'\div',
            '∫': r'\int',
            'Σ': r'\Sigma',
            '∂': r'\partial',
            '∈': r'\in',
            '∉': r'\notin',
            '∩': r'\cap',
            '∪': r'\cup',
            '∅': r'\emptyset',
            'ℝ': r'\mathbb{R}',
            'ℤ': r'\mathbb{Z}',
            'ℕ': r'\mathbb{N}',
            'ℚ': r'\mathbb{Q}',
            'ℂ': r'\mathbb{C}',
            'α': r'\alpha',
            'β': r'\beta',
            'γ': r'\gamma',
            'δ': r'\delta',
            'λ': r'\lambda',
            'μ': r'\mu',
            'σ': r'\sigma',
            'τ': r'\tau',
            'φ': r'\phi',
            'ω': r'\omega',
            'Ω': r'\Omega',
            '…': r'\ldots',
            '′': r"'",
            '″': r"''",
        }
        # Superscript and subscript Unicode to LaTeX
        superscripts = {
            '⁰': '^{0}', '¹': '^{1}', '²': '^{2}', '³': '^{3}', '⁴': '^{4}', '⁵': '^{5}', '⁶': '^{6}', '⁷': '^{7}', '⁸': '^{8}', '⁹': '^{9}',
            '⁺': '^{+}', '⁻': '^{-}', '⁼': '^{=}', '⁽': '^{(}', '⁾': '^{)}', 'ⁿ': '^{n}'
        }
        subscripts = {
            '₀': '_{0}', '₁': '_{1}', '₂': '_{2}', '₃': '_{3}', '₄': '_{4}', '₅': '_{5}', '₆': '_{6}', '₇': '_{7}', '₈': '_{8}', '₉': '_{9}',
            '₊': '_{+}', '₋': '_{-}', '₌': '_{=}', '₍': '_{(}', '₎': '_{)}', 'ₙ': '_{n}'
        }
        for uni, latex in replacements.items():
            s = s.replace(uni, latex)
        for uni, latex in superscripts.items():
            s = s.replace(uni, latex)
        for uni, latex in subscripts.items():
            s = s.replace(uni, latex)
        # Add more superscripts and subscripts (including rare ones)
        more_superscripts = {
            'ᵃ': '^{a}', 'ᵇ': '^{b}', 'ᶜ': '^{c}', 'ᵈ': '^{d}', 'ᵉ': '^{e}', 'ᶠ': '^{f}', 'ᵍ': '^{g}', 'ʰ': '^{h}', 'ᶦ': '^{i}', 'ʲ': '^{j}', 'ᵏ': '^{k}', 'ˡ': '^{l}', 'ᵐ': '^{m}', 'ⁿ': '^{n}', 'ᵒ': '^{o}', 'ᵖ': '^{p}', 'ʳ': '^{r}', 'ˢ': '^{s}', 'ᵗ': '^{t}', 'ᵘ': '^{u}', 'ᵛ': '^{v}', 'ʷ': '^{w}', 'ˣ': '^{x}', 'ʸ': '^{y}', 'ᶻ': '^{z}'
        }
        more_subscripts = {
            'ₐ': '_{a}', 'ₑ': '_{e}', 'ₕ': '_{h}', 'ᵢ': '_{i}', 'ⱼ': '_{j}', 'ₖ': '_{k}', 'ₗ': '_{l}', 'ₘ': '_{m}', 'ₙ': '_{n}', 'ₒ': '_{o}', 'ₚ': '_{p}', 'ₛ': '_{s}', 'ₜ': '_{t}', 'ᵤ': '_{u}', 'ᵥ': '_{v}', 'ₓ': '_{x}'
        }
        for uni, latex in more_superscripts.items():
            s = s.replace(uni, latex)
        for uni, latex in more_subscripts.items():
            s = s.replace(uni, latex)
        # Remove any remaining non-ASCII characters (as a last resort)
        s = ''.join(c if ord(c) < 128 else '' for c in s)
        return s

    def _add_latex_paragraph(self, story, text, font_size=10, force_plaintext=True):
        """Add a paragraph of text with plain text rendering."""
        if not text:
            return

        # Always render as plain text
        story.append(Paragraph(text, self.styles['CustomBodyText']))

    def _generate_section_figure(self, section_title: str, visual_desc: str) -> str:
        """Generate a simple placeholder figure for the section and return the file path."""
        import matplotlib.pyplot as plt
        import tempfile
        fig, ax = plt.subplots(figsize=(3, 2))
        ax.text(0.5, 0.5, visual_desc, fontsize=12, ha='center', va='center', wrap=True)
        ax.set_title(section_title, fontsize=14)
        ax.axis('off')
        tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
        plt.savefig(tmpfile.name, bbox_inches='tight', pad_inches=0.2, dpi=150)
        plt.close(fig)
        return tmpfile.name

    def _create_main_content(self, content):
        """Create the main content of the lecture packet."""
        story = []
        for section in content['sections']:
            # Add section title
            story.append(Paragraph(section['title'], self.styles['SectionHeader']))
            story.append(Spacer(1, 12))

            # Add explanation
            self._add_latex_paragraph(story, section['explanation'])
            story.append(Spacer(1, 12))

            # Add examples (fix: use Paragraphs, not tables)
            if section['examples']:
                story.append(Paragraph("Examples:", self.styles['ConceptTitle']))
                for example in section['examples']:
                    story.append(Paragraph(example, self.styles['CustomBodyText']))
                    story.append(Spacer(1, 8))
                story.append(Spacer(1, 12))

            # Add applications
            if section['applications']:
                story.append(Paragraph("Applications:", self.styles['ConceptTitle']))
                for application in section['applications']:
                    self._add_latex_paragraph(story, application, force_plaintext=True)
                story.append(Spacer(1, 12))

            # Add visual elements (always generate an image)
            if section['visual_elements']:
                story.append(Paragraph("Visual Elements:", self.styles['ConceptTitle']))
                for visual in section['visual_elements']:
                    try:
                        img_path = self._smart_generate_visual(section['title'], visual)
                        from PIL import Image as PILImage
                        pil_img = PILImage.open(img_path)
                        width, height = pil_img.size
                        max_width = 400
                        if width > max_width:
                            ratio = max_width / width
                            width = max_width
                            height = int(height * ratio)
                        story.append(Image(img_path, width=width, height=height))
                    except Exception as e:
                        story.append(Paragraph(visual, self.styles['CustomBodyText']))
                    story.append(Spacer(1, 12))
        return story

    def _fetch_image_from_web(self, query):
        """Fetch an image from DuckDuckGo with enhanced error logging and retry logic."""
        max_retries = 3
        base_delay = 2  # Base delay in seconds
        max_delay = 32  # Maximum delay in seconds
        
        for attempt in range(max_retries):
            try:
                print(f"[INFO] Attempt {attempt + 1}/{max_retries} to fetch image for query: {query}")
                
                # Calculate exponential backoff delay
                delay = min(base_delay * (2 ** attempt), max_delay)
                
                # Try DuckDuckGo first
                results = ddg_images(query, max_results=1)
                if results:
                    url = results[0]['image']
                    print(f"[INFO] Found image URL: {url}")
                    
                    # Try to download the image
                    response = requests.get(url, timeout=10)
                    if response.status_code == 200:
                        # Verify it's actually an image
                        try:
                            img = PILImage.open(BytesIO(response.content))
                            img.verify()  # Verify it's a valid image
                            
                            # Save the image
                            tmpfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
                            tmpfile.write(response.content)
                            tmpfile.close()
                            print(f"[INFO] Successfully downloaded and saved image to: {tmpfile.name}")
                            return tmpfile.name
                        except Exception as e:
                            print(f"[ERROR] Invalid image data: {str(e)}")
                    else:
                        print(f"[ERROR] Failed to download image. Status code: {response.status_code}")
                else:
                    print(f"[WARNING] No images found for query: {query}")
                
                # If we get here, we need to retry
                if attempt < max_retries - 1:
                    print(f"[INFO] Waiting {delay} seconds before retry...")
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"[ERROR] Attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    print(f"[INFO] Waiting {delay} seconds before retry...")
                    time.sleep(delay)
        
        return None

    def _smart_generate_visual(self, section_title, visual_desc):
        """Generate a subject-aware visual image for any topic with improved search queries."""
        # List of search strategies to try
        search_strategies = [
            # Original description
            lambda: self._fetch_image_from_web(visual_desc),
            
            # Section title context
            lambda: self._fetch_image_from_web(f"{section_title} {visual_desc}"),
            
            # Simplified query
            lambda: self._fetch_image_from_web(visual_desc.split(':')[0] if ':' in visual_desc else visual_desc),
            
            # Keywords only
            lambda: self._fetch_image_from_web(' '.join(visual_desc.split()[:3])),
            
            # Generic fallback
            lambda: self._fetch_image_from_web(f"educational diagram {visual_desc}")
        ]
        
        # Try each strategy in sequence
        for strategy in search_strategies:
            try:
                image_path = strategy()
                if image_path:
                    return image_path
            except Exception as e:
                print(f"[WARNING] Strategy failed: {str(e)}")
                continue
        
        # If all attempts fail, log the failure
        print(f"[WARNING] Failed to generate visual for: {visual_desc} in section: {section_title}")
        return None

    def _create_equation_box(self, equations: List[Dict]) -> List:
        """Create the equations section using plain text."""
        elements = []
        elements.append(Paragraph("Key Equations", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        for eq in equations:
            elements.append(Paragraph(f"{eq['name']}:", self.styles['ConceptTitle']))
            elements.append(Paragraph(eq['formula'], self.styles['CustomBodyText']))
            elements.append(Paragraph(f"Explanation: {eq['explanation']}", self.styles['CustomBodyText']))
            elements.append(Spacer(1, 12))
        
        return elements

    def _add_page_number(self, canvas, doc):
        page_num = canvas.getPageNumber()
        text = f"Page {page_num}"
        canvas.setFont('Helvetica', 9)
        canvas.drawRightString(540, 15, text)

    def create_lecture_packet(self, prompt: str) -> str:
        content = self.generate_content(prompt)
        if not content:
            raise Exception("Failed to generate content")
        sanitized_title = re.sub(r'[^a-zA-Z0-9]', '_', prompt.lower())
        output_path = f"lecture_{sanitized_title}.pdf"
        doc = SimpleDocTemplate(
            output_path,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )
        story = []
        story.extend(self._create_cover_page(content))
        story.append(PageBreak())
        story.extend(self._create_main_content(content))
        story.append(PageBreak())
        if content.get('equations'):
            story.extend(self._create_equation_box(content['equations']))
            story.append(PageBreak())
        story.extend(self._create_practice_section(content['practice_questions']))
        doc.build(story, onLaterPages=self._add_page_number, onFirstPage=self._add_page_number)
        return output_path

    def _create_cover_page(self, content: Dict) -> List:
        """Create the cover page content"""
        elements = []
        
        # Title
        elements.append(Paragraph(content['title'], self.styles['CustomTitle']))
        elements.append(Spacer(1, 20))
        
        # Subtitle with MIT-style formatting
        elements.append(Paragraph(
            "Educational Lecture Notes",
            self.styles['SectionHeader']
        ))
        elements.append(Spacer(1, 20))
        
        # Date
        elements.append(Paragraph(
            f"Generated on: {datetime.now().strftime('%B %d, %Y')}",
            self.styles['CustomBodyText']
        ))
        elements.append(Spacer(1, 30))
        
        # Abstract
        elements.append(Paragraph("Abstract", self.styles['SectionHeader']))
        elements.append(Paragraph(content['abstract'], self.styles['CustomBodyText']))
        
        return elements

    def _create_practice_section(self, questions: List[str]) -> List:
        """Create the practice questions section"""
        elements = []
        
        elements.append(Paragraph("Practice Questions", self.styles['SectionHeader']))
        elements.append(Spacer(1, 12))
        
        for i, question in enumerate(questions, 1):
            elements.append(Paragraph(
                f"Question {i}: {question}",
                self.styles['CustomBodyText']
            ))
            elements.append(Spacer(1, 30))  # Space for answers
        
        return elements

    def _clean_unicode_to_latex(self, text):
        """Clean Unicode characters and convert them to LaTeX format."""
        # Replace Unicode characters with their LaTeX equivalents
        replacements = {
            '∑': '\\sum',
            '∞': '\\infty',
            '→': '\\to',
            '≠': '\\neq',
            '≤': '\\leq',
            '≥': '\\geq',
            '±': '\\pm',
            '×': '\\times',
            '÷': '\\div',
            '∂': '\\partial',
            'α': '\\alpha',
            'β': '\\beta',
            'γ': '\\gamma',
            'δ': '\\delta',
            'λ': '\\lambda',
            'μ': '\\mu',
            'σ': '\\sigma',
            'τ': '\\tau',
            'φ': '\\phi',
            'ω': '\\omega',
            'Ω': '\\Omega'
        }
        for unicode_char, latex_cmd in replacements.items():
            text = text.replace(unicode_char, latex_cmd)
        return text

def main():
    engine = LecturePacketEngine()
    prompt = input("Enter your educational topic: ")
    try:
        output_path = engine.create_lecture_packet(prompt)
        print(f"Lecture packet generated successfully: {output_path}")
    except Exception as e:
        print(f"Error generating lecture packet: {e}")

if __name__ == "__main__":
    main() 