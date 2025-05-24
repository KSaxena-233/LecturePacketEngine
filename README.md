# LecturePacketEngine

A powerful Python-based educational content generation system that creates professional lecture packets with AI-generated content, equations, and visuals.

## Features

- **AI-Powered Content Generation**
  - Uses Google's Gemini API for intelligent content creation
  - Structured educational format with sections, examples, and applications
  - Automatic practice question generation

- **Smart Image Generation**
  - Multiple search strategies for finding relevant images
  - Exponential backoff retry mechanism (2s, 4s, 8s, up to 32s)
  - PIL-based image validation
  - Automatic image resizing and optimization

- **Professional PDF Generation**
  - MIT-style professional formatting
  - Custom fonts and colors
  - LaTeX equation rendering
  - Page numbering and section organization
  - Cover page generation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/LecturePacketEngine.git
cd LecturePacketEngine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
   - Get an API key from Google AI Studio
   - Add it to your environment variables or update the key in the code

## Usage

Basic usage:
```python
from lecture_packet_engine import LecturePacketEngine

# Initialize the engine
engine = LecturePacketEngine()

# Generate a lecture packet
prompt = "Your educational topic here"
output_path = engine.create_lecture_packet(prompt)
print(f"Lecture packet generated: {output_path}")
```

## Project Structure

```
LecturePacketEngine/
├── lecture_packet_engine.py    # Main engine class
├── requirements.txt            # Python dependencies
├── examples/                   # Example usage scripts
├── output/                     # Generated PDFs
└── tests/                      # Test files
```

## Dependencies

- Python 3.9+
- ReportLab
- PIL (Pillow)
- requests
- duckduckgo_search
- matplotlib
- numpy

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Google Gemini API for content generation
- ReportLab for PDF generation
- DuckDuckGo for image search 