import requests
import json
from typing import Dict, List, Optional

GEMINI_API_KEY = "AIzaSyDcfF_sos6xfCfgiIyokWGEVOqYTfsgLgk"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def generate_content(prompt: str) -> Optional[Dict]:
    """
    Generate content using Gemini API
    """
    headers = {
        'Content-Type': 'application/json'
    }
    
    data = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }
    
    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error generating content: {e}")
        return None

def generate_presentation_content(topic: str) -> Dict:
    """
    Generate structured presentation content including:
    - Title
    - Key points
    - Visual elements suggestions
    - Summary
    """
    prompt = f"""
    Create a structured educational presentation about {topic}. Include:
    1. A compelling title
    2. 3-5 key points with explanations
    3. Suggestions for relevant graphs, diagrams, or visual elements
    4. A concise summary
    
    Format the response as a JSON object with the following structure:
    {{
        "title": "string",
        "key_points": ["string"],
        "visual_elements": ["string"],
        "summary": "string"
    }}
    """
    
    response = generate_content(prompt)
    if response and 'candidates' in response:
        try:
            # Extract the text from the response and parse it as JSON
            content = response['candidates'][0]['content']['parts'][0]['text']
            return json.loads(content)
        except Exception as e:
            print(f"Error parsing response: {e}")
            return None
    return None 