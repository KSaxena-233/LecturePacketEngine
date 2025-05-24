from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import os

def create_notes_pdf(content: dict, output_path: str = "notes.pdf"):
    """
    Create a PDF with presentation notes and figures
    """
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    
    # Build the PDF content
    story = []
    
    # Add title
    story.append(Paragraph(content['title'], title_style))
    story.append(Spacer(1, 12))
    
    # Add key points
    for point in content['key_points']:
        story.append(Paragraph(f"• {point}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    story.append(Spacer(1, 20))
    
    # Add visual elements section
    story.append(Paragraph("Visual Elements", styles['Heading2']))
    story.append(Spacer(1, 12))
    for element in content['visual_elements']:
        story.append(Paragraph(f"• {element}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    story.append(Spacer(1, 20))
    
    # Add summary
    story.append(Paragraph("Summary", styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(content['summary'], styles['Normal']))
    
    # Build the PDF
    doc.build(story) 