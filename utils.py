"""
Utility functions for the Travel Plan Agent.
This module contains helper functions used throughout the application.
"""

import os
import logging
import datetime
from pathlib import Path
import re
import uuid
from typing import List, Dict, Any, Union

# Import sys for module checks
import sys

# Try to import markdown for HTML generation
try:
    import markdown
    MARKDOWN_AVAILABLE = True
except ImportError:
    MARKDOWN_AVAILABLE = False
    print("Markdown module not available. Basic HTML formatting will be used instead.")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(paths: List[Path]) -> None:
    """
    Create directories if they don't exist.
    
    Args:
        paths: List of Path objects to create
    """
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Ensured directory exists: {path}")

def extract_travel_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract travel-related entities from text using regex patterns.
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Dictionary of entity types and their values
    """
    entities = {
        "locations": [],
        "accommodations": [],
        "activities": [],
        "transport": []
    }
    
    # Simple regex patterns for demonstration
    location_patterns = [
        r'\b(?:city|town|village|country|island|destination|region)\s+of\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
        r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b(?=.*(?:visit|explore|travel|trip|tour))'
    ]
    
    accommodation_patterns = [
        r'\b(?:hotel|resort|hostel|apartment|airbnb|lodging|motel|inn)\s+([A-Za-z0-9\s]+)(?=[\.,])'
    ]
    
    activity_patterns = [
        r'(?:visit|explore|see|tour|experience)\s+the\s+([A-Za-z0-9\s]+)(?=[\.,])',
        r'(?:hiking|swimming|sightseeing|shopping|dining)\s+(?:in|at)\s+([A-Za-z0-9\s]+)(?=[\.,])'
    ]
    
    transport_patterns = [
        r'(?:train|bus|taxi|ferry|flight|rental car|bike|walk)\s+(?:to|from|between)\s+([A-Za-z0-9\s]+)(?=[\.,])',
        r'(?:travel|go|journey)\s+by\s+(train|bus|taxi|ferry|flight|car|bike|foot)'
    ]
    
    # Extract entities using patterns
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        entities["locations"].extend(matches)
    
    for pattern in accommodation_patterns:
        matches = re.findall(pattern, text)
        entities["accommodations"].extend(matches)
    
    for pattern in activity_patterns:
        matches = re.findall(pattern, text)
        entities["activities"].extend(matches)
    
    for pattern in transport_patterns:
        matches = re.findall(pattern, text)
        entities["transport"].extend(matches)
    
    # Remove duplicates and clean up entities
    for category in entities:
        entities[category] = list(set([item.strip() for item in entities[category]]))
    
    return entities

def format_plan_as_markdown(plan: Dict[str, Any]) -> str:
    """
    Format a travel plan dictionary as a markdown string.
    
    Args:
        plan: Travel plan dictionary
        
    Returns:
        Formatted markdown string
    """
    md = f"# {plan.get('title', 'Travel Plan')}\n\n"
    
    if 'summary' in plan:
        md += f"## Overview\n{plan['summary']}\n\n"
    
    if 'destinations' in plan:
        md += "## Destinations\n"
        for dest in plan['destinations']:
            md += f"- {dest}\n"
        md += "\n"
    
    if 'duration' in plan:
        md += f"## Duration\n{plan['duration']}\n\n"
    
    if 'itinerary' in plan:
        md += "## Itinerary\n"
        for day, activities in plan['itinerary'].items():
            md += f"### {day}\n"
            for activity in activities:
                md += f"- {activity}\n"
            md += "\n"
    
    if 'accommodations' in plan:
        md += "## Accommodations\n"
        for acc in plan['accommodations']:
            md += f"- **{acc['name']}**: {acc['description']}\n"
        md += "\n"
    
    if 'transportation' in plan:
        md += "## Transportation\n"
        for transport in plan['transportation']:
            md += f"- {transport}\n"
        md += "\n"
    
    if 'budget' in plan:
        md += "## Estimated Budget\n"
        for category, amount in plan['budget'].items():
            md += f"- **{category}**: {amount}\n"
        md += "\n"
    
    if 'tips' in plan:
        md += "## Travel Tips\n"
        for tip in plan['tips']:
            md += f"- {tip}\n"
        md += "\n"
    
    return md

def save_travel_plan(plan_text: str, destination_name: str = None) -> Dict[str, str]:
    """
    Save a travel plan as Markdown, TXT and HTML files.
    
    Args:
        plan_text: The travel plan text
        destination_name: Name of the main destination (for filename)
        
    Returns:
        Dictionary with paths to the saved files
    """
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate a filename based on destination and date
    if not destination_name:
        destination_match = re.search(r"# ([^#\n]+)", plan_text)
        if destination_match:
            destination_name = destination_match.group(1).strip()
        else:
            destination_name = "Travel Plan"
    
    # Clean filename
    clean_name = re.sub(r'[^\w\s-]', '', destination_name).strip().replace(' ', '_')
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    base_filename = f"{clean_name}_{date_str}"
    
    # Save as Markdown
    md_path = output_dir / f"{base_filename}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(plan_text)
    logger.info(f"Markdown saved to {md_path}")
    
    # Save as plain text
    txt_path = output_dir / f"{base_filename}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(plan_text)
    logger.info(f"Plain text saved to {txt_path}")
    
    # Save as HTML
    try:
        # Create a simple HTML version
        if MARKDOWN_AVAILABLE:
            # Use markdown if available
            html_content = markdown.markdown(plan_text)
        else:
            # Simple fallback if markdown isn't available
            html_content = plan_text.replace('\n', '<br>').replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>')
            for heading in re.findall(r'<h[123]>(.*)', html_content):
                html_content = html_content.replace(f'<h1>{heading}', f'<h1>{heading}</h1>')
                html_content = html_content.replace(f'<h2>{heading}', f'<h2>{heading}</h2>')
                html_content = html_content.replace(f'<h3>{heading}', f'<h3>{heading}</h3>')
        
        html_styled = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>{destination_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 2cm; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
                h3 {{ color: #2980b9; }}
                li {{ margin: 5px 0; }}
            </style>
        </head>
        <body>
            <h1>{destination_name}</h1>
            <p>Generated on {datetime.datetime.now().strftime("%d %B %Y")}</p>
            {html_content}
            <p style="text-align: center; margin-top: 30px; font-size: 0.8em; color: #7f8c8d;">
                Created with Travel Plan Agent
            </p>
        </body>
        </html>
        """
        
        html_path = output_dir / f"{base_filename}.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_styled)
        logger.info(f"HTML saved to {html_path}")
    except Exception as e:
        logger.error(f"HTML generation failed: {str(e)}")
        html_path = None
    
    # Return paths
    result = {
        "markdown": str(md_path),
        "txt": str(txt_path),
        "html": str(html_path) if 'html_path' in locals() and html_path and html_path.exists() else None
    }
    
    logger.info(f"Travel plan saved as: {result}")
    return result

def estimate_reading_time(text: str) -> int:
    """
    Estimate reading time in minutes for a text.
    
    Args:
        text: Text to estimate reading time for
        
    Returns:
        Estimated reading time in minutes
    """
    # Average reading speed: 200-250 words per minute
    words = len(text.split())
    minutes = round(words / 225)
    return max(1, minutes)  # Minimum 1 minute