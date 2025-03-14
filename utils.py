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

# Import required modules
import sys

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
    Save a travel plan as Markdown and TXT files.
    
    Args:
        plan_text: The travel plan text
        destination_name: Name of the main destination (for filename)
        
    Returns:
        Dictionary with paths to the saved files
    """
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Extract plan title for the filename
    # First try to find the main title (typically at the beginning)
    title_match = re.search(r"^#\s+(.+?)$", plan_text, re.MULTILINE)
    
    # If no match at the beginning, try looking for any h1 heading
    if not title_match:
        title_match = re.search(r"#\s+([^#\n]+)", plan_text)
    
    # If still no match, look for a title-like pattern
    if not title_match:
        title_match = re.search(r"(?i)(?:title|plan|itinerary|vacation|trip):\s*([^\n]+)", plan_text)
    
    if title_match:
        raw_title = title_match.group(1).strip()
        # Remove any metadata markers from the title
        raw_title = re.sub(r'PLAN_METADATA__ANALYSIS', '', raw_title)
    else:
        # Fall back to destination name or a default
        if destination_name:
            raw_title = destination_name
        else:
            # Last resort - check for mention of specific countries
            for country in ["Vietnam", "Thailand", "Japan", "China", "Bali", "Indonesia"]:
                if country in plan_text:
                    raw_title = f"{country} Travel Plan"
                    break
            else:
                raw_title = "Travel Plan"
    
    # Look for potential itinerary type info in the title or text
    days_prefix = ""
    if "days" in plan_text.lower() or "day" in plan_text.lower():
        # Try to extract number of days from text
        days_match = re.search(r'(\d+)[\s-]*day', plan_text.lower())
        if days_match:
            days = days_match.group(1)
            days_prefix = f"{days} Days - "
    
    # Extract country/region information if possible
    countries = []
    for country in ["Vietnam", "Thailand", "Japan", "China", "Bali", "Indonesia", 
                   "Malaysia", "Singapore", "Laos", "Cambodia", "Asia", "Europe"]:
        if country.lower() in plan_text.lower() or country.lower() in raw_title.lower():
            countries.append(country)
    
    # Create a descriptive plan title
    plan_title = raw_title
    if not any(country in plan_title for country in countries) and countries:
        # Add country to title if not already there
        countries_text = " & ".join(countries[:2])  # Limit to 2 countries
        plan_title = f"{countries_text}: {plan_title}"
    
    # Clean and format the title
    plan_title = plan_title.replace("PLAN_METADATA__ANALYSIS", "").strip()
    plan_title = re.sub(r'\s+', ' ', plan_title)  # Remove extra spaces
    
    # Use full title for display, but create a clean version for filename
    display_title = plan_title
    
    # Create a filename-friendly version (keeping more of the original title)
    # Still sanitize but less aggressively
    clean_title = re.sub(r'[<>:"/\\|?*]', '', plan_title)  # Remove illegal filename chars
    clean_title = re.sub(r'\s+', ' ', clean_title).strip()  # Normalize whitespace
    
    # Add date - use a more readable date format
    date_str = datetime.datetime.now().strftime("%Y-%m-%d")
    
    # Construct an improved base filename that's more readable but still safe
    # Format: "Title - YYYY-MM-DD"
    base_filename = f"{clean_title} - {date_str}"
    
    # If the filename is still too long for some systems, cap it at 180 chars
    if len(base_filename) > 180:
        # Keep the start and end portions for context
        base_filename = base_filename[:150] + "..." + base_filename[-20:]
    
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
    
    # Return paths with additional metadata
    result = {
        "markdown": str(md_path),
        "txt": str(txt_path),
        "title": display_title,  # Use the clean, full title for display
        "days": days_match.group(1) if 'days_match' in locals() and days_match else None,
        "countries": countries,
        "filename": base_filename,
        "created": date_str
    }
    
    # Log a nice summary
    logger.info(f"Travel plan saved as: {result['markdown']} and {result['txt']}")
    logger.info(f"Plan title: {plan_title}")
    if countries:
        logger.info(f"Countries: {', '.join(countries)}")
    if days_prefix:
        logger.info(f"Duration: {days_prefix}")
    
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