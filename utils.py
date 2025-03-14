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

def save_travel_plan(plan_text: str, metadata_text: str = "", destination_name: str = None) -> Dict[str, str]:
    """
    Save a travel plan as Markdown and TXT files with optimized filename.
    
    Args:
        plan_text: The travel plan text (email content)
        metadata_text: Technical metadata to include in a collapsible section
        destination_name: Name of the main destination (for filename)
        
    Returns:
        Dictionary with paths to the saved files
    """
    # Create output directory if it doesn't exist
    output_dir = Path("./output")
    output_dir.mkdir(exist_ok=True)
    
    # Generate a comprehensive metadata-rich filename
    filename_info = generate_optimized_filename(plan_text, destination_name)
    
    # Format the metadata as a collapsible section if provided
    if metadata_text:
        collapsible_metadata = f"""

<details>
<summary>TECHNISCHE DETAILS</summary>

{metadata_text}

</details>
"""
        # Full markdown document with collapsible section
        full_markdown = f"{plan_text}\n\n{collapsible_metadata}"
        
        # Plain text version (without HTML tags)
        full_plaintext = f"{plan_text}\n\n## TECHNISCHE DETAILS\n\n{metadata_text}"
    else:
        # If no metadata, just use the plan text directly
        full_markdown = plan_text
        full_plaintext = plan_text
    
    # Save as Markdown
    md_path = output_dir / f"{filename_info['filename']}.md"
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(full_markdown)
    logger.info(f"Markdown saved to {md_path}")
    
    # Save as plain text
    txt_path = output_dir / f"{filename_info['filename']}.txt"
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(full_plaintext)
    logger.info(f"Plain text saved to {txt_path}")
    
    # Return paths with additional metadata
    result = {
        "markdown": str(md_path),
        "txt": str(txt_path),
        "title": filename_info['title'],
        "destinations": filename_info['destinations'],
        "duration": filename_info['duration'],
        "themes": filename_info['themes'],
        "filename": filename_info['filename'],
        "created": datetime.datetime.now().strftime("%Y-%m-%d")
    }
    
    # Log a nice summary
    logger.info(f"Travel plan saved in markdown and text formats")
    logger.info(f"Plan title: {result['title']}")
    
    if result['destinations']:
        logger.info(f"Destinations: {', '.join(result['destinations'])}")
    
    if result['duration']:
        logger.info(f"Duration: {result['duration']}")
    
    if result['themes']:
        logger.info(f"Themes: {', '.join(result['themes'])}")
    
    return result

def generate_optimized_filename(plan_text: str, destination_name: str = None) -> Dict[str, str]:
    """
    Generate an optimized, descriptive filename based on plan content.
    
    Args:
        plan_text: The travel plan text
        destination_name: Optional destination name override
        
    Returns:
        Dictionary with filename information
    """
    result = {
        "filename": "Travel_Plan",
        "title": "Travel Plan", 
        "destinations": [],
        "duration": "",
        "themes": []
    }
    
    # Extract plan title - first look for H1 heading
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
        result["title"] = raw_title
    elif destination_name:
        result["title"] = f"{destination_name} Travel Plan"
    
    # Extract destinations - look for countries, cities, and regions
    # Comprehensive list of locations to check for
    location_list = [
        # Countries
        "Vietnam", "Thailand", "Japan", "China", "Indonesia", "Malaysia", 
        "Singapore", "Laos", "Cambodia", "Germany", "Austria", "Switzerland",
        "France", "Italy", "Spain", "United States", "Canada", "Australia",
        # Cities and regions
        "Hanoi", "Bangkok", "Tokyo", "Ho Chi Minh", "Hoi An", "Da Nang", 
        "Halong Bay", "Berlin", "Munich", "Stuttgart", "Paris", "Rome",
        "Barcelona", "London", "New York", "Los Angeles", "Sydney",
        "Leinfelden", "Echterdingen", "Leinfelden-Echterdingen"
    ]
    
    # Find all destinations mentioned in the plan
    destinations = []
    for location in location_list:
        if location in plan_text:
            destinations.append(location)
    
    if destinations:
        result["destinations"] = destinations
    
    # Extract trip duration with various patterns
    duration_patterns = [
        r'(\d+)[\s-]*(day|days|Day|Days)',
        r'(\d+)[\s-]*(night|nights|Night|Nights)',
        r'(\d+)[\s-]*(week|weeks|Week|Weeks)'
    ]
    
    for pattern in duration_patterns:
        duration_match = re.search(pattern, plan_text)
        if duration_match:
            duration_value = duration_match.group(1)
            duration_unit = duration_match.group(2).lower()
            
            # Normalize to 'Day' format
            if 'night' in duration_unit:
                result["duration"] = f"{duration_value}-Day"
            elif 'week' in duration_unit:
                # Convert weeks to days
                days = int(duration_value) * 7
                result["duration"] = f"{days}-Day"
            else:
                result["duration"] = f"{duration_value}-Day"
            break
    
    # Extract trip themes with comprehensive list
    theme_keywords = {
        "Relaxation": ["relaxation", "relax", "peaceful", "retreat", "tranquil", "serene", "calm"],
        "Adventure": ["adventure", "exciting", "thrill", "adrenaline", "expedition", "trek", "hiking"],
        "Beach": ["beach", "coastal", "ocean", "sea", "shore", "sand", "surf"],
        "Cultural": ["culture", "cultural", "history", "historical", "heritage", "tradition", "museum"],
        "Food": ["food", "culinary", "gastronomy", "cuisine", "dining", "tasting", "restaurant"],
        "Family": ["family", "kid-friendly", "children", "family-friendly", "parents", "kid", "kids"],
        "Party": ["party", "nightlife", "club", "bar", "dancing", "nightclub", "festival"],
        "Romantic": ["romantic", "romance", "couple", "honeymoon", "anniversary", "love", "intimate"],
        "Luxury": ["luxury", "luxurious", "high-end", "exclusive", "premium", "first-class", "5-star"],
        "Budget": ["budget", "affordable", "cheap", "economical", "backpacker", "inexpensive", "budget-friendly"]
    }
    
    found_themes = []
    for theme, keywords in theme_keywords.items():
        if any(keyword in plan_text.lower() for keyword in keywords):
            found_themes.append(theme)
    
    if found_themes:
        result["themes"] = found_themes
    
    # Current date in YYYYMMDD format
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    
    # Now build the optimized filename with the new format
    # "[Orte] - [Gesamt-Tage]T Reise - [Hauptzweck] - [Datum].md"
    
    # Part 1: Destinations
    destination_part = ""
    if destinations:
        if len(destinations) <= 3:
            destination_part = "-".join(destinations)
        else:
            # If more than 3 destinations, use first two + a count
            destination_part = f"{destinations[0]}-{destinations[1]}+{len(destinations)-2}"
    else:
        destination_part = "Unspecified"
    
    # Part 2: Total days
    duration_part = ""
    if result["duration"]:
        # Extract just the number from the duration (e.g., "10-Day" -> "10T")
        days_match = re.search(r'(\d+)', result["duration"])
        if days_match:
            duration_part = f"{days_match.group(1)}T Reise"
        else:
            duration_part = "Reise"
    else:
        duration_part = "Reise"
    
    # Part 3: Main purpose
    purpose_part = ""
    if found_themes and len(found_themes) > 0:
        # Use the first two themes as the main purpose
        if len(found_themes) == 1:
            purpose_part = found_themes[0]
        else:
            purpose_part = f"{found_themes[0]}-{found_themes[1]}"
    else:
        purpose_part = "Finalized_Itinerary"
    
    # Build the filename with the new format - directly using underscores
    filename = f"{destination_part}_{duration_part}_{purpose_part}_{date_str}"
    
    # Final safety checks for filename
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\s+', '_', filename)  # Replace any remaining spaces with underscores
    filename = re.sub(r'_+', '_', filename)  # Remove multiple underscores
    
    # Keep filename at a reasonable length
    if len(filename) > 180:
        filename = filename[:160] + f"_{date_str}"
    
    result["filename"] = filename
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