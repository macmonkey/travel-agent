"""
Document processor module for the Travel Plan Agent.
This module contains functionality for processing PDF and DOCX documents.
"""

import os
from pathlib import Path
from typing import List, Dict, Any

import PyPDF2
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentProcessor:
    """Class for processing travel documents and extracting relevant information."""
    
    def __init__(self, config):
        """
        Initialize the document processor.
        
        Args:
            config: Application configuration object
        """
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            length_function=len
        )
    
    def process_directory(self, directory_path: Path) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of processed document chunks with metadata
        """
        all_documents = []
        
        if not directory_path.exists():
            print(f"Directory {directory_path} does not exist.")
            return all_documents
        
        # Count total files for progress reporting
        pdf_files = list(directory_path.glob("**/*.pdf"))
        doc_files = list(directory_path.glob("**/*.doc")) + list(directory_path.glob("**/*.docx"))
        md_files = list(directory_path.glob("**/*.md"))
        txt_files = list(directory_path.glob("**/*.txt"))
        total_files = len(pdf_files) + len(doc_files) + len(md_files) + len(txt_files)
        
        print(f"Found {len(pdf_files)} PDF files, {len(doc_files)} Word documents, {len(md_files)} Markdown files, and {len(txt_files)} text files")
        
        # Process PDF files
        for i, file_path in enumerate(pdf_files):
            try:
                print(f"Processing PDF {i+1}/{len(pdf_files)}: {file_path.name}")
                documents = self.process_pdf(file_path)
                print(f"Extracted {len(documents)} chunks from {file_path.name}")
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing PDF {file_path}: {str(e)}")
        
        # Process Word documents
        for i, file_path in enumerate(doc_files):
            try:
                print(f"Processing Word document {i+1}/{len(doc_files)}: {file_path.name}")
                documents = self.process_docx(file_path)
                print(f"Extracted {len(documents)} chunks from {file_path.name}")
                all_documents.extend(documents)
            except Exception as e:
                print(f"Error processing Word document {file_path}: {str(e)}")
        
        # Process Markdown files with enhanced content extraction
        if md_files:
            print(f"Processing {len(md_files)} markdown files")
            for i, file_path in enumerate(md_files):
                try:
                    print(f"Processing markdown file {i+1}/{len(md_files)}: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # Extract any MUST SEE / IMPORTANT sections with priority levels
                    must_see_content = self._extract_must_see_section(text)
                    
                    # Process the document with special attention to must-see content
                    documents = self._split_and_create_documents(text, file_path, must_see_content)
                    print(f"Extracted {len(documents)} chunks from {file_path.name}")
                    
                    # Report on the priority content found
                    if must_see_content:
                        priority_counts = {level: len(content) for level, content in must_see_content.items()}
                        print(f"  ⭐ Found {sum(priority_counts.values())} priority content items in {file_path.name}:")
                        for level, count in priority_counts.items():
                            if level == "critical":
                                print(f"    ⭐⭐⭐ {count} CRITICAL/MUST SEE items")
                            elif level == "important":
                                print(f"    ⭐⭐ {count} IMPORTANT items")
                            elif level == "recommended":
                                print(f"    ⭐ {count} RECOMMENDED items")
                    
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing markdown file {file_path}: {str(e)}")
        
        # Process text files
        if txt_files:
            print(f"Processing {len(txt_files)} text files")
            for i, file_path in enumerate(txt_files):
                try:
                    print(f"Processing text file {i+1}/{len(txt_files)}: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                    
                    # Look for MUST SEE content in text files too
                    must_see_content = self._extract_must_see_section(text)
                    
                    # Process the document
                    documents = self._split_and_create_documents(text, file_path, must_see_content)
                    print(f"Extracted {len(documents)} chunks from {file_path.name}")
                    
                    if must_see_content:
                        priority_counts = {level: len(content) for level, content in must_see_content.items()}
                        print(f"  ⭐ Found {sum(priority_counts.values())} priority content items in {file_path.name}")
                    
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing text file {file_path}: {str(e)}")
        
        # Analyze the extracted chunks for improved reporting
        must_see_chunks = sum(1 for doc in all_documents if doc["metadata"].get("has_must_see") == "yes")
        critical_chunks = sum(1 for doc in all_documents if doc["metadata"].get("importance_level") == "critical")
        important_chunks = sum(1 for doc in all_documents if doc["metadata"].get("importance_level") == "important")
        
        # Count unique locations found
        locations = set()
        for doc in all_documents:
            if "chunk_locations" in doc["metadata"] and doc["metadata"]["chunk_locations"]:
                for loc in doc["metadata"]["chunk_locations"].split(", "):
                    locations.add(loc)
        
        # Count themes found
        themes = {}
        for doc in all_documents:
            if "chunk_themes" in doc["metadata"] and doc["metadata"]["chunk_themes"]:
                for theme in doc["metadata"]["chunk_themes"].split(", "):
                    themes[theme] = themes.get(theme, 0) + 1
        
        # Report detailed statistics
        print(f"\nProcessed {total_files} files, created {len(all_documents)} document chunks")
        print(f"Priority content: {must_see_chunks} chunks with priority information")
        print(f"  - {critical_chunks} critical/must-see chunks")
        print(f"  - {important_chunks} important chunks")
        print(f"Locations found: {len(locations)}")
        print(f"Top themes: {', '.join(sorted(themes, key=themes.get, reverse=True)[:5])}")
        
        return all_documents
    
    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a PDF file and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of document chunks with metadata
        """
        print(f"Processing PDF: {file_path}")
        text = ""
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF {file_path}: {str(e)}")
            return []
        
        return self._split_and_create_documents(text, file_path, "")
    
    def process_docx(self, file_path: Path) -> List[Dict[str, Any]]:
        """
        Process a DOCX file and extract text.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            List of document chunks with metadata
        """
        print(f"Processing DOCX: {file_path}")
        text = ""
        
        try:
            doc = Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX {file_path}: {str(e)}")
            return []
        
        return self._split_and_create_documents(text, file_path, "")
    
    def _extract_must_see_section(self, text: str) -> dict:
        """
        Extract MUST SEE or IMPORTANT section from markdown text with improved pattern matching.
        
        Args:
            text: Markdown text content
            
        Returns:
            Dictionary with must-see content and importance level, or empty dict if not found
        """
        import re
        
        # Enhanced patterns for different urgency levels
        patterns = {
            # Most important - MUST SEE with strong emphasis
            "critical": [
                r'#+\s*(?:MUST[-\s]?SEE|CRITICAL)\s*:?\s*.*?(?=^#+\s|\Z)',  # Headers (MUST SEE or CRITICAL)
                r'\*\*\s*(?:MUST[-\s]?SEE|CRITICAL|ESSENTIAL)\s*:?\s*.*?(?=\*\*|\n\n|\Z)',  # Bold markers
                r'> \*\*(?:MUST[-\s]?SEE|CRITICAL)\*\*:?\s*.*?(?=^>|\n\n|\Z)'  # Blockquote with bold
            ],
            # Important information 
            "important": [
                r'#+\s*IMPORTANT\s*:?\s*.*?(?=^#+\s|\Z)',  # Headers (IMPORTANT)
                r'\*\*\s*IMPORTANT\s*:?\s*.*?(?=\*\*|\n\n|\Z)',  # Bold IMPORTANT
                r'> \*\*IMPORTANT\*\*:?\s*.*?(?=^>|\n\n|\Z)'  # Blockquote with bold
            ],
            # Recommended items (lower priority)
            "recommended": [
                r'#+\s*RECOMMENDED\s*:?\s*.*?(?=^#+\s|\Z)',  # Headers (RECOMMENDED)
                r'\*\*\s*RECOMMENDED\s*:?\s*.*?(?=\*\*|\n\n|\Z)'  # Bold RECOMMENDED
            ]
        }
        
        results = {}
        
        # Search for each pattern type
        for importance_level, pattern_list in patterns.items():
            for pattern in pattern_list:
                compiled_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL | re.IGNORECASE)
                matches = compiled_pattern.findall(text)
                
                if matches:
                    if importance_level not in results:
                        results[importance_level] = []
                    results[importance_level].extend([match.strip() for match in matches])
        
        # Also look for inline important markers
        inline_patterns = [
            (r'\*\*MUST SEE:\*\*\s*(.*?)(?=\n\n|\Z)', "critical"),
            (r'\*\*IMPORTANT:\*\*\s*(.*?)(?=\n\n|\Z)', "important"),
            (r'\*\*RECOMMENDED:\*\*\s*(.*?)(?=\n\n|\Z)', "recommended"),
            (r'MUST SEE:\s*(.*?)(?=\n\n|\Z)', "critical"),
            (r'IMPORTANT:\s*(.*?)(?=\n\n|\Z)', "important")
        ]
        
        for pattern, level in inline_patterns:
            inline_matches = re.findall(pattern, text, re.IGNORECASE)
            if inline_matches:
                if level not in results:
                    results[level] = []
                results[level].extend([match.strip() for match in inline_matches])
        
        # Look for list items with important markers
        list_patterns = [
            (r'[-*]\s*\*\*MUST SEE\*\*:?\s*(.*?)(?=\n[-*]|\n\n|\Z)', "critical"),
            (r'[-*]\s*\*\*IMPORTANT\*\*:?\s*(.*?)(?=\n[-*]|\n\n|\Z)', "important"),
            (r'[-*]\s*\*\*RECOMMENDED\*\*:?\s*(.*?)(?=\n[-*]|\n\n|\Z)', "recommended")
        ]
        
        for pattern, level in list_patterns:
            list_matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
            if list_matches:
                if level not in results:
                    results[level] = []
                results[level].extend([match.strip() for match in list_matches])
                
        return results
    
    def _split_and_create_documents(self, text: str, file_path: Path, must_see_content: dict = None) -> List[Dict[str, Any]]:
        """
        Split text into chunks and create document objects with improved handling of priority content.
        
        Args:
            text: Text to split
            file_path: Original file path for metadata
            must_see_content: Dictionary of MUST SEE/IMPORTANT content by priority level
            
        Returns:
            List of document chunks with metadata
        """
        if must_see_content is None:
            must_see_content = {}
            
        # Extract potential entities from text (enhanced extraction)
        entities = self._extract_entities(text)
        
        # Extract location name from the first line of the file (if it's a title)
        location_name = ""
        first_line = text.strip().split("\n")[0] if text.strip() else ""
        if first_line.startswith("# "):
            location_name = first_line.replace("#", "").strip().split(":")[0].strip()
        
        # If no location name found in title, try to extract it from the content
        if not location_name:
            import re
            # Look for location names in headings and emphasized text
            location_patterns = [
                r'#+\s+([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)',  # Find headings with capitalized words
                r'\*\*([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)*)\*\*'  # Find bold text with capitalized words
            ]
            
            for pattern in location_patterns:
                matches = re.findall(pattern, text)
                if matches:
                    # Take the first match as the potential location name
                    location_name = matches[0]
                    break
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create a flat list of all must-see content pieces
        all_must_see_content = []
        if must_see_content:
            for level, content_list in must_see_content.items():
                for content in content_list:
                    all_must_see_content.append((content, level))
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Determine if this chunk contains must-see info and at what priority level
            has_must_see = "no"
            importance_level = ""
            matching_content = ""
            
            for content, level in all_must_see_content:
                if content in chunk:
                    # If multiple levels match, take the highest priority one
                    if level == "critical" or (level == "important" and importance_level != "critical"):
                        has_must_see = "yes"
                        importance_level = level
                        matching_content = content
                        break
            
            # Also mark the first chunk if it contains the document title/intro and there's any must-see content
            if i == 0 and all_must_see_content:
                has_must_see = "yes"
                if not importance_level:
                    importance_level = "document_intro"
            
            # Extract any locations mentioned in the chunk
            import re
            chunk_locations = []
            location_matches = re.findall(r'\b([A-Z][a-zA-Z]+(?:[\s-][A-Z][a-zA-Z]+)*)\b', chunk)
            common_non_locations = {'I', 'My', 'Me', 'Mine', 'The', 'A', 'An', 'And', 'Or', 'But', 'For', 'With', 'To', 'From'}
            for loc in location_matches:
                if loc not in common_non_locations and loc not in chunk_locations:
                    chunk_locations.append(loc)
            
            chunk_locations_str = ", ".join(chunk_locations) if chunk_locations else ""
            
            # Extract any themes mentioned in the chunk
            theme_keywords = {
                "beach": ["beach", "coast", "sea", "ocean", "shore", "seaside", "sand"],
                "food": ["food", "eat", "cuisine", "culinary", "gastronomy", "restaurant", "dining"],
                "culture": ["culture", "history", "historical", "traditional", "ancient", "museum", "heritage"],
                "adventure": ["adventure", "hiking", "trekking", "climbing", "rafting", "diving"],
                "relaxation": ["relax", "relaxation", "peaceful", "quiet", "calm", "spa", "wellness"],
                "nightlife": ["nightlife", "party", "club", "bar", "disco", "dancing"],
                "shopping": ["shopping", "shop", "market", "store", "mall"],
                "family": ["family", "children", "kids", "child", "family-friendly"]
            }
            
            chunk_themes = []
            for theme, keywords in theme_keywords.items():
                if any(keyword in chunk.lower() for keyword in keywords):
                    chunk_themes.append(theme)
            
            chunk_themes_str = ", ".join(chunk_themes) if chunk_themes else ""
            
            # Flatten entities to be compatible with ChromaDB metadata requirements
            # ChromaDB doesn't allow nested dictionaries in metadata
            flattened_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "chunk_id": i,
                "entities_locations": entities.get("locations", ""),
                "entities_hotels": entities.get("hotels", ""),
                "entities_activities": entities.get("activities", ""),
                "has_must_see": has_must_see,
                "importance_level": importance_level,
                "location_name": location_name,
                "chunk_locations": chunk_locations_str,
                "chunk_themes": chunk_themes_str
            }
            
            # Create enhanced content for must-see chunks to make them more findable
            if has_must_see == "yes":
                # Construct a prefix that highlights the importance and content type
                prefix = ""
                if importance_level == "critical":
                    prefix = f"⭐⭐⭐ CRITICAL TRAVEL ADVICE - MUST SEE FOR {location_name}:\n\n"
                elif importance_level == "important":
                    prefix = f"⭐⭐ IMPORTANT TRAVEL ADVICE FOR {location_name}:\n\n"
                else:
                    prefix = f"⭐ TRAVEL ADVICE FOR {location_name}:\n\n"
                
                # Add theme information if available
                if chunk_themes:
                    theme_info = f"[THEMES: {', '.join(chunk_themes)}] "
                    prefix = prefix.replace(":\n\n", f": {theme_info}\n\n")
                
                enhanced_text = f"{prefix}{chunk}"
                doc = {
                    "text": enhanced_text,
                    "metadata": flattened_metadata
                }
            else:
                doc = {
                    "text": chunk,
                    "metadata": flattened_metadata
                }
            
            documents.append(doc)
        
        return documents
    
    def _extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract potential travel-related entities from text.
        This is a simplified implementation that could be enhanced with NER.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            Dictionary of entity types and their values as JSON strings
        """
        # This is a simplified implementation
        # In a real application, you would use NER or other ML techniques
        entities = {
            "locations": [],
            "hotels": [],
            "activities": []
        }
        
        # Very basic keyword-based extraction for demonstration
        lines = text.split("\n")
        for line in lines:
            line = line.strip()
            # Simple heuristics - could be replaced with more sophisticated NER
            if "hotel" in line.lower() or "resort" in line.lower() or "hostel" in line.lower():
                entities["hotels"].append(line)
            elif "visit" in line.lower() or "tour" in line.lower() or "explore" in line.lower():
                entities["activities"].append(line)
            elif "city" in line.lower() or "town" in line.lower() or "destination" in line.lower():
                entities["locations"].append(line)
        
        # Convert entity lists to strings for ChromaDB compatibility
        flattened_entities = {}
        for key, values in entities.items():
            # Join lists into strings with a max length to prevent errors
            if values:
                flat_value = "; ".join(values[:5])  # Limit to 5 items per category
                if len(flat_value) > 500:  # Limit string length
                    flat_value = flat_value[:500] + "..."
                flattened_entities[key] = flat_value
            else:
                flattened_entities[key] = ""
                
        return flattened_entities