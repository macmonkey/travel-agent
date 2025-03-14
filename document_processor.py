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
        total_files = len(pdf_files) + len(doc_files)
        
        print(f"Found {len(pdf_files)} PDF files and {len(doc_files)} Word documents")
        
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
        
        # Also process any markdown files in the directory
        md_files = list(directory_path.glob("**/*.md"))
        if md_files:
            print(f"Found {len(md_files)} markdown files")
            for i, file_path in enumerate(md_files):
                try:
                    print(f"Processing markdown file {i+1}/{len(md_files)}: {file_path.name}")
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                        
                    # Extract any MUST SEE / IMPORTANT sections for special handling
                    must_see_content = self._extract_must_see_section(text)
                    
                    # Process the document with special attention to must-see content
                    documents = self._split_and_create_documents(text, file_path, must_see_content)
                    print(f"Extracted {len(documents)} chunks from {file_path.name}")
                    if must_see_content:
                        print(f"  ⭐ Found MUST SEE / IMPORTANT content in {file_path.name}")
                    all_documents.extend(documents)
                except Exception as e:
                    print(f"Error processing markdown file {file_path}: {str(e)}")
        
        print(f"Processed {total_files + len(md_files)} files, created {len(all_documents)} document chunks")
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
    
    def _extract_must_see_section(self, text: str) -> str:
        """
        Extract MUST SEE or IMPORTANT section from markdown text.
        
        Args:
            text: Markdown text content
            
        Returns:
            Content of MUST SEE/IMPORTANT section, or empty string if not found
        """
        import re
        
        # Look for MUST SEE or IMPORTANT section
        must_see_pattern = re.compile(r'#+\s*MUST[-\s]?SEE.*?(?=^#+\s|\Z)', re.MULTILINE | re.DOTALL | re.IGNORECASE)
        important_pattern = re.compile(r'#+\s*IMPORTANT.*?(?=^#+\s|\Z)', re.MULTILINE | re.DOTALL | re.IGNORECASE)
        
        # Check for both patterns
        must_see_match = must_see_pattern.search(text)
        important_match = important_pattern.search(text)
        
        # Use the first match found
        if must_see_match:
            return must_see_match.group().strip()
        elif important_match:
            return important_match.group().strip()
        else:
            return ""
    
    def _split_and_create_documents(self, text: str, file_path: Path, must_see_content: str = "") -> List[Dict[str, Any]]:
        """
        Split text into chunks and create document objects.
        
        Args:
            text: Text to split
            file_path: Original file path for metadata
            must_see_content: Optional MUST SEE/IMPORTANT content to flag
            
        Returns:
            List of document chunks with metadata
        """
        # Extract potential entities from text (simplified)
        entities = self._extract_entities(text)
        
        # Extract location name from the first line of the file (if it's a title)
        location_name = ""
        first_line = text.strip().split("\n")[0] if text.strip() else ""
        if first_line.startswith("# "):
            location_name = first_line.replace("#", "").strip().split(":")[0].strip()
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        documents = []
        for i, chunk in enumerate(chunks):
            # Determine if this chunk is a high priority (contains must-see info)
            has_must_see = bool(must_see_content and (must_see_content in chunk or i == 0))
            
            # Flatten entities to be compatible with ChromaDB metadata requirements
            # ChromaDB doesn't allow nested dictionaries in metadata
            flattened_metadata = {
                "source": str(file_path),
                "filename": file_path.name,
                "chunk_id": i,
                "entities_locations": entities.get("locations", ""),
                "entities_hotels": entities.get("hotels", ""),
                "entities_activities": entities.get("activities", ""),
                "has_must_see": "yes" if has_must_see else "no",
                "location_name": location_name
            }
            
            # Create special content for must-see chunks to make them more findable
            if has_must_see:
                # Enhance the text to make the MUST SEE content more prominent for the RAG
                enhanced_text = f"IMPORTANT TRAVEL ADVICE - MUST SEE FOR {location_name}:\n\n{chunk}"
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