# backend/logic/utils.py
import re

def normalize_entity_name(name: str) -> str:
    """
    A robust function to normalize entity names for better matching.
    - Lowercases
    - Removes parenthesized acronyms/text
    - Removes punctuation
    - Normalizes whitespace
    """
    if not name:
        return ""
    
    # Lowercase the string
    normalized = name.lower()
    
    # Remove parenthesized content (like acronyms)
    normalized = re.sub(r'\s*\([^)]*\)', '', normalized).strip()
    
    # Remove punctuation except hyphens
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    
    # Normalize whitespace to single spaces
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    return normalized