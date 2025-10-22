# backend/logic/analysis_logic.py
import os
import json
import re
from openai import AsyncOpenAI
from typing import Dict, List, Optional

from . import prompts

# Initialize Client
from dotenv import load_dotenv
from pathlib import Path

dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)
api_key = os.getenv("FPT_API_KEY")
base_url = os.getenv("FPT_API_BASE")
if not api_key:
    raise ValueError("CRITICAL ERROR: FPT_API_KEY is not set.")
client = AsyncOpenAI(api_key=api_key, base_url=base_url)


def find_and_parse_json(text: str) -> Dict:
    """Robust JSON parser handling markdown, whitespace, and string boundaries."""
    text = text.strip()
    
    # Try markdown code fences first
    match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    
    if match:
        json_str = match.group(1).strip()
    else:
        # Find outermost JSON object by tracking braces and strings
        depth = 0
        start_index = -1
        end_index = -1
        in_string = False
        escape_next = False
        
        for i, char in enumerate(text):
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\' and in_string:
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    if depth == 0:
                        start_index = i
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0 and start_index != -1:
                        end_index = i
                        break
        
        if start_index == -1 or end_index == -1:
            raise ValueError("No complete JSON object found in response")

        json_str = text[start_index:end_index + 1].strip()
    
    try:
        parsed_data = json.loads(json_str)
        
        if not isinstance(parsed_data, dict):
            raise ValueError(f"Parsed JSON is not an object, got {type(parsed_data)}")
            
        return parsed_data
    except json.JSONDecodeError as e:
        print("--- JSON PARSING FAILED ---")
        print(f"Attempted:\n{repr(json_str[:300])}")
        print(f"Error at position {e.pos}: {e.msg}")
        raise ValueError(f"JSON parsing failed: {str(e)}")


async def _make_llm_call(prompt: str, model_name: str) -> Dict:
    """Make a single LLM call and parse JSON response."""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return find_and_parse_json(content)
    except Exception as e:
        print(f"LLM call failed: {e}")
        raise


def deduplicate_relationships(relationships: List[Dict]) -> List[Dict]:
    """
    Deduplicate relationships: keep only one relationship per (source, target) pair.
    Prefer more specific labels over generic ones.
    """
    # Map: (source, target) -> relationship
    seen = {}
    
    # Generic/low-specificity labels to deprioritize
    generic_labels = {"is related to", "is associated with", "connects to"}
    
    for rel in relationships:
        source = rel.get("source")
        target = rel.get("target")
        label = rel.get("natural_language_label", "")
        
        if not source or not target:
            continue
        
        key = (source, target)
        
        # If we haven't seen this pair, add it
        if key not in seen:
            seen[key] = rel
        else:
            # If new one is more specific (not generic), replace
            existing_label = seen[key].get("natural_language_label", "")
            is_new_generic = label in generic_labels
            is_existing_generic = existing_label in generic_labels
            
            if is_existing_generic and not is_new_generic:
                seen[key] = rel
            # Otherwise keep the existing one
    
    return list(seen.values())


async def perform_full_analysis(text: str, model_name: str, task_id: Optional[str] = None) -> Dict:
    """
    Multi-pass analysis with validation, deduplication, and progress tracking.
    
    Args:
        text: The text to analyze
        model_name: LLM model to use
        task_id: Optional task ID for progress tracking
    
    Returns:
        Dict containing entities, relationships, and hierarchy
    """
    
    # Import progress tracker only if task_id is provided
    progress_tracker = None
    if task_id:
        from ..progress_tracker import progress_tracker as pt
        progress_tracker = pt
    
    try:
        # --- PASS 1: Comprehensive Entity Extraction ---
        if progress_tracker:
            await progress_tracker.update_progress(task_id, 1, "Extracting entities from text...")
        
        print("\n[PASS 1/5] Extracting comprehensive entity list...")
        entity_prompt = prompts.COMPREHENSIVE_ENTITY_EXTRACTION_PROMPT.format(text=text)
        entities_data = await _make_llm_call(entity_prompt, model_name)
        entities = entities_data.get("entities", [])
        
        if not entities:
            raise ValueError("No entities extracted in Pass 1")
        print(f"✓ Extracted {len(entities)} entities")
        
        # --- PASS 2: Hierarchy Extraction (with validation) ---
        if progress_tracker:
            await progress_tracker.update_progress(task_id, 2, "Building logical hierarchy...")
        
        print("[PASS 2/5] Extracting logical hierarchy...")
        hier_prompt = prompts.HIERARCHY_EXTRACTION_PROMPT.format(text=text)
        
        try:
            hierarchy_data = await _make_llm_call(hier_prompt, model_name)
            hierarchy = hierarchy_data.get("hierarchy", [])
        except Exception as e:
            print(f"⚠ Hierarchy extraction failed with error: {e}")
            print("Full response parsing failed. Attempting recovery...")
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": hier_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            raw_content = response.choices[0].message.content
            print(f"RAW LLM RESPONSE:\n{repr(raw_content[:800])}\n")
            raise ValueError(f"Could not parse hierarchy response: {e}")
        
        # Validate hierarchy is not empty
        if not hierarchy:
            print("⚠ WARNING: Hierarchy extraction returned empty. Retrying...")
            hierarchy_data = await _make_llm_call(hier_prompt, model_name)
            hierarchy = hierarchy_data.get("hierarchy", [])
            
            if not hierarchy:
                raise ValueError("Hierarchy extraction failed twice. Cannot proceed without logical flow.")
        
        print(f"✓ Extracted {len(hierarchy)} hierarchy levels")
        
        # --- PASS 3: Relationship Inference ---
        if progress_tracker:
            await progress_tracker.update_progress(task_id, 3, "Inferring relationships between entities...")
        
        print("[PASS 3/5] Inferring relationships between entities...")
        entities_json = json.dumps(entities, indent=2)
        rel_prompt = prompts.RELATIONSHIP_INFERENCE_PROMPT.format(
            entities_json=entities_json,
            text=text
        )
        relationships_data = await _make_llm_call(rel_prompt, model_name)
        relationships = relationships_data.get("relationships", [])
        
        print(f"✓ Inferred {len(relationships)} relationships (before deduplication)")
        
        # --- PASS 4: Relationship Deduplication ---
        if progress_tracker:
            await progress_tracker.update_progress(task_id, 4, "Deduplicating relationships...")
        
        print("[PASS 4/5] Deduplicating relationships...")
        relationships = deduplicate_relationships(relationships)
        print(f"✓ Deduplicated to {len(relationships)} unique relationships")
        
        # --- PASS 5: Entity Enrichment (optional, can skip if too slow) ---
        if progress_tracker:
            await progress_tracker.update_progress(task_id, 5, "Enriching entity descriptions...")
        
        print("[PASS 5/5] Enriching entity descriptions with relational context...")
        try:
            relationships_json = json.dumps(relationships, indent=2)
            enrich_prompt = prompts.ENTITY_CONTEXT_ENRICHMENT_PROMPT.format(
                entities_json=entities_json,
                relationships_json=relationships_json,
                text=text
            )
            enriched_data = await _make_llm_call(enrich_prompt, model_name)
            enriched_entities = enriched_data.get("entities", [])
            
            if enriched_entities:
                entities = enriched_entities
                print("✓ Entity descriptions enriched")
            else:
                print("⚠ Entity enrichment returned empty, using original entities")
        except Exception as e:
            print(f"⚠ Entity enrichment failed (continuing with original): {e}")
        
        # Assemble final output
        analysis_data = {
            "hierarchy": hierarchy,
            "entities": entities,
            "relationships": relationships
        }

        # ==================================================================
        # --- NEW FINAL STEP: Prune REDUNDANT Lonely Nodes ---
        # ==================================================================
        print("[FINAL STEP] Pruning redundant lonely nodes...")
        
        # 1. Get all entities that are part of a relationship
        connected_entity_names = set()
        for rel in relationships:
            connected_entity_names.add(rel.get("source"))
            connected_entity_names.add(rel.get("target"))

        # 2. Create a list of the *full text* of connected entity names
        #    We will use this to check for substrings.
        connected_name_list = list(connected_entity_names)
        
        # 3. Separate entities into connected vs. lonely
        connected_entities = []
        lonely_entities = []
        for entity in entities:
            if entity.get("name") in connected_entity_names:
                connected_entities.append(entity)
            else:
                lonely_entities.append(entity)
        
        # 4. Iterate through lonely entities and decide which ones to keep
        final_entities = list(connected_entities) # Start our final list with all connected entities
        pruned_count = 0
        
        for lonely_entity in lonely_entities:
            lonely_name = lonely_entity.get("name")
            if not lonely_name:
                pruned_count += 1
                continue # Prune entities with no name

            # Check if this lonely name is just a redundant part of a connected entity
            is_redundant = False
            for connected_name in connected_name_list:
                if connected_name:
                    # e.g., if lonely "BCI" is in connected "Synapse-1 BCI"
                    if lonely_name in connected_name and lonely_name != connected_name:
                        is_redundant = True
                        break
            
            if not is_redundant:
                # This is a "meaningful" lonely node, so we keep it
                final_entities.append(lonely_entity)
            else:
                # This is a redundant substring, so we prune it
                pruned_count += 1

        print(f"✓ Pruned {pruned_count} redundant lonely node(s).")
        print(f"✓ Kept {len(final_entities) - len(connected_entities)} meaningful isolated node(s).")
        
        # Overwrite the original list with our final, filtered list
        entities = final_entities 
        # ==================================================================
        
        # Assemble final output
        analysis_data = {
            "hierarchy": hierarchy,
            "entities": entities,  # Use the new, intelligently pruned list
            "relationships": relationships
        }
        
        if progress_tracker:
            await progress_tracker.complete_task(task_id, success=True)
        
        print(f"\n✓ Analysis complete!")
        print(f"  - Hierarchy: {len(hierarchy)} levels")
        print(f"  - Entities: {len(entities)}")
        print(f"  - Relationships: {len(relationships)}")
        return analysis_data

    except Exception as e:
        if progress_tracker:
            await progress_tracker.complete_task(task_id, success=False, error=str(e))
        print(f"\nError in perform_full_analysis: {e}")
        raise ValueError(f"Analysis failed: {e}")