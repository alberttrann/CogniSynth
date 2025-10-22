# backend/logic/entity_resolution.py (COMPLETE REWRITE)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict, Tuple
from openai import AsyncOpenAI
import asyncio
import json
import re
from collections import defaultdict

from . import prompts
# --- ADD THESE NEW IMPORTS ---
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
# -----------------------------


import re # Make sure 're' is imported

def normalize_entity_name(name: str) -> str:
    """
    Robust normalization for entity matching.
    Handles: "Synapse-1 BCI" vs "Synapse-1 Brain-Computer Interface (BCI)"
    Returns a "|" separated string of keys.
    """
    if not name:
        return ""
    
    keys = set()
    name_lower = name.lower()
    
    # 1. Extract acronym from parentheses, e.g., "...(BCI)"
    paren_acronym_match = re.search(r'\(([A-Z]{2,5})\)', name) # Must be caps in parens
    paren_acronym = paren_acronym_match.group(1).lower() if paren_acronym_match else ""
    
    # 2. Get the base name (everything outside parentheses)
    base_name_paren_removed = re.sub(r'\s*\([^)]*\)', '', name_lower).strip()
    
    # 3. Clean the base name
    cleaned_base = re.sub(r'[^\w\s-]', '', base_name_paren_removed)
    cleaned_base = re.sub(r'\s+', ' ', cleaned_base).strip()
    if cleaned_base:
        keys.add(cleaned_base) # e.g., "synapse-1 brain-computer interface"

    # 4. Add the parenthesis-based acronym
    if paren_acronym:
        keys.add(paren_acronym) # e.g., "bci"

    # 5. Check for trailing acronyms IF no parenthesis acronym was found
    if not paren_acronym and cleaned_base:
        words = re.split(r'[\s-]+', cleaned_base) # "synapse-1 bci" -> ["synapse", "1", "bci"]
        if len(words) > 1:
            last_word_lower = words[-1]
            
            # Check if last word in *original* name was all caps
            original_words = re.split(r'[\s-]+', name)
            if original_words[-1].isupper() and len(original_words[-1]) >= 2:
                keys.add(last_word_lower) # e.g., "bci"
    
    # 6. Handle case where name IS the acronym, e.g., "BCI"
    # If no parens, and the *entire original name* is uppercase
    if not paren_acronym and name.isupper() and len(name) >= 2:
        keys = {name_lower} # Only key is the acronym itself

    if not keys:
        return ""
        
    return "|".join(sorted(list(keys)))


class EntityClusterer:
    """
    Uses e5-instruct embeddings to find candidate entity matches.
    This class is now model-specific to handle the required prompting.
    """
    
    def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
        # Load the tokenizer and model from HuggingFace
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        # Use the threshold you set
        self.similarity_threshold = 0.88
        print(f"✅ EntityClusterer initialized with {model_name}")

    @staticmethod
    def _average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """Helper function from e5-instruct model card."""
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

    def _format_query(self, signature: str) -> str:
        """
        Formats the entity signature into the e5-instruct prompt format.
        We use an instruction appropriate for semantic similarity/clustering.
        """
        instruction = "Instruct: Generate a descriptive embedding for the following entity to find duplicates."
        return f'{instruction}\nQuery: {signature}'

    def create_entity_signature(self, entity: Dict) -> str:
        """Create rich text signature for embedding (the "Query" part)."""
        name = entity.get("name", "")
        category = entity.get("category", "")
        description = entity.get("description", "")[:250]
        
        # This signature is the "Query"
        return f"[{category}] {name}: {description}"

    def _encode_signatures(self, signatures: List[str]) -> np.ndarray:
        """
        Encodes a batch of signatures using the e5-instruct model,
        prompting, pooling, and normalization.
        """
        # 1. Format all signatures with the instruction prompt
        formatted_queries = [self._format_query(sig) for sig in signatures]

        # 2. Tokenize the batch
        batch_dict = self.tokenizer(
            formatted_queries, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors='pt'
        )

        # 3. Get model outputs
        with torch.no_grad():
            outputs = self.model(**batch_dict)

        # 4. Perform average pooling
        embeddings = self._average_pool(
            outputs.last_hidden_state,
            batch_dict['attention_mask']
        )

        # 5. Normalize embeddings (CRITICAL step)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        
        # 6. Return as numpy array for scikit-learn
        return normalized_embeddings.cpu().numpy()

    def find_candidate_matches(
        self, 
        entities_by_doc: Dict[str, List[Dict]]
    ) -> List[Tuple[str, str, float]]:
        """Returns candidate matches above similarity threshold."""
        
        all_entities = []
        entity_map = {}
        
        for doc_id, entities in entities_by_doc.items():
            for entity in entities:
                key = f"{doc_id}::{entity['name']}"
                all_entities.append({
                    "key": key,
                    "signature": self.create_entity_signature(entity),
                    "doc_id": doc_id
                })
                entity_map[key] = entity
        
        if len(all_entities) < 2:
            return []
        
        # 7. Compute embeddings using the new e5-specific method
        print(f"Generating {len(all_entities)} embeddings with e5-instruct...")
        signatures = [e["signature"] for e in all_entities]
        embeddings = self._encode_signatures(signatures)
        print("...Embeddings generated.")
        
        # 8. Find high-similarity pairs (same as before)
        similarity_matrix = cosine_similarity(embeddings)
        matches = []
        
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                score = similarity_matrix[i][j]
                
                if score >= self.similarity_threshold:
                    # Only match entities from different documents
                    if all_entities[i]["doc_id"] != all_entities[j]["doc_id"]:
                        matches.append((
                            all_entities[i]["key"],
                            all_entities[j]["key"],
                            score
                        ))
        
        return sorted(matches, key=lambda x: x[2], reverse=True)




class EntityResolver:
    """Orchestrates multi-stage entity resolution."""
    
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
        self.clusterer = EntityClusterer()
    
    async def _validate_match_with_llm(
        self,
        entity1: Dict,
        entity2: Dict,
        doc1_title: str,
        doc2_title: str
    ) -> Dict:
        """Ask LLM if two entities are the same."""
        prompt = prompts.ENTITY_MATCHING_VALIDATION_PROMPT.format(
            entity1_name=entity1["name"],
            entity1_category=entity1.get("category", "Unknown"),
            entity1_description=entity1.get("description", ""),
            doc1_title=doc1_title,
            entity2_name=entity2["name"],
            entity2_category=entity2.get("category", "Unknown"),
            entity2_description=entity2.get("description", ""),
            doc2_title=doc2_title
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"    ⚠️ LLM validation failed: {e}")
            return {"is_match": False, "confidence": 0.0}
    
    async def _create_canonical_entity(
        self,
        entity_cluster: List[Dict],
        doc_titles: Dict[str, str]
    ) -> Dict:
        """Create canonical merged entity from cluster."""
        variants_for_llm = [
            {
                "name": e["name"],
                "category": e.get("category", "Unknown"),
                "description": e.get("description", ""),
                "source_doc": doc_titles.get(e.get("_doc_id"), "Unknown")
            }
            for e in entity_cluster
        ]
        
        prompt = prompts.ENTITY_CLUSTER_NAMING_PROMPT.format(
            entity_variants_json=json.dumps(variants_for_llm, indent=2)
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            result = json.loads(response.choices[0].message.content)
            
            # --- FIX: Remove canonical name from aliases ---
            if "canonical_name" in result and "aliases" in result:
                canonical = result["canonical_name"]
                # Filter out the canonical name and any exact duplicates
                unique_aliases = sorted(
                    list(set(a for a in result["aliases"] if a.strip() != canonical.strip())),
                    key=len,
                    reverse=True
                )
                result["aliases"] = unique_aliases
            # --- END FIX ---
            
            return result
            
        except Exception as e:
            print(f"    ⚠️ Canonical entity creation failed: {e}")
            # Fallback: use longest name
            longest = max(entity_cluster, key=lambda e: len(e["name"]))
            
            # --- FIX: Apply same filtering logic to fallback ---
            canonical_fallback = longest["name"]
            aliases_fallback = [
                e["name"] for e in entity_cluster 
                if e["name"].strip() != canonical_fallback.strip()
            ]
            # --- END FIX ---
            
            return {
                "canonical_name": canonical_fallback,
                "aliases": aliases_fallback,
                "merged_description": longest.get("description", ""),
                "category": longest.get("category", "Unknown")
            }
    
    async def _wrap_singleton(self, entity: Dict) -> Dict:
        """Wrap singleton entity in canonical format."""
        return {
            "canonical_name": entity["name"],
            "category": entity.get("category", "Unknown"),
            "merged_description": entity.get("description", ""),
            "aliases": []
        }
    
    async def resolve_entities(
        self,
        entities_by_doc: Dict[str, List[Dict]],
        doc_metadata: Dict[str, Dict]
    ) -> Tuple[List[Dict], Dict[str, str]]:
        """
        Multi-stage entity resolution with normalization.
        """
        
        print("\n[ENTITY RESOLUTION] Starting multi-stage resolution...")
        
        # Prepare data
        all_entities_flat = []
        entity_lookup = {}
        
        for doc_id, entities in entities_by_doc.items():
            for entity in entities:
                key = f"{doc_id}::{entity['name']}"
                entity['_key'] = key
                entity['_doc_id'] = doc_id
                all_entities_flat.append(entity)
                entity_lookup[key] = entity
        
        # Union-Find data structure
        parent = {e['_key']: e['_key'] for e in all_entities_flat}
        
        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x, y):
            root_x, root_y = find(x), find(y)
            if root_x != root_y:
                parent[root_y] = root_x
        
        # STAGE 1: Exact normalization matching
        print("[1/3] Normalizing and finding exact matches...")
        normalized_map = defaultdict(list)
        
        for entity in all_entities_flat:
            norm_name = normalize_entity_name(entity['name'])
            if norm_name:
                # Handle dual keys (name|acronym)
                if '|' in norm_name:
                    parts = norm_name.split('|')
                    for part in parts:
                        normalized_map[part].append(entity['_key'])
                else:
                    normalized_map[norm_name].append(entity['_key'])
        
        exact_matches = 0
        for norm_name, keys in normalized_map.items():
            if len(keys) > 1:
                # Merge entities with same normalized name from different docs
                for i in range(len(keys)):
                    for j in range(i + 1, len(keys)):
                        e1, e2 = entity_lookup[keys[i]], entity_lookup[keys[j]]
                        if e1['_doc_id'] != e2['_doc_id']:
                            # Also check category match
                            if e1.get('category') == e2.get('category'):
                                union(keys[i], keys[j])
                                exact_matches += 1
        
        print(f"    ✅ Found {exact_matches} exact matches via normalization")
        
        # STAGE 2: Embedding-based candidates
        print("[2/3] Finding semantic candidates via embeddings...")
        candidates = self.clusterer.find_candidate_matches(entities_by_doc)
        print(f"    Found {len(candidates)} semantic candidate pairs")
        
        # STAGE 3: LLM validation
        print("[3/3] Validating candidates with LLM...")
        validation_tasks = []
        
        for key1, key2, score in candidates:
            # Skip if already merged
            if find(key1) == find(key2):
                continue
            
            doc1_id = key1.split("::")[0]
            doc2_id = key2.split("::")[0]
            
            task = self._validate_match_with_llm(
                entity_lookup[key1],
                entity_lookup[key2],
                doc_metadata[doc1_id]["title"],
                doc_metadata[doc2_id]["title"]
            )
            validation_tasks.append((key1, key2, task))
        
        llm_confirmations = 0
        if validation_tasks:
            validation_results = await asyncio.gather(
                *[task for _, _, task in validation_tasks],
                return_exceptions=True
            )
            
            for i, result in enumerate(validation_results):
                if isinstance(result, Exception):
                    continue
                
                key1, key2, _ = validation_tasks[i]
                if result.get("is_match") and result.get("confidence", 0) >= 0.7:
                    union(key1, key2)
                    llm_confirmations += 1
        
        print(f"    ✅ Confirmed {llm_confirmations} additional matches via LLM")
        
        # Create canonical entities
        print("[FINALIZATION] Creating canonical entities...")
        clusters = defaultdict(list)
        for entity in all_entities_flat:
            root = find(entity['_key'])
            clusters[root].append(entity)
        
        doc_titles = {doc_id: meta["title"] for doc_id, meta in doc_metadata.items()}
        canonical_creation_tasks = []
        
        for root_key, entity_cluster in clusters.items():
            if len(entity_cluster) > 1:
                task = self._create_canonical_entity(entity_cluster, doc_titles)
            else:
                task = self._wrap_singleton(entity_cluster[0])
            canonical_creation_tasks.append((entity_cluster, asyncio.create_task(task)))
        
        canonical_results = await asyncio.gather(
            *[task for _, task in canonical_creation_tasks],
            return_exceptions=True
        )
        
        merged_entities = []
        entity_map = {}
        
        for i, canonical in enumerate(canonical_results):
            if isinstance(canonical, Exception):
                print(f"    ⚠️ Failed to create canonical entity: {canonical}")
                continue
            
            entity_cluster, _ = canonical_creation_tasks[i]
            
            merged_entities.append({
                "name": canonical["canonical_name"],
                "category": canonical["category"],
                "description": canonical["merged_description"],
                "aliases": canonical.get("aliases", []),
                "source_documents": list(set(e["_doc_id"] for e in entity_cluster))
            })
            
            # Map all original names to canonical
            for entity in entity_cluster:
                entity_map[entity["_key"]] = canonical["canonical_name"]
        
        print(f"    ✅ Created {len(merged_entities)} canonical entities from {len(all_entities_flat)} originals")
        print(f"    📊 Reduction: {len(all_entities_flat) - len(merged_entities)} entities merged")
        
        return merged_entities, entity_map