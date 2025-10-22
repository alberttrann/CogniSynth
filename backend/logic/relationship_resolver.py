# backend/logic/relationship_resolver.py
from typing import Dict, List
from typing import Tuple
class RelationshipResolver:
    """
    Handles relationship deduplication and conflict resolution.
    """
    
    def rewrite_relationships(
        self,
        relationships_by_doc: Dict[str, List[Dict]],
        entity_map: Dict[str, str]  # {doc_id::entity_name -> canonical_name}
    ) -> List[Dict]:
        """
        Rewrite all relationships to use canonical entity names.
        """
        rewritten = []
        
        for doc_id, relationships in relationships_by_doc.items():
            for rel in relationships:
                source_key = f"{doc_id}::{rel['source']}"
                target_key = f"{doc_id}::{rel['target']}"
                
                canonical_source = entity_map.get(source_key, rel['source'])
                canonical_target = entity_map.get(target_key, rel['target'])
                
                rewritten.append({
                    "source": canonical_source,
                    "target": canonical_target,
                    "natural_language_label": rel.get("natural_language_label", "related to"),
                    "explanation": rel.get("explanation", ""),
                    "category": rel.get("category", "ASSOCIATIVE"),
                    "_source_doc": doc_id
                })
        
        return rewritten
    
    def deduplicate_and_resolve_conflicts(self, relationships: List[Dict]) -> List[Dict]:
        """
        Deduplicate relationships while preserving directionality.
        CRITICAL: Uses directional keys to maintain source->target semantics.
        """
        from collections import defaultdict
        
        grouped = defaultdict(list)
        
        for rel in relationships:
            # CRITICAL FIX #1: Skip self-loops entirely
            if rel["source"] == rel["target"]:
                print(f"⚠️ Skipping self-loop: {rel['source']} -> {rel['target']}")
                continue
            
            # CRITICAL FIX #2: Use directional key (do NOT sort)
            # This preserves the semantic difference between:
            # (A, "causes", B) vs (B, "is caused by", A)
            key = (rel["source"], rel["target"])
            grouped[key].append(rel)
        
        deduplicated = []
        
        for key, rel_group in grouped.items():
            if len(rel_group) == 1:
                deduplicated.append(rel_group[0])
            else:
                # Keep only the MOST SPECIFIC relationship
                best = self._resolve_relationship_conflict(rel_group)
                deduplicated.append(best)
        
        return deduplicated
    
    # In relationship_resolver.py
    def _resolve_relationship_conflict(self, rel_group: List[Dict]) -> Dict:
        """Choose the best relationship from a group of duplicates."""
        
        # Prioritize specific relationship types
        category_priority = {
            "CAUSAL": 5,          # "causes", "enables", "results in"
            "COMPOSITIONAL": 4,    # "is composed of", "contains"
            "EVIDENTIARY": 3,      # "provides evidence for", "confirms"
            "STRUCTURAL": 2,       # "is part of", "belongs to"
            "TEMPORAL": 1,         # "occurs during", "follows"
            "ASSOCIATIVE": 0       # Generic relationships
        }
        
        # Generic labels to deprioritize
        generic_labels = {"is related to", "is associated with", "connects to"}
        
        def score_relationship(rel: Dict) -> Tuple[int, int, int, int]:
            category_score = category_priority.get(rel.get("category", "ASSOCIATIVE"), 0)
            label = rel.get("natural_language_label", "")
            specificity_score = 0 if label in generic_labels else 1
            
            # --- FIX: Penalize merged explanations ---
            explanation = rel.get("explanation", "")
            merge_penalty = -explanation.count(" | ")  # Each " | " indicates a merge
            # --- END FIX ---
            
            explanation_length = len(explanation)
            
            return (category_score, specificity_score, merge_penalty, explanation_length)
        
        # Choose best relationship
        best_rel = max(rel_group, key=score_relationship)
        
        # --- FIX: Don't merge explanations if they're fundamentally different ---
        all_explanations = [r["explanation"] for r in rel_group if r.get("explanation")]
        
        # Only combine if there are 2-3 similar explanations
        if 2 <= len(all_explanations) <= 3:
            unique_explanations = []
            seen = set()
            for exp in all_explanations:
                exp_normalized = exp.lower().strip()
                if exp_normalized not in seen:
                    unique_explanations.append(exp)
                    seen.add(exp_normalized)
            
            # Only merge if explanations are complementary (not contradictory)
            if len(unique_explanations) <= 2:
                best_rel["explanation"] = " | ".join(unique_explanations)
        # Otherwise, keep only the best explanation (don't merge)
        # --- END FIX ---
        
        return best_rel