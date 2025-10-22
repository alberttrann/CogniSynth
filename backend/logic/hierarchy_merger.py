# backend/logic/hierarchy_merger.py

import json
from typing import Any, Dict, List
from openai import AsyncOpenAI

from . import prompts

class HierarchyMerger:
    """
    Merges hierarchies from multiple documents using LLM-based semantic fusion.
    """
    
    def __init__(self, client: AsyncOpenAI, model_name: str):
        self.client = client
        self.model_name = model_name
    
    async def merge_hierarchies(
        self,
        hierarchies_by_doc: Dict[str, List[Dict]],
        doc_metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Merge hierarchies using LLM to identify common threads.
        
        Args:
            hierarchies_by_doc: {document_id: [hierarchy_topics]}
            doc_metadata: {document_id: {title, word_count}}
        
        Returns:
            Merged hierarchy with cross-document synthesis
        """
        print("\n[HIERARCHY FUSION] Merging logical flows...")
        
        # Check if we have any hierarchies at all
        total_topics = sum(len(h) for h in hierarchies_by_doc.values())
        
        if total_topics == 0:
            print("    ⚠️ No hierarchies to merge (all documents returned empty)")
            return []
        
        # Prepare hierarchies for LLM with document context
        hierarchies_with_context = []
        for doc_id, hierarchy in hierarchies_by_doc.items():
            if hierarchy:  # Only include non-empty hierarchies
                doc_title = doc_metadata[doc_id]["title"]
                hierarchies_with_context.append({
                    "document_title": doc_title,
                    "document_id": doc_id,
                    "hierarchy": hierarchy
                })
        
        # If only one document has hierarchy, return it as-is
        if len(hierarchies_with_context) == 1:
            print(f"    ℹ️ Only one document has hierarchy, returning it directly")
            return hierarchies_with_context[0]["hierarchy"]
        
        # Prepare prompt
        prompt = prompts.HIERARCHY_FUSION_PROMPT.format(
            hierarchies_json=json.dumps(hierarchies_with_context, indent=2)
        )
        
        try:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.2  # Slightly higher for creative synthesis
            )
            
            content = response.choices[0].message.content
            result = json.loads(content)
            
            # Try different possible keys for the hierarchy
            merged_hierarchy = (
                result.get("hierarchy") or 
                result.get("merged_hierarchy") or 
                result.get("topics") or
                []
            )
            
            if not merged_hierarchy:
                print("    ⚠️ LLM returned empty hierarchy, using fallback")
                return self._fallback_merge(hierarchies_by_doc, doc_metadata)
            
            print(f"    ✅ Created {len(merged_hierarchy)} merged topics")
            return merged_hierarchy
            
        except Exception as e:
            print(f"    ⚠️ Hierarchy fusion failed: {e}. Using fallback concatenation.")
            return self._fallback_merge(hierarchies_by_doc, doc_metadata)
    
    def _fallback_merge(
        self,
        hierarchies_by_doc: Dict[str, List[Dict]],
        doc_metadata: Dict[str, Dict]
    ) -> List[Dict]:
        """
        Improved fallback: Group semantically similar topics together.
        """
        all_topics = []
        
        # Collect all topics with metadata
        for doc_id, hierarchy in hierarchies_by_doc.items():
            if not hierarchy:
                continue
            
            doc_title = doc_metadata[doc_id]["title"]
            
            for topic in hierarchy:
                all_topics.append({
                    "main_topic": topic.get("main_topic", "Unnamed Topic"),
                    "supporting_details": topic.get("supporting_details", []),
                    "source_document": doc_title,
                    "source_doc_id": doc_id,
                    "_normalized_topic": self._normalize_topic(topic.get("main_topic", ""))
                })
        
        # Group by normalized topic name
        from collections import defaultdict
        topic_groups = defaultdict(list)
        
        for topic in all_topics:
            norm_topic = topic["_normalized_topic"]
            topic_groups[norm_topic].append(topic)
        
        # Merge grouped topics
        merged = []
        for norm_topic, topics in topic_groups.items():
            if len(topics) == 1:
                # Single topic, keep as is
                merged.append(topics[0])
            else:
                # Multiple topics with similar names - merge them
                merged_topic = {
                    "main_topic": topics[0]["main_topic"],  # Use first as canonical
                    "supporting_details": [],
                    "source_documents": list(set(t["source_document"] for t in topics))
                }
                
                # Combine all supporting details, removing duplicates
                seen_details = set()
                for topic in topics:
                    for detail in topic.get("supporting_details", []):
                        detail_norm = detail.lower().strip()
                        if detail_norm not in seen_details:
                            merged_topic["supporting_details"].append(detail)
                            seen_details.add(detail_norm)
                
                merged.append(merged_topic)
        
        print(f"    ℹ️ Fallback merge: {len(all_topics)} topics → {len(merged)} (deduplicated)")
        return merged
    
    def _normalize_topic(self, topic: str) -> str:
        """Normalize topic name for grouping."""
        import re
        
        # Lowercase
        norm = topic.lower().strip()
        
        # Remove numbers at start (like "1. ")
        norm = re.sub(r'^\d+\.\s*', '', norm)
        
        # Remove punctuation
        norm = re.sub(r'[^\w\s]', '', norm)
        
        # Normalize whitespace
        norm = re.sub(r'\s+', ' ', norm).strip()
        
        return norm