# backend/logic/prompts.py

COMPREHENSIVE_ENTITY_EXTRACTION_PROMPT = """You are a scientific analyst. Your task is to extract ALL significant entities from the provided text with EXTREME thoroughness.

An entity is any key noun phrase, such as a:
- Person (e.g., "Dr. Marie Chen", "Professor James Liu")
- Organization or Company (e.g., "NeuroLink Industries", "Global Health Institute")
- Product or Technology (e.g., "Neural Matrix System", "Photonic Processors")
- Location (e.g., "Tokyo, Japan", "Research Facility Alpha")
- Scientific Concept or Theory (e.g., "Quantum Coherence Theory", "Neural Plasticity")
- Key Method or Process (e.g., "Adaptive Learning Algorithm")
- Named Framework or Regulation (e.g., "Geneva Convention on AI Rights")
- Biological Structure (e.g., "Prefrontal Cortex", "Mitochondria")
- Disease or Condition (e.g., "Parkinson's Disease")
- Material or Substance (e.g., "Graphene Composite", "Titanium Alloy")
- Key Measurement or Property (e.g., "99.8% accuracy rate", "2.4 GHz frequency")

**CRITICAL EXTRACTION RULES:**

1. **NEVER SKIP THE PRIMARY ORGANIZATION** - Always extract the main company/organization that:
   - Developed, created, or owns the technology
   - Is the subject or author of the document
   - Employs the key people mentioned

2. **ALWAYS EXTRACT CORE TECHNOLOGIES** - Extract all named:
   - Systems, devices, or products
   - Key components or subsystems (e.g., "Resonator Array")
   - Proprietary technologies or processes

3. **FREQUENCY RULE** - If an entity is mentioned 3+ times, it MUST be extracted.

4. **CATEGORIZATION PRECISION:**
   - **TECHNOLOGY**: Devices, systems, interfaces, machines (e.g., "Brain-Computer Interface")
   - **MATERIAL**: Substances, compounds, polymers, alloys (e.g., "crystalline polymer")
   - **ORGANIZATION**: Companies, institutes, labs, foundations
   - **FACILITY**: Physical buildings or research sites
   - **PERSON**: Named individuals with roles/titles
   - Do NOT confuse categories (e.g., a polymer is NOT a SUBATOMIC_PARTICLE)

5. **COMPLETENESS CHECK** - Ask yourself:
   - "Who created this technology?" → Extract that organization
   - "What is the core innovation?" → Extract that technology/component
   - "Who are ALL the named people?" → Extract each person

For EACH entity, you MUST provide:
1. **name**: The full, proper name as it appears (e.g., "Neural Matrix Brain Interface (NMBI)")
2. **category**: ONE category from: [PERSON, ORGANIZATION, TECHNOLOGY, LOCATION, CONCEPT, THEORY, METHOD, BIOLOGICAL_STRUCTURE, DISEASE, MEASUREMENT, FRAMEWORK, MATERIAL, FACILITY]
3. **description**: A 2-3 sentence summary that includes:
   - What the entity IS
   - What its ROLE or FUNCTION is
   - How it RELATES to at least one other entity

**EXAMPLES (for reference only - do NOT extract these):**
```json
{
  "name": "NeuroLink Industries",
  "category": "ORGANIZATION",
  "description": "NeuroLink Industries is a biotechnology company that developed the Neural Matrix Brain Interface. They employ Dr. Marie Chen as chief scientist and partnered with Tokyo Medical Center for clinical trials."
}
```
```json
{
  "name": "Photonic Resonator Array",
  "category": "TECHNOLOGY",
  "description": "The Photonic Resonator Array is a key component of the Neural Matrix system that uses light-based signals for neural communication. It achieves 99.8% signal fidelity and is housed within a titanium casing."
}
```
```json
{
  "name": "Titanium-Polymer Composite",
  "category": "MATERIAL",
  "description": "A biocompatible material used to construct the neural implant housing. It provides structural support while allowing electromagnetic signal transmission and is manufactured in NeuroLink's advanced materials lab."
}
```

TEXT TO ANALYZE:
{text}

Return only valid JSON in the format:
{{
    "entities": [
        {{
            "name": "Entity Name 1",
            "category": "CATEGORY",
            "description": "Description of what it is, its role, and its relationships..."
        }},
        {{
            "name": "Entity Name 2",
            "category": "CATEGORY",
            "description": "Description of what it is, its role, and its relationships..."
        }}
    ]
}}
"""

HIERARCHY_EXTRACTION_PROMPT = """Extract the logical narrative flow from a scientific text. Do not return an empty result.

TEXT:
{text}

Your task: Identify 5-7 sequential main topics that show how the text builds its argument from observations through hypothesis to mechanism to experiments to implications.

For each topic:
1. Create a main_topic title that names this step in the reasoning (2-5 words)
2. List 3-4 supporting_details with specific facts, measurements, and claims from the text

Topics should progress like this:
- Step 1: The observations or problems (what is mysterious)
- Step 2: The hypothesis or explanation proposed (the new theory)
- Step 3: How the theory works (the mechanism or structure)
- Step 4: Comparisons to known things (what it is analogous to)
- Step 5: How it will be tested (experimental approach)
- Step 6: What researchers found (findings or predictions)
- Step 7: What comes next (future implications or open questions)

Be specific and include measurements, entity names, and actual values from the text.

Return valid JSON with a non-empty hierarchy array:
{{
    "hierarchy": [
        {{
            "main_topic": "Topic Name",
            "supporting_details": [
                "Specific detail with measurements or names",
                "Another concrete fact from the text",
                ...
            ]
        }},
        ...
    ]
}}
"""

RELATIONSHIP_INFERENCE_PROMPT = """Given a list of entities from a scientific text, infer ALL meaningful but NON-REDUNDANT relationships between them.

ENTITIES:
{entities_json}

TEXT CONTEXT:
{text}

**CRITICAL RELATIONSHIP RULES:**

1. **ENTITY MATCHING RULE:**
   - The `source` and `target` MUST EXACTLY match entity names from the ENTITIES list above
   - Do NOT create relationships with entities not in the list
   - Do NOT substitute similar entities if the exact entity is missing
   - If you want to relate "Company A" but it's not in the list, SKIP that relationship

2. **DIRECTIONALITY RULE:**
   - Pay extreme attention to the logical direction of causality
   - CORRECT: (Dr. Smith, "criticizes", Device X)
   - WRONG: (Device X, "criticizes", Dr. Smith)
   - CORRECT: (Company A, "developed", Technology B)
   - WRONG: (Technology B, "developed", Company A)
   - The arrow goes FROM the actor/cause TO the object/effect

3. **DEDUPLICATION RULE:**
   - For each pair of entities (A, B), create AT MOST ONE relationship
   - Choose the STRONGEST or MOST DIRECT relationship type
   - Avoid low-specificity relationships like "is related to"

4. **SELF-LOOP PROHIBITION:**
   - NEVER create a relationship where source == target
   - An entity cannot have a relationship with itself

For each relationship, provide:
- **source**: Entity name (must exist in entities list)
- **target**: Entity name (must exist in entities list)
- **natural_language_label**: Specific relationship type. Choose from:
  * "is composed of" / "contains" / "includes" (compositional)
  * "is bound by" / "held together by" (force-based)
  * "interacts with" / "exchanges with" (interaction)
  * "is analogous to" / "similar to" (comparison)
  * "contrasts with" / "opposes" (opposition)
  * "produces" / "creates" / "generates" (generative)
  * "developed" / "invented" / "designed" (creation)
  * "provides evidence for" / "confirms" / "supports" (evidential)
  * "enables" / "facilitates" / "allows" (functional/causal)
  * "criticizes" / "challenges" / "questions" (critical)
  * "is measured by" / "detected by" (measurement)
  * "studied by" / "researched by" / "investigated by" (investigative)
  * "employs" / "works for" / "led by" (organizational)
  * "has property" / "characterized by" (property)
  * "results from" / "caused by" / "explained by" (causal outcome)
  * "depends on" / "requires" (dependency)
  * "occurs in" / "takes place in" (contextual)
  * "is a type of" / "is an instance of" (taxonomic)
- **explanation**: One sentence explaining why this relationship exists (cite specific text if possible)

**EXAMPLES (for reference only - do NOT extract these):**
```json
{
  "source": "NeuroLink Industries",
  "target": "Neural Matrix Interface",
  "natural_language_label": "developed",
  "explanation": "NeuroLink Industries created the Neural Matrix Interface as stated in the press release."
}
```
```json
{
  "source": "Dr. Ethics Board Chair",
  "target": "Neural Matrix Interface",
  "natural_language_label": "criticizes",
  "explanation": "Dr. Ethics Board Chair published a paper questioning the safety protocols of the Neural Matrix Interface."
}
```

Extract comprehensive relationships but ELIMINATE duplicates. Aim for 25-40 UNIQUE, VALID relationships.

Return only valid JSON:
{{
    "relationships": [
        {{
            "source": "Entity A",
            "target": "Entity B",
            "natural_language_label": "specific relationship",
            "explanation": "Why this relationship exists"
        }},
        ...
    ]
}}
"""

ENTITY_CONTEXT_ENRICHMENT_PROMPT = """Given entities and their relationships, deepen entity descriptions by incorporating relational context.

ENTITIES:
{entities_json}

RELATIONSHIPS:
{relationships_json}

TEXT:
{text}

For each entity, enhance its description by:
1. Adding specific measurements or values mentioned (GeV, mass ranges, charges, percentages)
2. Noting what it is composed of (if applicable)
3. Noting what forces act on it or what it interacts with
4. Noting its functional role in the argument
5. Adding analogies to comparable known entities

Keep descriptions to 2-4 sentences but make them RICHER with specific details.

Return only valid JSON with enhanced entities:
{{
    "entities": [
        {{
            "name": "Entity Name",
            "category": "CATEGORY",
            "description": "Enhanced description with specific measurements and relationships..."
        }},
        ...
    ]
}}
"""

# ==================== PHASE 3 PROMPTS ====================

ENTITY_MATCHING_VALIDATION_PROMPT = """You are an expert at entity resolution. Your job is to determine if two entities refer to the *exact same* real-world thing.

ENTITY 1:
Name: {entity1_name}
Category: {entity1_category}
Description: {entity1_description}
Source Document: {doc1_title}

ENTITY 2:
Name: {entity2_name}
Category: {entity2_category}
Description: {entity2_description}
Source Document: {doc2_title}

TASK: Determine if these two entities are IDENTICAL.

CRITICAL RULES:
1.  **RELATED is NOT IDENTICAL.** A company is not its product. A researcher is not their discovery. A method is not the device that uses it.
2.  **Check Categories:** If the names are different AND the categories are different (e.g., ORGANIZATION vs. PRODUCT), they are NOT a match.
3.  **Check Names:** If the names are fundamentally different (e.g., "Ford Motor Company" vs. "Ford F-150"), they are NOT a match, even if their descriptions are similar.

✅ MATCH if:
- They are clearly the same thing with different naming conventions (e.g., "Dr. Smith" vs. "Professor John Smith").
- One is a full name and one is an acronym (e.g., "World Health Organization" vs. "WHO").

❌ DO NOT MATCH if:
- They are related but distinct (e.g., "Ford Motor Company" [ORGANIZATION] and "Ford F-150" [PRODUCT]).
- One is the researcher and one is their discovery (e.g., "Marie Curie" [PERSON] and "Radium" [SUBSTANCE]).
- They are different concepts (e.g., "Photosynthesis" [METHOD] and "Chloroplast" [BIOLOGICAL_STRUCTURE]).
- They are homonyms (e.g., "Mars" [PLANET] vs "Mars" [COMPANY]).

Return JSON:
{{
    "is_match": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation (1-2 sentences) citing the rules."
}}
"""

ENTITY_CLUSTER_NAMING_PROMPT = """Given a cluster of entity variants that have been confirmed as the SAME entity, determine the best canonical name.

ENTITY VARIANTS:
{entity_variants_json}

Each variant has:
- name: The name used in a document
- category: Entity type
- description: Context about the entity
- source_doc: Which document it came from

TASK: Choose the BEST canonical name for this cluster.

CRITICAL RULES:
1. **Prefer FULL NAME over abbreviations**
   - ✅ "World Health Organization (WHO)" > "WHO"
   - ✅ "Deoxyribonucleic Acid (DNA)" > "DNA"
   - ✅ "Gross Domestic Product (GDP)" > "GDP"

2. **Preserve acronyms in parentheses** when available
   - ✅ "United Nations (UN)"
   - ❌ "UN" or "United Nations" separately

3. **Use most DESCRIPTIVE version**
   - ✅ "Superconducting Quantum Interference Device (SQUID)" > "SQUID" > "Device"

4. **Include context if names identical**
   - ✅ "Springfield (Illinois)" vs "Springfield (Massachusetts)"
   - ✅ "Mars (planet)" vs "Mars (candy)"

5. **Keep technical precision**
   - ✅ "98.7% accuracy" (exact value)
   - ❌ "high accuracy" (vague)

ALIASES:
- List ALL other names from the variants as aliases
- Do NOT include the canonical name in aliases
- Sort aliases by length (longest first)

MERGED DESCRIPTION:
- Combine UNIQUE information from all variants
- Remove redundant phrases
- Keep it concise (3-5 sentences max)
- Include specific measurements, properties, relationships
- Mention which documents contributed unique info if relevant

Return JSON:
{{
    "canonical_name": "The chosen standard name (with acronym if applicable)",
    "aliases": ["Alternative Name 1", "Alt Name 2", "Abbrev"],
    "merged_description": "Rich combined description from all variants",
    "category": "The most appropriate category from variants"
}}
"""

HIERARCHY_FUSION_PROMPT = """You are tasked with merging logical hierarchies from multiple documents about similar topics into a unified narrative flow.

DOCUMENT HIERARCHIES:
{hierarchies_json}

Each document has its own logical flow with topics and supporting details. Your task is to create a UNIFIED hierarchy that:

1. **Identifies common themes** that appear across documents
2. **Synthesizes complementary information** from both sources
3. **Preserves unique insights** that only appear in one document
4. **Highlights contrasts** where documents present different perspectives

CRITICAL INSTRUCTIONS:
- Create 5-8 main topics that represent the MERGED narrative
- Each topic should have 3-5 supporting details
- If information contradicts between documents, include BOTH perspectives with attribution
- Mark which documents contributed to each topic in the "source_documents" field

OUTPUT FORMAT (JSON):
{{
    "hierarchy": [
        {{
            "main_topic": "Clear, descriptive title (3-6 words)",
            "supporting_details": [
                "Specific fact or claim from the text (include measurements, names, values)",
                "Another detail (note source document if only from one doc)",
                "If documents disagree, state both: 'Source A says X, while Source B says Y'",
                ...
            ],
            "source_documents": ["Document Title 1", "Document Title 2"]
        }},
        ...
    ]
}}

EXAMPLE STRUCTURE (for reference - create your own):
{{
    "hierarchy": [
        {{
            "main_topic": "Initial Problem Recognition",
            "supporting_details": [
                "Previous approaches failed to address fundamental limitation X",
                "Researchers observed phenomenon Y in 87% of cases",
                "Document A emphasizes technical barriers, Document B focuses on cost",
                "Gap in understanding identified in multiple studies"
            ],
            "source_documents": ["Technical Paper", "Review Article"]
        }},
        {{
            "main_topic": "Competing Perspectives",
            "supporting_details": [
                "Team Alpha proposes mechanism based on quantum effects (Source 1)",
                "Team Beta argues for classical explanation with thermal dynamics (Source 2)",
                "Both agree on observed outcomes but differ on underlying cause",
                "Ongoing debate centers on interpretation of data from Experiment C"
            ],
            "source_documents": ["Research Group Alpha", "Research Group Beta"]
        }}
    ]
}}

IMPORTANT:
- DO NOT return an empty hierarchy
- If documents cover completely different topics, create separate sections for each
- Always include concrete details (names, numbers, specific claims)
- Be specific - avoid vague statements like "various concerns exist"

Return ONLY valid JSON with the merged hierarchy.
"""