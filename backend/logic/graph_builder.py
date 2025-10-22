# backend/logic/graph_builder.py
import os
import json
import networkx as nx
from openai import AsyncOpenAI
from typing import Dict, List, Any
from dotenv import load_dotenv
from pathlib import Path
import asyncio

from . import prompts
from ..models import Node, Edge, Font

# --- ROBUST INITIALIZATION BLOCK (from previous fix, still valid) ---
dotenv_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=dotenv_path)

api_key = os.getenv("FPT_API_KEY")
base_url = os.getenv("FPT_API_BASE")

if not api_key:
    raise ValueError("CRITICAL ERROR: FPT_API_KEY is not set.")

client = AsyncOpenAI(api_key=api_key, base_url=base_url)


def get_node_color(category: str) -> str:
    """Assigns a meaningful color based on node category."""
    category = category.upper()
    if category in ["PERSON", "ORGANIZATION", "LOCATION"]:
        return "#FFC300"  # Vivid Yellow (Entities)
    elif category in ["PROBLEM", "CHALLENGE"]:
        return "#C70039"  # Crimson Red (Problems/Conflicts)
    elif category in ["SOLUTION", "METHOD", "TECHNOLOGY", "FINDING"]:
        return "#2ECC71"  # Emerald Green (Solutions/Methods/Findings)
    elif category in ["THEORY", "CONCEPT", "TIME_PERIOD", "MEASUREMENT"]:
        return "#3498DB"  # Bright Blue (Core Concepts/Abstracts)
    else:
        return "#99A3A4"  # Grey (Generic/Other)

def get_edge_color(category: str) -> str:
    """Assigns a color based on edge category."""
    category = category.upper()
    if category == "COMPOSITIONAL":
        return "#3498DB" # Blue
    elif category == "CAUSAL":
        return "#E74C3C" # Red
    elif category == "EVIDENTIARY":
        return "#2ECC71" # Green
    elif category == "STRUCTURAL": # For explicit hierarchy if we re-introduce later, or strong association
        return "#FFC300" # Yellow
    else: # ASSOCIATIVE
        return "#95A5A6" # Grey


async def _make_llm_call(prompt: str, model_name: str) -> Dict:
    """A robust, asynchronous wrapper for making LLM calls."""
    try:
        response = await client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        print(f"Error during LLM call: {e}")
        return {}


async def generate_graph_from_text(text: str, model_name: str) -> Dict[str, Any]:
    """
    Builds a flat graph with rich node tooltips and hybrid natural language edges.
    """
    # --- STAGE 0: Node Extraction (Now flat, more robust) ---
    node_prompt = prompts.NODE_EXTRACTION_PROMPT.format(text=text)
    node_extraction_result = await _make_llm_call(node_prompt, model_name)
    
    extracted_nodes_data = node_extraction_result.get("nodes", []) # Expect a list under "nodes" key
    
    if not extracted_nodes_data:
        raise ValueError("Stage 0 (Node Extraction) failed. The LLM returned no nodes.")

    G = nx.DiGraph()
    all_node_names = []

    # Add all extracted concepts as nodes
    for node_item in extracted_nodes_data:
        node_name = node_item.get("name")
        if node_name and node_name not in G:
            G.add_node(node_name, 
                       category=node_item.get("category", "Unknown"), 
                       description=node_item.get("description", ""))
            all_node_names.append(node_name)

    if len(all_node_names) < 2:
        return format_graph_for_frontend(G)

    # --- STAGE 1 & 2: Find and Classify Logical Links ---
    finder_prompt = prompts.RELATIONSHIP_FINDER_PROMPT.format(text=text, nodes_list=json.dumps(all_node_names))
    potential_links_data = await _make_llm_call(finder_prompt, model_name)
    potential_links = potential_links_data.get("potential_links", [])

    if not potential_links:
        return format_graph_for_frontend(G)

    classification_tasks = []
    for link in potential_links:
        source, target, sentence = link.get("source"), link.get("target"), link.get("connecting_sentence")
        if source in G and target in G and sentence:
            classifier_prompt = prompts.RELATIONSHIP_CLASSIFIER_PROMPT.format(connecting_sentence=sentence, source=source, target=target)
            task = _make_llm_call(classifier_prompt, model_name)
            classification_tasks.append((source, target, task))
            
    results = await asyncio.gather(*(task for _, _, task in classification_tasks))

    for i, classification_result in enumerate(results):
        source, target, _ = classification_tasks[i]
        if classification_result:
            # HYBRID EDGES: Use the new natural language label and category
            label = classification_result.get("natural_language_label", "is related to")
            category = classification_result.get("category", "ASSOCIATIVE")
            explanation = classification_result.get("explanation", "")
            G.add_edge(source, target, label=label, category=category, explanation=explanation)
            
    return format_graph_for_frontend(G)

def format_graph_for_frontend(G: nx.DiGraph) -> Dict[str, Any]:
    """Formats a flat graph with rich node tooltips and hybrid edges."""
    nodes_for_frontend = []
    for node_id, data in G.nodes(data=True):
        nodes_for_frontend.append(
            Node(
                id=node_id,
                label=node_id,
                size=30, # Consistent size for all main concepts
                color=get_node_color(data.get("category", "")),
                # RICH TOOLTIP: Includes description directly
                title=f"<b>{node_id}</b><br><em>Category: {data.get('category', 'N/A')}</em><br>{data.get('description', 'No description.')}",
                font=Font(color="#FFFFFF", size=18) # Slightly reduced for readability on a busy graph
            )
        )

    edges_for_frontend = []
    for source, target, data in G.edges(data=True):
        category = data.get("category", "ASSOCIATIVE")
        
        edges_for_frontend.append(
            Edge(
                source=source,
                target=target,
                label=data.get("label", ""), # Natural language label
                color=get_edge_color(category),
                font=Font(color="#FFFFFF", size=12) # Slightly reduced for clarity
            )
        )

    graph_data = nx.node_link_data(G)
    return {"nodes": nodes_for_frontend, "edges": edges_for_frontend, "graph_data": graph_data}

def analyze_node_connections(node_id: str, graph_data: Dict[str, Any]) -> str:
    """
    Performs the interactive breakdown and returns HTML.
    Now leverages the richer node description for primary analysis.
    """
    if not graph_data:
        return "<h3>Error: Graph data not found.</h3>"

    G = nx.node_link_graph(graph_data)

    if node_id not in G:
        return f"<h3>Error: Node '{node_id}' not found in graph.</h3>"

    node_data = G.nodes[node_id]
    html = f"<h3>Analysis for: {node_id}</h3>"
    html += f"<em><b>Category:</b> {node_data.get('category', 'N/A')}</em><br>"
    # Use the rich description from the node itself
    html += f"<p>{node_data.get('description', 'No detailed description available.')}</p><hr>"

    in_edges = G.in_edges(node_id, data=True)
    if in_edges:
        html += "<h4>Influenced By / Caused By:</h4><ul>"
        for source, _, data in in_edges:
            html += f"<li><b>{source}</b> (<span style='color:{get_edge_color(data.get('category', 'ASSOCIATIVE'))}'>{data.get('label', 'is related to')}</span>)<br><small><i>Explanation: {data.get('explanation', '...')}</i></small></li>"
        html += "</ul>"

    out_edges = G.out_edges(node_id, data=True)
    if out_edges:
        html += "<h4>Influences / Leads To:</h4><ul>"
        for _, target, data in out_edges:
            html += f"<li><b>{target}</b> (<span style='color:{get_edge_color(data.get('category', 'ASSOCIATIVE'))}'>{data.get('label', 'is related to')}</span>)<br><small><i>Explanation: {data.get('explanation', '...')}</i></small></li>"
        html += "</ul>"
    
    if not in_edges and not out_edges:
        html += "<p>This is a foundational concept with no direct logical links to other extracted nodes in this text.</p>"

    return html