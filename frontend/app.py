# frontend/app.py
import streamlit as st
import requests
import pandas as pd
import time
import json
from streamlit_echarts import st_pyecharts
from pyecharts import options as opts
from pyecharts.charts import Graph
from typing import List, Dict, Optional
import threading

# --- Configuration ---
st.set_page_config(page_title="CogniSynth Multi-Doc", layout="wide")
API_URL = "http://127.0.0.1:8000"

# --- Session State ---
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'selected_doc_ids' not in st.session_state:
    st.session_state.selected_doc_ids = []
if 'current_graph' not in st.session_state:
    st.session_state.current_graph = None
if 'selected_node' not in st.session_state:
    st.session_state.selected_node = None
if 'processing_docs' not in st.session_state:
    st.session_state.processing_docs = {}
if 'viewing_doc_id' not in st.session_state:
    st.session_state.viewing_doc_id = None

# --- Helper Functions ---

def upload_document(title: str, text: str) -> Optional[Dict]:
    """Upload a document to the backend."""
    try:
        response = requests.post(
            f"{API_URL}/documents/",
            json={"title": title, "text": text}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Upload failed: {e}")
        return None

def fetch_all_documents() -> List[Dict]:
    """Fetch all documents from backend."""
    try:
        response = requests.get(f"{API_URL}/documents/")
        response.raise_for_status()
        return response.json()
    except:
        return []

def delete_document(doc_id: str) -> bool:
    """Delete a document."""
    try:
        response = requests.delete(f"{API_URL}/documents/{doc_id}")
        response.raise_for_status()
        return True
    except:
        return False

def get_document_content(doc_id: str) -> Optional[Dict]:
    """Get document content by ID."""
    try:
        response = requests.get(f"{API_URL}/documents/{doc_id}")
        response.raise_for_status()
        return response.json()
    except:
        return None

def get_merged_graph(doc_ids: List[str]) -> Optional[Dict]:
    """Get merged knowledge graph for selected documents."""
    try:
        response = requests.post(
            f"{API_URL}/graphs/merge",
            json={"document_ids": doc_ids}
        )
        response.raise_for_status()
        return response.json().get('analysis_data')
    except requests.exceptions.RequestException as e:
        st.error(f"Graph merge failed: {e}")
        return None

def get_status_icon(status: str) -> str:
    """Return emoji based on processing status."""
    icons = {
        "completed": "✅",
        "processing": "⏳",
        "pending": "🕐",
        "failed": "❌"
    }
    return icons.get(status, "❓")

def listen_to_progress_sse(document_id: str, progress_container):
    """
    Listen to Server-Sent Events for real-time progress updates.
    Updates the UI with progress bars and messages.
    """
    try:
        import sseclient
        
        response = requests.get(
            f"{API_URL}/documents/{document_id}/progress",
            stream=True,
            headers={'Accept': 'text/event-stream'},
            timeout=120
        )
        
        client = sseclient.SSEClient(response)
        
        for event in client.events():
            if event.event == 'progress':
                data = json.loads(event.data)
                
                # Update progress in session state
                st.session_state.processing_docs[document_id] = data
                
                # Update the UI
                with progress_container:
                    progress_value = data.get('progress', 0) / 100.0
                    st.progress(progress_value, text=f"{data.get('message', 'Processing...')} ({data.get('progress', 0)}%)")
                
                # Check if done
                if data.get('status') in ['completed', 'failed']:
                    break
            
            elif event.event == 'done':
                break
                
    except Exception as e:
        print(f"SSE Progress stream error: {e}")
    finally:
        # Clean up
        if document_id in st.session_state.processing_docs:
            del st.session_state.processing_docs[document_id]

def poll_document_status(document_id: str, max_attempts: int = 60) -> str:
    """
    Fallback polling method if SSE doesn't work.
    Poll document status until completion or timeout.
    """
    for _ in range(max_attempts):
        try:
            response = requests.get(f"{API_URL}/documents/{document_id}/status")
            if response.status_code == 200:
                status = response.json()['status']
                if status in ['completed', 'failed']:
                    return status
        except:
            pass
        time.sleep(2)
    return 'timeout'

# --- UI Layout ---
st.title("🧠 CogniSynth: Multi-Document Knowledge Graph")
st.markdown("Build unified knowledge graphs from multiple text sources with **real-time progress tracking**")

# === SIDEBAR: Document Management ===
with st.sidebar:
    st.header("📚 Document Sources")
    
    # Add new document section
    with st.expander("➕ Add New Source", expanded=False):
        new_title = st.text_input("Document Title:", placeholder="e.g., Dark Matter Research")
        new_text = st.text_area("Paste text here:", height=200, placeholder="Minimum 100 characters...")
        
        if st.button("📤 Upload Document", use_container_width=True, type="primary"):
            if len(new_title.strip()) < 1:
                st.warning("Please provide a title")
            elif len(new_text.strip()) < 100:
                st.warning("Text must be at least 100 characters")
            else:
                with st.spinner("Uploading document..."):
                    result = upload_document(new_title, new_text)
                    
                    if result:
                        doc_id = result['id']
                        st.success(f"✅ '{new_title}' uploaded!")
                        
                        # Show progress tracking
                        st.info("📊 Starting real-time analysis...")
                        progress_placeholder = st.empty()
                        
                        # Try SSE for real-time updates
                        try:
                            import sseclient
                            
                            # Start SSE listener in background
                            thread = threading.Thread(
                                target=listen_to_progress_sse,
                                args=(doc_id, progress_placeholder)
                            )
                            thread.daemon = True
                            thread.start()
                            
                            # Wait for completion (with timeout)
                            thread.join(timeout=120)
                            
                            if thread.is_alive():
                                st.warning("⚠️ Analysis taking longer than expected. It will continue in the background.")
                            else:
                                st.success("✅ Analysis complete!")
                            
                        except ImportError:
                            # Fallback to polling if sseclient not installed
                            st.info("Real-time progress not available. Polling status...")
                            with st.spinner("Analyzing document..."):
                                status = poll_document_status(doc_id)
                                if status == 'completed':
                                    st.success("✅ Analysis complete!")
                                elif status == 'failed':
                                    st.error("❌ Analysis failed. Check logs.")
                                else:
                                    st.warning("⚠️ Analysis is taking longer than expected.")
                        
                        time.sleep(1)
                        st.rerun()
    
    st.markdown("---")
    
    # Refresh documents list
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Refresh", use_container_width=True):
            st.session_state.documents = fetch_all_documents()
            st.rerun()
    
    with col2:
        # Show active processing count
        active_count = sum(1 for d in st.session_state.documents if d['status'] == 'processing')
        if active_count > 0:
            st.metric("⏳ Processing", active_count)
    
    # Auto-load on first run
    if not st.session_state.documents:
        st.session_state.documents = fetch_all_documents()
    
    st.markdown("### Active Sources")
    
    if not st.session_state.documents:
        st.info("No documents yet. Add your first source above!")
    else:
        # Show processing documents with live progress
        processing_docs = [d for d in st.session_state.documents if d['status'] == 'processing']
        
        if processing_docs:
            st.markdown("**🔄 Currently Processing:**")
            for doc in processing_docs:
                with st.container():
                    st.caption(f"⏳ {doc['title'][:40]}...")
                    # Show progress if available
                    if doc['id'] in st.session_state.processing_docs:
                        progress_data = st.session_state.processing_docs[doc['id']]
                        st.progress(
                            progress_data.get('progress', 0) / 100,
                            text=f"{progress_data.get('current_step', 0)}/{progress_data.get('total_steps', 5)}: {progress_data.get('message', '')}"
                        )
            st.markdown("---")
        
        # Track selected documents
        selected_ids = []
        
        st.markdown("**📄 All Documents:**")
        
        for doc in st.session_state.documents:
            col1, col2, col3 = st.columns([0.15, 0.65, 0.2])
            
            with col1:
                is_selected = st.checkbox(
                    "",
                    value=doc['id'] in st.session_state.selected_doc_ids,
                    key=f"select_{doc['id']}",
                    disabled=(doc['status'] != 'completed')
                )
                if is_selected and doc['status'] == 'completed':
                    selected_ids.append(doc['id'])
            
            with col2:
                status_icon = get_status_icon(doc['status'])
                # Make document title clickable
                if st.button(
                    f"{status_icon} **{doc['title'][:30]}**",
                    key=f"view_{doc['id']}",
                    help="Click to view document content",
                    use_container_width=True
                ):
                    st.session_state.viewing_doc_id = doc['id']
                    st.rerun()
                
                st.caption(f"{doc['word_count']} words • {doc['status']}")
            
            with col3:
                if st.button("🗑️", key=f"del_{doc['id']}", help="Delete document"):
                    if delete_document(doc['id']):
                        st.success("Deleted!")
                        time.sleep(0.5)
                        st.rerun()
        
        # Update selected documents in session state
        st.session_state.selected_doc_ids = selected_ids
        
        st.markdown("---")
        
        # Generate graph button
        st.markdown(f"**Selected:** {len(selected_ids)} document(s)")
        
        if st.button("🔗 Generate Knowledge Graph", 
                    use_container_width=True, 
                    type="primary",
                    disabled=(len(selected_ids) == 0)):
            
            if len(selected_ids) == 0:
                st.warning("Select at least one completed document")
            else:
                cache_status = "💾 Loading from cache..." if len(selected_ids) > 1 else "📊 Loading graph..."
                with st.spinner(cache_status):
                    graph_data = get_merged_graph(selected_ids)
                    if graph_data:
                        st.session_state.current_graph = graph_data
                        st.success("✅ Graph generated!")
                        time.sleep(0.5)
                        st.rerun()

# === MAIN AREA: Document Viewer or Graph Visualization ===

# Check if we're viewing a document
if st.session_state.viewing_doc_id:
    # Get document content
    doc_content = get_document_content(st.session_state.viewing_doc_id)
    
    if doc_content:
        st.header(f"📄 {doc_content['title']}")
        
        # Add a back button
        if st.button("← Back to Graph View", key="back_to_graph"):
            st.session_state.viewing_doc_id = None
            st.rerun()
        
        st.markdown("---")
        
        # Display document content
        st.subheader("Document Content")
        st.text_area("", value=doc_content['text'], height=500, disabled=True, label_visibility="collapsed")
        
        # Document metadata
        st.markdown("---")
        st.subheader("Document Information")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Word Count", doc_content['word_count'])
        
        with col2:
            st.metric("Status", doc_content['status'])
        
        with col3:
            created_at = doc_content.get('created_at', 'Unknown')
            if created_at != 'Unknown':
                try:
                    # Format the date if it's in ISO format
                    from datetime import datetime
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime('%Y-%m-%d %H:%M')
                    st.metric("Created", created_date)
                except:
                    st.metric("Created", created_at)
            else:
                st.metric("Created", "Unknown")
        
        # Show document's own graph if available
        if doc_content['status'] == 'completed':
            st.markdown("---")
            st.subheader("Document Knowledge Graph")
            
            # Get graph for this single document
            single_doc_graph = get_merged_graph([st.session_state.viewing_doc_id])
            
            if single_doc_graph:
                entities = single_doc_graph.get('entities', [])
                relationships = single_doc_graph.get('relationships', [])
                
                if entities:
                    # Create entity lookup
                    entity_dict = {entity['name']: entity for entity in entities}
                    
                    # Build graph nodes
                    nodes = [
                        opts.GraphNode(
                            name=entity['name'],
                            symbol_size=30,
                            tooltip_opts=opts.TooltipOpts(
                                formatter=f"{entity.get('category', 'N/A')}: {entity.get('description', '')[:100]}..."
                            )
                        ) for entity in entities
                    ]
                    
                    # Build graph edges
                    links = []
                    for rel in relationships:
                        if rel['source'] in entity_dict and rel['target'] in entity_dict:
                            label = rel.get('natural_language_label', 'related to')
                            links.append(
                                opts.GraphLink(
                                    source=rel['source'],
                                    target=rel['target'],
                                    value=label,
                                    label_opts=opts.LabelOpts(
                                        is_show=True, 
                                        formatter=label
                                    )
                                )
                            )
                    
                    # Render graph
                    c = (
                        Graph(init_opts=opts.InitOpts(width="100%", height="500px"))
                        .add(
                            "", 
                            nodes, 
                            links, 
                            repulsion=5000, 
                            layout="force",
                            edge_symbol=["", "arrow"],
                            edge_label=opts.LabelOpts(is_show=True, position="middle", font_size=10)
                        )
                        .set_global_opts(
                            title_opts=opts.TitleOpts(title=""),
                            tooltip_opts=opts.TooltipOpts(
                                trigger="item",
                                formatter="{b}: {c}"
                            )
                        )
                    )
                    st_pyecharts(c, height="500px")
                else:
                    st.info("No entities were extracted from this document.")
            else:
                st.info("No graph available for this document.")
    else:
        st.error("Failed to load document content.")
        if st.button("← Back to Graph View", key="back_to_graph_error"):
            st.session_state.viewing_doc_id = None
            st.rerun()

# Show the graph view if not viewing a document
elif st.session_state.current_graph:
    data = st.session_state.current_graph
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Visual Graph", "🔗 Entity Table", "📜 Logical Flow", "📑 Raw JSON"])
    
    with tab1:
        st.subheader("Visual Knowledge Graph")
        
        # Create two columns: graph on left, info panel on right
        col1, col2 = st.columns([3, 1])
        
        with col1:
            entities = data.get('entities', [])
            relationships = data.get('relationships', [])
            
            if not entities:
                st.warning("No entities were extracted to build a graph.")
            else:
                # Create entity lookup
                entity_dict = {entity['name']: entity for entity in entities}
                
                # Build graph nodes
                nodes = [
                    opts.GraphNode(
                        name=entity['name'],
                        symbol_size=30,
                        tooltip_opts=opts.TooltipOpts(
                            formatter=f"{entity.get('category', 'N/A')}: {entity.get('description', '')[:100]}..."
                        )
                    ) for entity in entities
                ]
                
                # Build graph edges
                links = []
                edge_tooltip_data = {}
                
                for rel in relationships:
                    if rel['source'] in entity_dict and rel['target'] in entity_dict:
                        label = rel.get('natural_language_label', 'related to')
                        explanation = rel.get('explanation', '')
                        
                        edge_key = f"{rel['source']}-{rel['target']}"
                        edge_tooltip_data[edge_key] = {
                            'label': label,
                            'explanation': explanation
                        }
                        
                        links.append(
                            opts.GraphLink(
                                source=rel['source'],
                                target=rel['target'],
                                value=label,
                                label_opts=opts.LabelOpts(
                                    is_show=True, 
                                    formatter=label
                                )
                            )
                        )
                
                # Render graph with edge-aware tooltip
                c = (
                    Graph(init_opts=opts.InitOpts(width="100%", height="700px"))
                    .add(
                        "", 
                        nodes, 
                        links, 
                        repulsion=5000, 
                        layout="force",
                        edge_symbol=["", "arrow"],
                        edge_label=opts.LabelOpts(is_show=True, position="middle", font_size=10)
                    )
                    .set_global_opts(
                        title_opts=opts.TitleOpts(title=""),
                        tooltip_opts=opts.TooltipOpts(
                            trigger="item",
                            formatter="{b}: {c}"
                        )
                    )
                )
                st_pyecharts(c, height="700px")
                
                # Node selector
                st.markdown("---")
                st.markdown("**💡 Click to view detailed information:**")
                selected_node_name = st.selectbox(
                    "Select a node to view details:",
                    options=[""] + [entity['name'] for entity in entities],
                    key="node_selector"
                )
                
                if selected_node_name:
                    st.session_state.selected_node = selected_node_name
        
        with col2:
            st.markdown("### 📋 Details Panel")
            st.markdown("*Select a node from the dropdown*")
            
            if st.session_state.selected_node and st.session_state.selected_node in entity_dict:
                selected_entity = entity_dict[st.session_state.selected_node]
                
                st.markdown(f"#### 🔷 {selected_entity['name']}")
                st.markdown(f"**Category:** `{selected_entity.get('category', 'N/A')}`")
                st.markdown("**Description:**")
                st.markdown(f"> {selected_entity.get('description', 'No description available.')}")
                
                # Show relationships
                st.markdown("---")
                st.markdown("**🔗 Relationships:**")
                
                # Incoming relationships
                incoming = [rel for rel in relationships if rel['target'] == st.session_state.selected_node]
                if incoming:
                    st.markdown("*Incoming:*")
                    for rel in incoming:
                        st.markdown(f"- **{rel['source']}** → *{rel.get('natural_language_label', 'related to')}* → **{rel['target']}**")
                        if rel.get('explanation'):
                            st.caption(rel['explanation'])
                
                # Outgoing relationships
                outgoing = [rel for rel in relationships if rel['source'] == st.session_state.selected_node]
                if outgoing:
                    st.markdown("*Outgoing:*")
                    for rel in outgoing:
                        st.markdown(f"- **{rel['source']}** → *{rel.get('natural_language_label', 'related to')}* → **{rel['target']}**")
                        if rel.get('explanation'):
                            st.caption(rel['explanation'])
                
                if not incoming and not outgoing:
                    st.info("This entity has no direct relationships.")
            else:
                st.info("👈 Select a node from the dropdown to see details")
    
    with tab2:
        st.subheader("Extracted Entities & Concepts")
        if data.get('entities'):
            df = pd.DataFrame(data['entities'])
            
            # Add source count if multiple documents
            if len(st.session_state.selected_doc_ids) > 1:
                st.info(f"📚 Showing entities from {len(st.session_state.selected_doc_ids)} documents")
            
            # Display with search
            search = st.text_input("🔍 Search entities:", placeholder="Type to filter...")
            if search:
                df = df[df['name'].str.contains(search, case=False, na=False) | 
                       df['description'].str.contains(search, case=False, na=False)]
            
            st.dataframe(df, use_container_width=True, height=600)
        else:
            st.warning("No entities were extracted.")
    
    with tab3:
        st.subheader("Hierarchical Logical Flow")
        
        if len(st.session_state.selected_doc_ids) > 1:
            st.info(f"📚 Combined hierarchy from {len(st.session_state.selected_doc_ids)} documents")
        
        if data.get('hierarchy'):
            for i, topic in enumerate(data['hierarchy']):
                with st.expander(f"**{i+1}. {topic.get('main_topic', 'Unnamed Topic')}**", expanded=True):
                    details = topic.get('supporting_details', [])
                    if details:
                        for detail in details:
                            st.markdown(f"- {detail}")
                    else:
                        st.markdown("*No details available*")
        else:
            st.warning("No hierarchical structure was extracted.")
    
    with tab4:
        st.subheader("Raw AI Output")
        
        # Show cache status
        if len(st.session_state.selected_doc_ids) > 1:
            st.info("💾 This merged graph is cached for instant future access")
        
        st.json(data)

else:
    # Welcome screen
    st.info("👈 **Get Started:** Add documents in the sidebar, then click 'Generate Knowledge Graph'")
    
    st.markdown("### 🎯 Features")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📚 Multi-Source")
        st.markdown("Combine insights from multiple documents into one unified graph")
    
    with col2:
        st.markdown("#### ⚡ Real-Time Progress")
        st.markdown("Watch live updates as documents are analyzed (5 steps)")
    
    with col3:
        st.markdown("#### 💾 Smart Caching")
        st.markdown("Instant loading for previously generated graph combinations")
    
    st.markdown("---")
    st.markdown("### 💡 Quick Start Guide")
    st.markdown("""
    1. **Add a document** using the sidebar form
    2. **Watch real-time progress** (Extracting entities → Building hierarchy → Inferring relationships...)
    3. Wait for analysis to complete (you'll see ✅)
    4. **Select one or more** completed documents
    5. Click **Generate Knowledge Graph**
    6. Explore the interactive visualization!
    """)

# Footer
st.markdown("---")
caption_text = f"CogniSynth Multi-Document v2.0 | {len(st.session_state.documents)} documents | {len(st.session_state.selected_doc_ids)} selected"

# Add viewing document info to footer if applicable
if st.session_state.viewing_doc_id:
    doc = next((d for d in st.session_state.documents if d['id'] == st.session_state.viewing_doc_id), None)
    if doc:
        caption_text += f" | Viewing: {doc['title']}"

st.caption(caption_text)