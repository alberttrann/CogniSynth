# debug_merge.py - Run this to test your merge

import asyncio
import sys
sys.path.append('.')

from backend.database import SessionLocal, DocumentCRUD, AnalysisCRUD
from backend.main import merge_analyses

async def test_merge():
    """Test the merge function with your two documents."""
    db = SessionLocal()
    
    # Your document IDs
    doc_ids = [
        "4741b74f-0702-44ba-9af4-adde6b6aaea4",  # Technical doc
        "dfbc4a37-cd72-4d4d-859a-3fe1f7b72e7f"   # Ethics doc
    ]
    
    print("="*80)
    print("DEBUGGING MULTI-DOCUMENT MERGE")
    print("="*80)
    
    # Get analyses
    analyses = AnalysisCRUD.get_multiple(db, doc_ids)
    
    if len(analyses) != 2:
        print(f"❌ ERROR: Expected 2 analyses, got {len(analyses)}")
        return
    
    print(f"\n📊 Input Statistics:")
    for analysis in analyses:
        doc = DocumentCRUD.get_by_id(db, analysis.document_id)
        print(f"  Document: {doc.title}")
        print(f"    - Entities: {len(analysis.entities)}")
        print(f"    - Relationships: {len(analysis.relationships)}")
        print(f"    - Hierarchy Topics: {len(analysis.hierarchy)}")
    
    # Check for the specific duplicate problem
    print(f"\n🔍 Checking for 'Synapse-1' variants:")
    for analysis in analyses:
        synapse_variants = [
            e['name'] for e in analysis.entities 
            if 'synapse' in e['name'].lower()
        ]
        if synapse_variants:
            print(f"  Found in doc {analysis.document_id[:8]}:")
            for variant in synapse_variants:
                print(f"    - {variant}")
    
    # Run merge
    print(f"\n{'='*80}")
    print("STARTING MERGE PROCESS")
    print("="*80)
    
    try:
        result = await merge_analyses(analyses, db)
        
        print(f"\n{'='*80}")
        print("MERGE RESULTS")
        print("="*80)
        
        print(f"\n📊 Output Statistics:")
        print(f"  - Entities: {len(result['entities'])} (was {sum(len(a.entities) for a in analyses)})")
        print(f"  - Relationships: {len(result['relationships'])} (was {sum(len(a.relationships) for a in analyses)})")
        print(f"  - Hierarchy Topics: {len(result['hierarchy'])} (was {sum(len(a.hierarchy) for a in analyses)})")
        
        # Check if Synapse-1 variants were merged
        print(f"\n🔍 Checking if 'Synapse-1' variants merged:")
        synapse_entities = [
            e['name'] for e in result['entities'] 
            if 'synapse' in e['name'].lower()
        ]
        print(f"  Found {len(synapse_entities)} Synapse entities in merged result:")
        for name in synapse_entities:
            entity = next(e for e in result['entities'] if e['name'] == name)
            sources = entity.get('source_documents', [])
            aliases = entity.get('aliases', [])
            print(f"    - {name}")
            print(f"      Sources: {len(sources)} document(s)")
            if aliases:
                print(f"      Aliases: {aliases}")
        
        # Check hierarchy deduplication
        print(f"\n🔍 Checking hierarchy topics:")
        topic_names = [t.get('main_topic', 'Unnamed') for t in result['hierarchy']]
        duplicates = [name for name in set(topic_names) if topic_names.count(name) > 1]
        
        if duplicates:
            print(f"  ⚠️ DUPLICATE TOPICS FOUND:")
            for dup in duplicates:
                count = topic_names.count(dup)
                print(f"    - '{dup}' appears {count} times")
        else:
            print(f"  ✅ No duplicate topics found")
        
        # Show all topics
        print(f"\n  All topics ({len(result['hierarchy'])}):")
        for i, topic in enumerate(result['hierarchy'], 1):
            sources = topic.get('source_documents', topic.get('source_document', 'Unknown'))
            print(f"    {i}. {topic.get('main_topic', 'Unnamed')}")
            print(f"       Source: {sources}")
        
        print(f"\n{'='*80}")
        print("✅ MERGE TEST COMPLETE")
        print("="*80)
        
        return result
        
    except Exception as e:
        print(f"\n❌ MERGE FAILED WITH ERROR:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    finally:
        db.close()

if __name__ == "__main__":
    result = asyncio.run(test_merge())
    
    if result:
        print(f"\n💾 To see full output, check the returned 'result' object")
        print(f"   - result['entities'] = {len(result['entities'])} entities")
        print(f"   - result['relationships'] = {len(result['relationships'])} relationships")
        print(f"   - result['hierarchy'] = {len(result['hierarchy'])} topics")