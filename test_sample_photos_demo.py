#!/usr/bin/env python3
"""
Demo: Show how Sample_Photos would be organized with local LLM.
Uses actual photo metadata and local LLM for intelligent naming.
"""

import sys
from pathlib import Path
from datetime import datetime
import requests
import json
sys.path.append(str(Path(__file__).parent))

from src.media_detector import MediaDetector
from src.temporal_clustering import TemporalClusterer

def query_ollama_direct(prompt, model="llama3.1:8b"):
    """Direct query to Ollama for event naming."""
    try:
        data = {
            "model": model,
            "prompt": f"You are an AI assistant that creates descriptive, concise folder names for photo events. Follow the requested format exactly.\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 50
            }
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        else:
            return None
    except Exception as e:
        print(f"LLM query failed: {e}")
        return None

def create_naming_prompt(cluster_info):
    """Create a prompt for intelligent event naming."""
    prompt = f"""Create a folder name for this photo event:
- Date: {cluster_info['date_range']}
- Time: {cluster_info['time_of_day']}
- Photos: {cluster_info['photo_count']} files
- Duration: {cluster_info['duration']}

Format: YYYY_MM_DD - Event Name
Response:"""
    return prompt

def demo_photo_organization():
    """Demo how Sample_Photos would be organized."""
    print("📸 SAMPLE PHOTOS ORGANIZATION DEMO")
    print("=" * 60)
    print("Using your local LLM for intelligent event naming")
    print()

    # Scan sample photos
    detector = MediaDetector()
    sample_dir = Path("Sample_Photos/iPhone Automatic")

    if not sample_dir.exists():
        print("❌ Sample_Photos directory not found")
        return

    print(f"🔍 Scanning: {sample_dir}")
    media_files = detector.scan_directory(sample_dir)
    print(f"✅ Found {len(media_files)} photos")
    print()

    # Create temporal clusters
    print("🕒 Creating temporal clusters...")
    clusterer = TemporalClusterer()
    clusters = clusterer.cluster_by_day(media_files)
    print(f"✅ Created {len(clusters)} day-based clusters")
    print()

    # Show first 10 clusters with LLM naming
    print("🧠 Generating intelligent event names with local LLM:")
    print("-" * 60)

    for i, cluster in enumerate(clusters[:10]):
        # Handle different cluster types
        if hasattr(cluster, 'files'):
            files = cluster.files
        elif hasattr(cluster, 'media_files'):
            files = cluster.media_files
        elif isinstance(cluster, list):
            files = cluster
        else:
            print(f"Skipping cluster {i+1}: Unknown type {type(cluster)}")
            continue

        if not files:
            continue

        # Get cluster info
        dates = [f.date for f in files]
        start_date = min(dates)
        end_date = max(dates)

        # Determine time of day from first file
        first_file = files[0]
        if hasattr(first_file.date, 'hour'):
            hour = first_file.date.hour
            if hour < 12:
                time_of_day = "Morning"
            elif hour < 17:
                time_of_day = "Afternoon"
            else:
                time_of_day = "Evening"
        else:
            time_of_day = "Day"  # Default if no time info

        # Calculate duration
        duration_hours = (end_date - start_date).total_seconds() / 3600
        if duration_hours < 1:
            duration = "Short event"
        elif duration_hours < 4:
            duration = "Few hours"
        else:
            duration = "All day"

        cluster_info = {
            'date_range': start_date.strftime("%B %d, %Y"),
            'time_of_day': time_of_day,
            'photo_count': len(files),
            'duration': duration
        }

        # Generate LLM name
        prompt = create_naming_prompt(cluster_info)
        llm_name = query_ollama_direct(prompt)

        # Fallback name
        fallback_name = f"{start_date.strftime('%Y_%m_%d')} - {time_of_day} Photos"

        final_name = llm_name if llm_name else fallback_name

        print(f"{i+1:2d}. {final_name}")
        print(f"    📅 {start_date.strftime('%Y-%m-%d')} ({len(files)} photos)")
        print(f"    🕐 {time_of_day} ({duration})")

        # Show sample filenames
        sample_files = [f.filename for f in files[:3]]
        if len(files) > 3:
            sample_files.append(f"... and {len(files)-3} more")
        print(f"    📁 {', '.join(sample_files)}")
        print()

    if len(clusters) > 10:
        print(f"... and {len(clusters) - 10} more clusters")

    print("=" * 60)
    print("🎉 ORGANIZATION PREVIEW COMPLETE")
    print("=" * 60)
    print(f"✅ Your {len(media_files)} photos would be organized into {len(clusters)} smart folders")
    print("✅ Each folder gets an intelligent, descriptive name")
    print("✅ All processing happens locally with your Ollama LLM")
    print("✅ No data leaves your MacBook Pro!")

if __name__ == "__main__":
    demo_photo_organization()