#!/usr/bin/env python3
"""Test content analyzer for photo analysis capabilities."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.content_analyzer import ContentAnalyzer
from src.media_detector import MediaDetector

def test_content_analyzer():
    """Test the content analyzer functionality."""
    print("🔍 Testing Content Analyzer...")

    try:
        # Initialize components
        analyzer = ContentAnalyzer(use_gpu=False, enable_local_models=False)
        detector = MediaDetector()

        # Get sample photos for testing
        all_files = detector.scan_iphone_automatic()
        photo_files = [f for f in all_files if f.file_type == 'photo']

        if not photo_files:
            print("❌ No photos found for testing")
            return

        print(f"Found {len(photo_files)} photos for analysis")

        # Test basic analysis (without ML models)
        print(f"\n📸 Testing basic content analysis...")

        sample_photos = photo_files[:3]  # Test first 3 photos

        for i, photo in enumerate(sample_photos):
            print(f"\n--- Photo {i+1}: {photo.filename} ---")
            print(f"Path: {photo.path}")
            print(f"Time: {photo.time}")

            # Analyze content
            analysis = analyzer.analyze_photo_content(photo.path)

            if analysis:
                print(f"✅ Analysis successful:")
                print(f"  Model: {analysis.analysis_model}")
                print(f"  Confidence: {analysis.confidence_score:.2f}")
                print(f"  Description: {analysis.description}")
                print(f"  Objects: {', '.join(analysis.objects) if analysis.objects else 'None detected'}")
                print(f"  Scenes: {', '.join(analysis.scenes) if analysis.scenes else 'None detected'}")
                print(f"  Activities: {', '.join(analysis.activities) if analysis.activities else 'None detected'}")
            else:
                print(f"❌ Analysis failed")

        # Test batch analysis
        print(f"\n🔄 Testing batch analysis...")

        batch_photos = [photo.path for photo in sample_photos]
        batch_results = analyzer.analyze_batch(batch_photos, max_photos=3)

        print(f"Batch analysis completed: {len(batch_results)} photos processed")

        # Generate content summary
        if batch_results:
            summary = analyzer.get_content_summary(batch_results)
            print(f"\n📊 Content Summary:")
            print(f"  Photos analyzed: {summary['total_photos_analyzed']}")
            print(f"  Average confidence: {summary['average_confidence']:.3f}")
            print(f"  Unique objects: {summary['unique_objects']}")
            print(f"  Unique scenes: {summary['unique_scenes']}")
            print(f"  Unique activities: {summary['unique_activities']}")

            if summary['top_objects']:
                print(f"  Top objects: {', '.join([obj for obj, count in summary['top_objects'][:3]])}")
            if summary['top_scenes']:
                print(f"  Top scenes: {', '.join([scene for scene, count in summary['top_scenes'][:3]])}")
            if summary['top_activities']:
                print(f"  Top activities: {', '.join([activity for activity, count in summary['top_activities'][:3]])}")

        # Test cache functionality
        print(f"\n💾 Testing cache functionality...")

        cache_file = Path("content_analysis_cache.json")

        # Save cache
        analyzer.save_analysis_cache(cache_file)

        # Clear cache and reload
        analyzer.analysis_cache.clear()
        print(f"  Cache cleared: {len(analyzer.analysis_cache)} entries")

        analyzer.load_analysis_cache(cache_file)
        print(f"  Cache reloaded: {len(analyzer.analysis_cache)} entries")

        # Clean up test file
        if cache_file.exists():
            cache_file.unlink()
            print(f"  Test cache file cleaned up")

        # Test with ML models if available
        print(f"\n🤖 Testing ML model availability...")

        try:
            import torch
            from transformers import CLIPModel
            print(f"  ✅ PyTorch available: {torch.__version__}")
            print(f"  ✅ Transformers available")

            # Test GPU availability
            if torch.cuda.is_available():
                print(f"  ✅ CUDA GPU available: {torch.cuda.get_device_name()}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                print(f"  ✅ Apple MPS available")
            else:
                print(f"  ℹ️  CPU only (no GPU acceleration)")

            # Note about model initialization
            print(f"\n💡 Note: ML models not initialized in this test to avoid downloads")
            print(f"   To test full functionality, set enable_local_models=True")

        except ImportError as e:
            print(f"  ❌ ML dependencies not available: {e}")
            print(f"  💡 Install with: pip install torch transformers")

        print(f"\n✅ Content analyzer testing completed!")

        # Cleanup
        analyzer.cleanup()

    except Exception as e:
        print(f"❌ Error in content analyzer test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_content_analyzer()