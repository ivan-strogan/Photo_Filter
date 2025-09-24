#!/usr/bin/env python3
"""
Test the complete photo organization system.

For junior developers:
This script demonstrates the end-to-end photo organization pipeline
including clustering, naming, folder creation, and file organization.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.photo_organizer_pipeline import PhotoOrganizerPipeline

def test_complete_organization():
    print("🚀 Testing Complete Photo Organization Pipeline")
    print("=" * 60)

    # Configure pipeline for testing
    pipeline = PhotoOrganizerPipeline(
        max_photos=20,           # Small number for testing
        operation_mode="copy",   # Safe copy mode
        dry_run=True,           # Safe dry-run mode
        verify_checksums=True   # Ensure file integrity
    )

    print("📋 Pipeline Configuration:")
    print(f"   Max photos: {pipeline.max_photos}")
    print(f"   Operation mode: {pipeline.operation_mode}")
    print(f"   Dry run: {pipeline.dry_run}")
    print(f"   Verify checksums: {pipeline.verify_checksums}")
    print()

    # Create progress callback
    def progress_callback(message, progress):
        percentage = int(progress * 100)
        print(f"[{percentage:3d}%] {message}")

    # Run the complete pipeline
    print("🎬 Starting complete organization pipeline...")
    print()

    results = pipeline.run_complete_pipeline(progress_callback=progress_callback)

    print("\n" + "=" * 60)
    print("📊 PIPELINE RESULTS")
    print("=" * 60)

    # Overall pipeline summary
    pipeline_summary = results['pipeline_summary']
    print(f"⏱️  Execution time: {pipeline_summary['execution_time_seconds']:.2f} seconds")
    print(f"🔧 Dry run mode: {pipeline_summary['dry_run_mode']}")
    print(f"📁 Operation mode: {pipeline_summary['operation_mode']}")
    print(f"✅ Overall success: {pipeline_summary['overall_success']}")
    print()

    # Stage-by-stage breakdown
    stage_summaries = results['stage_summaries']

    print("📈 STAGE-BY-STAGE BREAKDOWN:")
    print("-" * 40)

    # Stage 1: Scan
    scan = stage_summaries['scan']
    print(f"1️⃣  SCAN STAGE:")
    print(f"    Files scanned: {scan['files_scanned']}")
    print(f"    Files selected: {scan['files_selected']}")
    print(f"    Selection rate: {scan['selection_rate']:.1%}")
    print()

    # Stage 2: Clustering
    clustering = stage_summaries['clustering']
    print(f"2️⃣  CLUSTERING STAGE:")
    print(f"    Files input: {clustering['files_input']}")
    print(f"    Clusters created: {clustering['clusters_created']}")
    print(f"    Avg cluster size: {clustering['avg_cluster_size']:.1f} files")
    print()

    # Stage 3: Naming
    naming = stage_summaries['naming']
    print(f"3️⃣  NAMING STAGE:")
    print(f"    Clusters input: {naming['clusters_input']}")
    print(f"    Naming success rate: {naming['naming_success_rate']:.1%}")
    print(f"    High confidence names: {naming['high_confidence_names']}")
    print()

    # Stage 4: Folders
    folders = stage_summaries['folders']
    print(f"4️⃣  FOLDER CREATION STAGE:")
    print(f"    Folders created: {folders['folders_created']}")
    print(f"    Conflicts resolved: {folders['conflicts_resolved']}")
    print()

    # Stage 5: Organization
    organization = stage_summaries['organization']
    print(f"5️⃣  FILE ORGANIZATION STAGE:")
    print(f"    Files processed: {organization['files_processed']}")
    print(f"    Files successful: {organization['files_successful']}")
    print(f"    Success rate: {organization['success_rate']:.1%}")
    print()

    # Show output directory
    if results.get('output_directory'):
        print(f"📁 Output directory: {results['output_directory']}")
        print()

    # Show sample folder structure if available
    detailed_results = results['detailed_results']
    if 'folders' in detailed_results and 'year_organization' in detailed_results['folders']:
        year_org = detailed_results['folders']['year_organization']

        print("📂 SAMPLE FOLDER STRUCTURE:")
        print("-" * 40)

        for year, folders_list in list(year_org.items())[:1]:  # Show one year
            print(f"📅 {year}/ ({len(folders_list)} folders)")

            for folder in folders_list[:5]:  # Show first 5 folders
                folder_name = folder['suggested_name']
                file_count = folder['file_count']
                print(f"   ├── 📂 {folder_name} ({file_count} files)")

            if len(folders_list) > 5:
                print(f"   └── ... and {len(folders_list) - 5} more folders")
        print()

    # Show errors if any
    if results['errors']:
        print("⚠️  ERRORS ENCOUNTERED:")
        print("-" * 40)
        for error in results['errors']:
            print(f"   • {error['stage']}: {error['error']}")
        print()

    # Show warnings if any
    if results['warnings']:
        print("⚠️  WARNINGS:")
        print("-" * 40)
        for warning in results['warnings']:
            print(f"   • {warning}")
        print()

    # Success message
    if pipeline_summary['overall_success']:
        print("🎉 COMPLETE ORGANIZATION PIPELINE TEST SUCCESSFUL!")
        print()
        print("💡 What this means:")
        print("   ✅ Your photos were successfully scanned and analyzed")
        print("   ✅ Intelligent clusters were created based on time and location")
        print("   ✅ Smart folder names were generated")
        print("   ✅ Organized folder structure was created")
        print("   ✅ File organization operations were planned (dry-run mode)")
        print()
        print("🚀 Ready for production use!")
        print("   • Use --no-dry-run to actually organize files")
        print("   • Use --mode move for more efficient organization")
        print("   • Use --save-report to generate detailed reports")
    else:
        print("❌ PIPELINE TEST ENCOUNTERED ISSUES")
        print("   Check the errors above for troubleshooting information")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    test_complete_organization()