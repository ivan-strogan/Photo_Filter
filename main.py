#!/usr/bin/env python3
"""
Photo Filter AI App - Main Entry Point

This is the command-line interface (CLI) for the Photo Filter AI application.
It provides several commands to help organize photos and videos automatically
using artificial intelligence and machine learning techniques.

Key Features:
- Scans existing photo libraries to understand organization patterns
- Processes new photos from iPhone Automatic folder
- Uses temporal clustering (grouping by time)
- Uses location-based clustering (grouping by GPS coordinates)
- Performs content analysis using computer vision
- Provides intelligent event naming suggestions

For junior developers:
- This file uses the Click library for creating CLI commands
- Each @cli.command() decorator creates a new command (scan, process, etc.)
- The PhotoOrganizerPipeline handles the complete processing workflow
- Error handling is implemented for each command
"""

# Standard library imports
import click  # For creating command-line interfaces
import sys    # For system operations like exit codes
import json   # For JSON data handling
from pathlib import Path  # Modern way to handle file paths

# Our application imports
from src.config_manager import get_config_manager   # Configuration management

@click.group()
def cli():
    """
    Photo Filter AI App - Organize photos and videos intelligently.

    This is the main command group that holds all our CLI commands.
    Run 'python main.py --help' to see all available commands.
    """
    pass

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def scan(verbose):
    """
    Scan existing Pictures library to analyze organization patterns.

    This command examines your already-organized photos in the Pictures folder
    to understand how they're currently structured. It helps us learn your
    organization preferences before processing new photos.

    What it does:
    1. Scans all photos/videos in the Pictures directory
    2. Analyzes folder naming patterns (date prefixes, locations, etc.)
    3. Counts different file types and sizes
    4. Reports on organization quality

    For junior developers:
    - The @click.option decorator adds command-line flags
    - verbose=True enables more detailed logging output
    - We use try/except for error handling
    - Results are displayed using click.echo() for formatted output
    """
    click.echo("🔍 Scanning existing Pictures library...")

    try:
        # Import components directly instead of using MediaProcessor
        from src.media_detector import MediaDetector
        from src.logging_utils import setup_logging

        # Set up logging
        log_level = "DEBUG" if verbose else "INFO"
        setup_logging(log_level)

        # Scan Pictures library
        detector = MediaDetector()
        organized_files = detector.scan_pictures_library()

        if not organized_files:
            click.echo("\n⚠️  No organized files found in Pictures library")
            return

        # Get statistics
        stats = detector.get_media_stats(organized_files)

        # Analyze organization patterns
        event_folders = {}
        for media_file in organized_files:
            event_folder = getattr(media_file, 'event_folder', 'Unknown')
            if event_folder not in event_folders:
                event_folders[event_folder] = {
                    'count': 0,
                    'photos': 0,
                    'videos': 0
                }

            event_info = event_folders[event_folder]
            event_info['count'] += 1

            if media_file.file_type == 'photo':
                event_info['photos'] += 1
            elif media_file.file_type == 'video':
                event_info['videos'] += 1

        # Analyze naming patterns
        patterns = {
            'date_prefixed': 0,
            'location_mentioned': 0,
            'to_filter_folders': 0
        }

        for folder_name in event_folders.keys():
            folder_lower = folder_name.lower()

            # Check for date prefix (YYYY_MM_DD pattern)
            if any(char.isdigit() for char in folder_name[:10]):
                patterns['date_prefixed'] += 1

            # Check for location mentions
            location_keywords = ['mexico', 'trip', 'vacation', 'birthday', 'wedding', 'calgary', 'edmonton']
            if any(keyword in folder_lower for keyword in location_keywords):
                patterns['location_mentioned'] += 1

            # Check for "to filter" folders
            if 'to filter' in folder_lower or 'to_filter' in folder_lower:
                patterns['to_filter_folders'] += 1

        # Display results
        click.echo(f"\n📊 Scan Results:")
        click.echo(f"  Total files: {stats['total_files']}")
        click.echo(f"  Photos: {stats['photos']}")
        click.echo(f"  Videos: {stats['videos']}")
        click.echo(f"  Total size: {stats['total_size_mb']:.1f} MB")
        click.echo(f"  Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")

        click.echo(f"\n📁 Organization Analysis:")
        click.echo(f"  Event folders: {len(event_folders)}")
        click.echo(f"  Date-prefixed folders: {patterns['date_prefixed']}")
        click.echo(f"  Location-mentioned folders: {patterns['location_mentioned']}")
        click.echo(f"  Folders needing filtering: {patterns['to_filter_folders']}")

        click.echo(f"\n✅ Scan completed successfully!")

    except Exception as e:
        click.echo(f"❌ Error during scanning: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--dry-run/--no-dry-run', default=True, help='Preview mode - don\'t create folders or move files (default: --dry-run)')
@click.option('--max-photos', type=int, default=10000, help='Maximum photos to process (default: 10000)')
@click.option('--mode', type=click.Choice(['copy', 'move']), default='copy', help='File operation mode (default: copy)')
@click.option('--save-report', is_flag=True, help='Save detailed report to file')
def process(verbose, dry_run, max_photos, mode, save_report):
    """
    Complete photo organization using intelligent clustering and file organization.

    This command performs the full photo organization workflow:
    1. Scans unfiltered photos in iPhone Automatic folder
    2. Performs intelligent clustering (temporal, location, content)
    3. Generates smart event names for each cluster
    4. Creates organized folder structure
    5. Moves/copies files to organized folders

    Use --no-dry-run to actually organize files.
    """
    if verbose:
        click.echo("🚀 Starting complete photo organization pipeline...")

    try:
        from src.photo_organizer_pipeline import PhotoOrganizerPipeline

        # Create progress callback for CLI updates
        def progress_callback(message, progress):
            if verbose:
                percentage = int(progress * 100)
                click.echo(f"[{percentage:3d}%] {message}")

        # Initialize pipeline
        pipeline = PhotoOrganizerPipeline(
            max_photos=max_photos,
            operation_mode=mode,
            dry_run=dry_run,
            verify_checksums=True
        )

        click.echo(f"📸 Organizing {max_photos} photos (mode: {mode}, dry-run: {dry_run})")

        # Run complete pipeline
        results = pipeline.run_complete_pipeline(progress_callback=progress_callback)

        # Display results
        pipeline_summary = results['pipeline_summary']
        stage_summaries = results['stage_summaries']

        click.echo(f"\n🎉 Pipeline completed in {pipeline_summary['execution_time_seconds']:.1f} seconds")

        # Vector database stage results
        if 'vector_db' in stage_summaries:
            vector_stage = stage_summaries['vector_db']
            click.echo(f"\n🗄️  Vector Database Results:")
            if vector_stage['vectorization_enabled']:
                click.echo(f"   Photos scanned: {vector_stage['photos_scanned']}")
                click.echo(f"   Photos vectorized: {vector_stage['photos_vectorized']}")
                click.echo(f"   Photos added to DB: {vector_stage['photos_added_to_db']}")
                click.echo(f"   Event folders processed: {vector_stage['event_folders_processed']}")
                if vector_stage['execution_time_seconds'] > 0:
                    click.echo(f"   Vectorization time: {vector_stage['execution_time_seconds']:.1f} seconds")
            else:
                click.echo(f"   Vectorization: Disabled or failed")

        # Scan stage results
        scan_stage = stage_summaries['scan']
        click.echo(f"\n📊 Scan Results:")
        click.echo(f"   Files scanned: {scan_stage['files_scanned']}")
        click.echo(f"   Files selected: {scan_stage['files_selected']}")

        # Clustering stage results
        clustering_stage = stage_summaries['clustering']
        click.echo(f"\n🧠 Clustering Results:")
        click.echo(f"   Clusters created: {clustering_stage['clusters_created']}")
        click.echo(f"   Average cluster size: {clustering_stage['avg_cluster_size']:.1f} files")

        # Naming stage results
        naming_stage = stage_summaries['naming']
        click.echo(f"\n🎯 Naming Results:")
        click.echo(f"   Naming success rate: {naming_stage['naming_success_rate']:.1%}")
        click.echo(f"   High confidence names: {naming_stage['high_confidence_names']}")

        # Folder stage results
        folder_stage = stage_summaries['folders']
        click.echo(f"\n📁 Folder Results:")
        click.echo(f"   Folders created: {folder_stage['folders_created']}")
        if folder_stage['conflicts_resolved'] > 0:
            click.echo(f"   Conflicts resolved: {folder_stage['conflicts_resolved']}")

        # Organization stage results
        org_stage = stage_summaries['organization']
        click.echo(f"\n📂 File Organization Results:")
        click.echo(f"   Files processed: {org_stage['files_processed']}")
        click.echo(f"   Files successful: {org_stage['files_successful']}")
        click.echo(f"   Success rate: {org_stage['success_rate']:.1%}")

        # Show output directory
        if results.get('output_directory'):
            click.echo(f"\n📁 Output directory: {results['output_directory']}")

        # Show mode-specific message
        if dry_run:
            click.echo(f"\n💡 This was a preview! Use --no-dry-run to actually organize files.")
        else:
            click.echo(f"\n✅ Photos successfully organized using {mode} mode!")

        # Show errors/warnings if any
        if results['errors']:
            click.echo(f"\n⚠️  {len(results['errors'])} errors occurred during processing")
            if verbose:
                for error in results['errors']:
                    click.echo(f"   • {error['stage']}: {error['error']}")

        # Save report if requested
        if save_report:
            report_file = pipeline.save_pipeline_report(results)
            click.echo(f"\n📄 Detailed report saved: {report_file}")

    except Exception as e:
        click.echo(f"❌ Error during processing: {e}")
        if verbose:
            import traceback
            click.echo(traceback.format_exc())
        sys.exit(1)

@cli.command()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--max-photos', default=10, help='Maximum photos to analyze')
def analyze_content(verbose, max_photos):
    """Analyze content of sample photos for testing."""
    click.echo(f"🔍 Analyzing photo content (max {max_photos} photos)...")

    try:
        # Import components directly instead of using MediaProcessor
        from src.media_detector import MediaDetector
        from src.content_analyzer import ContentAnalyzer
        from src.config import USE_GPU

        # Initialize components directly
        media_detector = MediaDetector()
        content_analyzer = ContentAnalyzer(use_gpu=USE_GPU)

        # Get sample photos
        all_files = media_detector.scan_iphone_automatic()
        photo_files = [f for f in all_files if f.file_type == 'photo']

        if not photo_files:
            click.echo("❌ No photos found for analysis")
            return

        sample_photos = photo_files[:max_photos]
        click.echo(f"Analyzing {len(sample_photos)} photos...")

        # Analyze content
        photo_paths = [f.path for f in sample_photos]
        content_results = content_analyzer.analyze_batch(photo_paths, max_photos=max_photos)

        if content_results:
            # Display individual results
            click.echo(f"\n📸 Individual Photo Analysis:")
            for i, (photo_path, analysis) in enumerate(content_results.items()):
                filename = Path(photo_path).name
                click.echo(f"\n  {i+1}. {filename}")
                click.echo(f"     Model: {analysis.analysis_model}")
                click.echo(f"     Confidence: {analysis.confidence_score:.2f}")
                click.echo(f"     Description: {analysis.description}")
                if analysis.objects:
                    click.echo(f"     Objects: {', '.join(analysis.objects)}")
                if analysis.scenes:
                    click.echo(f"     Scenes: {', '.join(analysis.scenes)}")
                if analysis.activities:
                    click.echo(f"     Activities: {', '.join(analysis.activities)}")

            # Display summary
            summary = content_analyzer.get_content_summary(content_results)
            click.echo(f"\n📊 Content Analysis Summary:")
            click.echo(f"  Photos analyzed: {summary['total_photos_analyzed']}")
            click.echo(f"  Average confidence: {summary['average_confidence']:.3f}")

            if summary['top_objects']:
                top_objects = [f"{obj} ({count})" for obj, count in summary['top_objects'][:5]]
                click.echo(f"  Top objects: {', '.join(top_objects)}")

            if summary['top_scenes']:
                top_scenes = [f"{scene} ({count})" for scene, count in summary['top_scenes'][:3]]
                click.echo(f"  Top scenes: {', '.join(top_scenes)}")

            if summary['top_activities']:
                top_activities = [f"{activity} ({count})" for activity, count in summary['top_activities'][:3]]
                click.echo(f"  Top activities: {', '.join(top_activities)}")

        else:
            click.echo("❌ No content analysis results available")

        click.echo(f"\n✅ Content analysis completed!")

        # Cleanup
        content_analyzer.cleanup()

    except Exception as e:
        click.echo(f"❌ Error during content analysis: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

@cli.command()
def status():
    """Show system status and configuration."""
    click.echo("🔧 Photo Filter System Status")

    try:
        # Import config constants directly instead of using MediaProcessor
        from src.config import (
            BASE_DIR, SAMPLE_PHOTOS_DIR, IPHONE_AUTOMATIC_DIR,
            PICTURES_DIR, VECTOR_DB_DIR, TIME_THRESHOLD_HOURS,
            LOCATION_THRESHOLD_KM, MIN_CLUSTER_SIZE, SUPPORTED_EXTENSIONS
        )

        # Directory status
        directories = {
            'base_dir': str(BASE_DIR),
            'sample_photos_dir': str(SAMPLE_PHOTOS_DIR),
            'iphone_automatic_dir': str(IPHONE_AUTOMATIC_DIR),
            'pictures_dir': str(PICTURES_DIR),
            'vector_db_dir': str(VECTOR_DB_DIR)
        }

        click.echo(f"\n📁 Directories:")
        for name, path in directories.items():
            exists = "✅" if Path(path).exists() else "❌"
            click.echo(f"  {name}: {exists} {path}")

        # Configuration
        click.echo(f"\n⚙️  Configuration:")
        click.echo(f"  Time threshold: {TIME_THRESHOLD_HOURS} hours")
        click.echo(f"  Location threshold: {LOCATION_THRESHOLD_KM} km")
        click.echo(f"  Min cluster size: {MIN_CLUSTER_SIZE} files")
        click.echo(f"  Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")

    except Exception as e:
        click.echo(f"❌ Error getting status: {e}")
        sys.exit(1)

@cli.command()
@click.argument('cluster_id', type=int)
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
def show_cluster(cluster_id, verbose):
    """Show detailed information about a specific cluster."""
    click.echo(f"🔍 Showing details for cluster {cluster_id}...")

    # This would require storing cluster results or re-running analysis
    # For now, show a placeholder message
    click.echo("📋 Cluster details not yet implemented.")
    click.echo("💡 Run 'python main.py process' to see cluster suggestions.")

@cli.group()
def config():
    """Manage configuration settings."""
    pass

@config.command()
def show():
    """Show current configuration."""
    click.echo("⚙️  Current Configuration")

    try:
        config_mgr = get_config_manager()
        summary = config_mgr.get_config_summary()

        click.echo(f"\n📄 Config file: {summary['config_file']}")
        click.echo(f"   Exists: {'✅' if summary['config_exists'] else '❌'}")

        # Clustering settings
        clustering = summary['clustering']
        click.echo(f"\n🎯 Clustering Settings:")
        click.echo(f"   Time threshold: {clustering['time_threshold_hours']} hours")
        click.echo(f"   Location threshold: {clustering['location_threshold_km']} km")
        click.echo(f"   Min cluster size: {clustering['min_cluster_size']} files")

        # Processing settings
        processing = summary['processing']
        click.echo(f"\n🔧 Processing Settings:")
        click.echo(f"   Max photos per event: {processing['max_photos_per_event']}")
        click.echo(f"   Use GPU: {'✅' if processing['use_gpu'] else '❌'}")
        click.echo(f"   Enable vectorization: {'✅' if processing['enable_vectorization'] else '❌'}")

        # Paths
        paths = summary['paths']
        click.echo(f"\n📁 Paths:")
        click.echo(f"   Sample photos: {paths['sample_photos_dir']}")
        click.echo(f"   Vector database: {paths['vector_db_dir']}")

        # Validation
        validation = summary['validation']
        if validation['valid']:
            click.echo(f"\n✅ Configuration is valid")
        else:
            click.echo(f"\n❌ Configuration has errors:")
            for error in validation['errors']:
                click.echo(f"     • {error}")

        if validation['warnings']:
            click.echo(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                click.echo(f"     • {warning}")

    except Exception as e:
        click.echo(f"❌ Error reading configuration: {e}")

@config.command()
@click.option('--time-threshold', type=float, help='Time threshold in hours')
@click.option('--location-threshold', type=float, help='Location threshold in km')
@click.option('--min-cluster-size', type=int, help='Minimum cluster size')
@click.option('--max-photos', type=int, help='Max photos per event')
@click.option('--use-gpu/--no-gpu', help='Enable/disable GPU usage')
def update(time_threshold, location_threshold, min_cluster_size, max_photos, use_gpu):
    """Update configuration parameters."""
    click.echo("🔧 Updating configuration...")

    try:
        config_mgr = get_config_manager()

        updates_made = []

        # Update clustering parameters
        clustering_updates = {}
        if time_threshold is not None:
            clustering_updates['time_threshold_hours'] = time_threshold
            updates_made.append(f"Time threshold: {time_threshold} hours")

        if location_threshold is not None:
            clustering_updates['location_threshold_km'] = location_threshold
            updates_made.append(f"Location threshold: {location_threshold} km")

        if min_cluster_size is not None:
            clustering_updates['min_cluster_size'] = min_cluster_size
            updates_made.append(f"Min cluster size: {min_cluster_size}")

        if clustering_updates:
            config_mgr.update_clustering_config(**clustering_updates)

        # Update processing parameters
        processing_updates = {}
        if max_photos is not None:
            processing_updates['max_photos_per_event'] = max_photos
            updates_made.append(f"Max photos per event: {max_photos}")

        if use_gpu is not None:
            processing_updates['use_gpu'] = use_gpu
            updates_made.append(f"GPU usage: {'enabled' if use_gpu else 'disabled'}")

        if processing_updates:
            config_mgr.update_processing_config(**processing_updates)

        if updates_made:
            click.echo(f"\n✅ Configuration updated:")
            for update in updates_made:
                click.echo(f"   • {update}")
        else:
            click.echo(f"\n💡 No parameters specified to update")
            click.echo(f"   Use --help to see available options")

    except Exception as e:
        click.echo(f"❌ Error updating configuration: {e}")

@config.command()
def reset():
    """Reset configuration to defaults."""
    click.echo("🔄 Resetting configuration to defaults...")

    try:
        config_mgr = get_config_manager()
        config_mgr.reset_to_defaults()
        click.echo("✅ Configuration reset to defaults")

    except Exception as e:
        click.echo(f"❌ Error resetting configuration: {e}")

@config.command()
def validate():
    """Validate current configuration."""
    click.echo("🔍 Validating configuration...")

    try:
        config_mgr = get_config_manager()
        validation = config_mgr.validate_config()

        if validation['valid']:
            click.echo("✅ Configuration is valid")
        else:
            click.echo("❌ Configuration has errors:")
            for error in validation['errors']:
                click.echo(f"   • {error}")

        if validation['warnings']:
            click.echo("⚠️  Warnings:")
            for warning in validation['warnings']:
                click.echo(f"   • {warning}")

    except Exception as e:
        click.echo(f"❌ Error validating configuration: {e}")

@cli.group()
def faces():
    """Manage face recognition and people identification."""
    pass

@faces.command()
@click.option('--max-events', type=int, default=5, help='Maximum event folders to scan (default: 5)')
@click.option('--max-photos-per-event', type=int, default=20, help='Maximum photos per event to process (default: 20)')
def scan(max_events, max_photos_per_event):
    """Scan existing organized photos to build face recognition database."""
    click.echo("👥 Scanning organized photos for face recognition...")

    try:
        from src.face_recognizer import FaceRecognizer
        from src.people_database import PeopleDatabase
        from src.organized_photos_scanner import OrganizedPhotosScanner
        from src.config_manager import get_config

        # Check if face recognition is enabled
        config = get_config()
        if not config.faces.enable_face_detection:
            click.echo("❌ Face detection is disabled in configuration")
            click.echo("💡 Enable with: python main.py config update --enable-face-detection")
            return

        # Initialize components
        face_recognizer = FaceRecognizer(
            detection_model=config.faces.detection_model,
            recognition_tolerance=config.faces.recognition_tolerance,
            enable_caching=config.faces.store_encodings
        )

        if not face_recognizer.enabled:
            click.echo("❌ Face recognition not available (missing dependencies)")
            return

        people_db = PeopleDatabase()
        scanner = OrganizedPhotosScanner()

        # Scan organized photos
        results = scanner.scan_and_build_database(
            max_photos_per_event=max_photos_per_event,
            max_events=max_events,
            quick_scan=True
        )

        # Display results
        click.echo(f"\n📊 Scan Results:")
        click.echo(f"   Events found: {results['total_events_found']}")
        click.echo(f"   Events processed: {results['events_processed']}")
        click.echo(f"   Photos processed: {results['photos_processed']}")
        click.echo(f"   Photos vectorized: {results['photos_vectorized']}")

        # Display people statistics
        stats = people_db.get_statistics()
        click.echo(f"\n👥 People Database:")
        click.echo(f"   Known people: {stats['total_people']}")
        click.echo(f"   Total encodings: {stats['total_encodings']}")

        if results['photos_vectorized'] > 0:
            click.echo("✅ Face recognition database updated successfully")
        else:
            click.echo("⚠️  No faces were detected or processed")

    except Exception as e:
        click.echo(f"❌ Error scanning for faces: {e}")

@faces.command()
@click.argument('person_name')
@click.argument('image_paths', nargs=-1, required=True)
def add(person_name, image_paths):
    """Add a person to the face recognition database.

    PERSON_NAME: Name of the person to add
    IMAGE_PATHS: One or more image files containing this person
    """
    click.echo(f"👤 Adding person '{person_name}' to face recognition database...")

    try:
        from src.face_recognizer import FaceRecognizer
        from src.people_database import PeopleDatabase
        from src.config_manager import get_config
        from pathlib import Path

        config = get_config()
        if not config.faces.enable_face_detection:
            click.echo("❌ Face detection is disabled in configuration")
            return

        # Initialize components - use shared database
        people_db = PeopleDatabase()
        face_recognizer = FaceRecognizer(
            detection_model=config.faces.detection_model,
            recognition_tolerance=config.faces.recognition_tolerance,
            people_database=people_db
        )

        # Convert paths and validate
        image_paths_list = [Path(p) for p in image_paths]
        for path in image_paths_list:
            if not path.exists():
                click.echo(f"❌ Image not found: {path}")
                return

        # Add person
        success = face_recognizer.add_person(person_name, image_paths_list)

        if success:
            click.echo(f"✅ Successfully added '{person_name}' to face recognition database")

            # Display updated statistics
            stats = face_recognizer.get_statistics()
            click.echo(f"👥 Database now contains {stats['known_people']} people")
        else:
            click.echo(f"❌ Failed to add '{person_name}' (no faces found or error occurred)")

    except Exception as e:
        click.echo(f"❌ Error adding person: {e}")

@faces.command()
def list():
    """List all people in the face recognition database."""
    click.echo("👥 People in Face Recognition Database")

    try:
        from src.people_database import PeopleDatabase

        people_db = PeopleDatabase()
        people_list = people_db.export_person_list()

        if not people_list:
            click.echo("📭 No people in database")
            click.echo("💡 Add people with: python main.py faces add <name> <image_files>")
            return

        click.echo(f"\n📊 Total: {len(people_list)} people\n")

        for person in people_list:
            click.echo(f"👤 {person['name']} ({person['person_id']})")
            click.echo(f"   Encodings: {person['encoding_count']}")
            click.echo(f"   Photos: {person['photo_count']}")
            click.echo(f"   Added: {person['created_date'][:10]}")
            if person['notes']:
                click.echo(f"   Notes: {person['notes']}")
            click.echo()

    except Exception as e:
        click.echo(f"❌ Error listing people: {e}")

@faces.command()
@click.argument('person_name')
def remove(person_name):
    """Remove a person from the face recognition database.

    PERSON_NAME: Name of the person to remove
    """
    click.echo(f"🗑️  Removing person '{person_name}' from face recognition database...")

    try:
        from src.people_database import PeopleDatabase

        people_db = PeopleDatabase()

        # Find person by name
        person = people_db.find_person_by_name(person_name)
        if not person:
            click.echo(f"❌ Person '{person_name}' not found in database")
            return

        # Confirm deletion
        if click.confirm(f"Are you sure you want to remove '{person.name}'?"):
            success = people_db.remove_person(person.person_id)
            if success:
                click.echo(f"✅ Successfully removed '{person_name}' from database")
            else:
                click.echo(f"❌ Failed to remove '{person_name}'")
        else:
            click.echo("Cancelled")

    except Exception as e:
        click.echo(f"❌ Error removing person: {e}")

@faces.command()
def status():
    """Show face recognition system status."""
    click.echo("👥 Face Recognition System Status")

    try:
        from src.face_recognizer import FaceRecognizer
        from src.people_database import PeopleDatabase
        from src.config_manager import get_config

        config = get_config()

        click.echo(f"\n⚙️  Configuration:")
        click.echo(f"   Face detection enabled: {'✅' if config.faces.enable_face_detection else '❌'}")
        click.echo(f"   Detection model: {config.faces.detection_model}")
        click.echo(f"   Recognition tolerance: {config.faces.recognition_tolerance}")
        click.echo(f"   Store encodings: {'✅' if config.faces.store_encodings else '❌'}")

        if config.faces.enable_face_detection:
            # Test face recognizer
            face_recognizer = FaceRecognizer()
            stats = face_recognizer.get_statistics()

            click.echo(f"\n🔍 Face Recognizer:")
            click.echo(f"   Available: {'✅' if stats['enabled'] else '❌'}")
            click.echo(f"   Cached images: {stats['cached_images']}")
            click.echo(f"   Known people: {stats['known_people']}")
            click.echo(f"   Total encodings: {stats['total_known_encodings']}")

            # People database stats
            people_db = PeopleDatabase()
            people_stats = people_db.get_statistics()

            click.echo(f"\n👥 People Database:")
            click.echo(f"   Total people: {people_stats['total_people']}")
            click.echo(f"   Total encodings: {people_stats['total_encodings']}")
            click.echo(f"   Database size: {people_stats['database_file_size']} bytes")

        else:
            click.echo("\n💡 To enable face recognition:")
            click.echo("   python main.py config update --enable-face-detection")

    except Exception as e:
        click.echo(f"❌ Error checking status: {e}")

# Add face recognition options to config update command
@config.command()
@click.option('--time-threshold', type=float, help='Time threshold for clustering (hours)')
@click.option('--location-threshold', type=float, help='Location threshold for clustering (km)')
@click.option('--min-cluster-size', type=int, help='Minimum cluster size')
@click.option('--enable-face-detection', is_flag=True, help='Enable face detection')
@click.option('--disable-face-detection', is_flag=True, help='Disable face detection')
@click.option('--face-model', type=click.Choice(['hog', 'cnn']), help='Face detection model')
@click.option('--face-tolerance', type=float, help='Face recognition tolerance (0.0-1.0)')
def update(time_threshold, location_threshold, min_cluster_size,
          enable_face_detection, disable_face_detection, face_model, face_tolerance):
    """Update configuration settings."""
    click.echo("⚙️  Updating Configuration")

    try:
        config_mgr = get_config_manager()
        config = config_mgr.load_config()

        # Update clustering settings
        if time_threshold is not None:
            config.clustering.time_threshold_hours = time_threshold
            click.echo(f"   Time threshold: {time_threshold} hours")

        if location_threshold is not None:
            config.clustering.location_threshold_km = location_threshold
            click.echo(f"   Location threshold: {location_threshold} km")

        if min_cluster_size is not None:
            config.clustering.min_cluster_size = min_cluster_size
            click.echo(f"   Minimum cluster size: {min_cluster_size}")

        # Update face recognition settings
        if enable_face_detection:
            config.faces.enable_face_detection = True
            click.echo("   Face detection: ✅ Enabled")

        if disable_face_detection:
            config.faces.enable_face_detection = False
            click.echo("   Face detection: ❌ Disabled")

        if face_model is not None:
            config.faces.detection_model = face_model
            click.echo(f"   Face detection model: {face_model}")

        if face_tolerance is not None:
            config.faces.recognition_tolerance = face_tolerance
            click.echo(f"   Face recognition tolerance: {face_tolerance}")

        # Save updated configuration
        config_mgr.save_config(config)
        click.echo("✅ Configuration updated successfully")

    except Exception as e:
        click.echo(f"❌ Error updating configuration: {e}")

if __name__ == "__main__":
    cli()