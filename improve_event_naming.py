#!/usr/bin/env python3
"""
Improve event naming system with validation and better prompts.
"""

import json
import sys
from pathlib import Path

def backup_and_clear_cache():
    """Backup current cache and clear it for fresh start."""
    cache_file = Path("data/event_naming_cache.json")
    backup_file = Path("data/event_naming_cache_backup_before_improvements.json")

    if cache_file.exists():
        # Create another backup before clearing
        with open(cache_file, 'r') as f:
            cache_data = json.load(f)

        with open(backup_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"✅ Backed up current cache to {backup_file}")

        # Clear the cache for fresh start
        with open(cache_file, 'w') as f:
            json.dump({}, f)

        print(f"✅ Cleared cache for fresh start")
        return len(cache_data)
    else:
        print("ℹ️ No existing cache file to backup")
        return 0

def improve_event_namer_prompt():
    """Improve the LLM prompt in EventNamer to be more accurate."""

    event_namer_file = Path("src/event_namer.py")

    # Read current content
    with open(event_namer_file, 'r') as f:
        content = f.read()

    # Find and replace the prompt building section
    old_prompt_section = '''**Examples:**
- 2024_10_31 - Halloween Party - Edmonton
- 2024_07_04 - Beach Day - Vancouver
- 2024_12_25 - Christmas Morning - Home
- 2024_06_15 - Sarah & Mike Wedding - Calgary
- 2024_08_20 - Hiking Adventure - Banff
- 2024_05_10 - Emma Birthday Party - Home
- 2024_03_15 - Family Dinner - Restaurant

Generate the folder name:"""'''

    new_prompt_section = '''**IMPORTANT CONSTRAINTS:**
- ONLY use the provided location: {location['city'] or 'Unknown'}
- DO NOT invent or hallucinate other locations (no Paris, Vegas, Hawaii, etc.)
- Be specific and descriptive, avoid generic terms like "Walk", "Outing"
- Consider the season and weather for the location
- If no specific activity detected, use time/duration/setting context

**Examples for Edmonton (winter city):**
- 2024_01_15 - Indoor Family Gathering - Edmonton
- 2024_07_20 - Summer Festival - Edmonton
- 2024_11_10 - Autumn Photography Session - Edmonton
- 2024_12_25 - Christmas Morning - Home
- 2024_06_15 - Outdoor Concert - Edmonton
- 2024_03_20 - Spring Garden Visit - Edmonton

**Examples for other locations:**
- 2024_08_10 - Beach Day - Vancouver
- 2024_09_05 - Mountain Hiking - Calgary

Generate ONLY the folder name using the EXACT location provided above:"""'''

    if old_prompt_section in content:
        improved_content = content.replace(old_prompt_section, new_prompt_section)

        # Write improved version
        with open(event_namer_file, 'w') as f:
            f.write(improved_content)

        print("✅ Improved LLM prompt with location constraints")
        return True
    else:
        print("⚠️ Could not find exact prompt section to replace")
        return False

def improve_template_naming():
    """Improve template-based naming to be more location and context aware."""

    event_namer_file = Path("src/event_namer.py")

    with open(event_namer_file, 'r') as f:
        content = f.read()

    # Find the template naming method and improve it
    # Look for the section that builds template names
    old_template_logic = '''        # Add location if available
        location_part = ""
        if location['city']:
            if location['location_nickname']:
                location_part = f" - {location['location_nickname']}"
            elif not template.endswith(location['city']):
                location_part = f" - {location['city']}"'''

    new_template_logic = '''        # Make template more specific based on location and season
        if location['city'] == 'Edmonton':
            # Edmonton-specific improvements
            if temporal['season'] == 'winter' and 'outdoor' in content['scenes']:
                template = "Winter Outdoor Activity"
            elif temporal['season'] == 'summer' and 'outdoor' in content['scenes']:
                template = "Summer Outdoor Activity"
            elif 'indoor' in content['scenes'] and temporal['time_of_day'] == 'evening':
                template = "Indoor Evening Event"

        # Avoid generic "Walk" - be more specific
        if template == "Morning Activity" and 'outdoor' in content['scenes']:
            if temporal['season'] == 'winter':
                template = "Winter Morning Activity"
            else:
                template = "Outdoor Morning Session"

        # Add location if available
        location_part = ""
        if location['city']:
            if location['location_nickname']:
                location_part = f" - {location['location_nickname']}"
            elif not template.endswith(location['city']):
                location_part = f" - {location['city']}"'''

    if old_template_logic in content:
        improved_content = content.replace(old_template_logic, new_template_logic)

        with open(event_namer_file, 'w') as f:
            f.write(improved_content)

        print("✅ Improved template naming logic")
        return True
    else:
        print("⚠️ Could not find exact template logic to replace")
        return False

def add_name_validation():
    """Add validation to reject obviously bad names before caching."""

    event_namer_file = Path("src/event_namer.py")

    with open(event_namer_file, 'r') as f:
        content = f.read()

    # Add validation method after the generate_event_name method
    validation_method = '''
    def _validate_event_name(self, event_name: str, context: Dict[str, Any]) -> bool:
        """
        Validate event name for obvious issues before caching.

        Args:
            event_name: Generated event name
            context: Event context used for generation

        Returns:
            True if name is acceptable, False if it should be rejected
        """
        location = context['location']
        temporal = context['temporal']

        # Extract the descriptive part (after date)
        parts = event_name.split(' - ', 1)
        description = parts[1] if len(parts) > 1 else event_name

        # Reject names with wrong geography
        wrong_locations = ['Paris', 'Vegas', 'Hawaii', 'NYC', 'Beach']
        actual_location = location['city'] or ''

        for wrong_loc in wrong_locations:
            if wrong_loc in description and wrong_loc not in actual_location:
                self.logger.warning(f"Rejecting geographically inappropriate name: {event_name}")
                return False

        # Reject seasonal mismatches
        if 'Beach' in description and temporal['season'] == 'winter':
            self.logger.warning(f"Rejecting seasonally inappropriate name: {event_name}")
            return False

        # Reject overly generic names if we have better context
        generic_terms = ['Walk', 'Outing', 'Event', 'Activity']
        if any(term in description for term in generic_terms):
            # Check if we have specific content that could make it better
            content = context['content']
            if content['activities'] or content['event_type'] != 'general':
                # We have specific context, so generic name is poor quality
                if len(description.split()) <= 2:  # Very short generic description
                    self.logger.warning(f"Rejecting overly generic name: {event_name}")
                    return False

        return True
'''

    # Find where to insert the validation method (before _build_event_context)
    insert_point = "    def _build_event_context(self"

    if insert_point in content:
        content = content.replace(insert_point, validation_method + "\n    def _build_event_context(self")

        # Now update the generate_event_name method to use validation
        old_caching = '''            # Cache the result for similar future events
            self.naming_cache[cache_key] = event_name
            self._save_cache()'''

        new_caching = '''            # Validate the name before caching
            if self._validate_event_name(event_name, context):
                # Cache the result for similar future events
                self.naming_cache[cache_key] = event_name
                self._save_cache()
            else:
                # Use simple fallback for rejected names
                event_name = self._generate_simple_name(context)
                self.logger.info(f"Used simple fallback after rejection: {event_name}")'''

        if old_caching in content:
            content = content.replace(old_caching, new_caching)

        with open(event_namer_file, 'w') as f:
            f.write(content)

        print("✅ Added name validation system")
        return True
    else:
        print("⚠️ Could not find insertion point for validation method")
        return False

def main():
    """Main improvement function."""
    print("🔧 Improving Event Naming System")
    print("=" * 50)

    # Step 1: Backup and clear cache
    print("\n📦 Step 1: Backup and clear cache...")
    cleared_count = backup_and_clear_cache()
    print(f"   Cleared {cleared_count} cached entries for fresh start")

    # Step 2: Improve LLM prompt
    print("\n✍️ Step 2: Improve LLM prompt...")
    prompt_improved = improve_event_namer_prompt()

    # Step 3: Improve template naming
    print("\n🎯 Step 3: Improve template naming...")
    template_improved = improve_template_naming()

    # Step 4: Add validation
    print("\n✅ Step 4: Add name validation...")
    validation_added = add_name_validation()

    # Summary
    print(f"\n📋 Improvement Summary:")
    print(f"   Cache cleared: ✅")
    print(f"   LLM prompt improved: {'✅' if prompt_improved else '❌'}")
    print(f"   Template naming improved: {'✅' if template_improved else '❌'}")
    print(f"   Validation added: {'✅' if validation_added else '❌'}")

    if all([prompt_improved, template_improved, validation_added]):
        print(f"\n🎉 All improvements applied successfully!")
        print(f"\n📝 Next steps:")
        print(f"   1. Test the improved system with photo processing")
        print(f"   2. Run analyze_naming_quality.py to compare results")
        print(f"   3. Roll back if quality doesn't improve")
    else:
        print(f"\n⚠️ Some improvements failed - review the changes manually")

if __name__ == "__main__":
    main()