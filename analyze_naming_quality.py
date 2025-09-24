#!/usr/bin/env python3
"""
Analyze and compare event naming quality.

This script helps evaluate the quality of event names before and after improvements.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import re

def load_naming_cache(cache_file: Path) -> Dict[str, str]:
    """Load naming cache from JSON file."""
    if not cache_file.exists():
        return {}

    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {cache_file}: {e}")
        return {}

def analyze_naming_issues(cache: Dict[str, str]) -> Dict[str, List[str]]:
    """Analyze naming cache for common issues."""
    issues = {
        'geographic_nonsense': [],
        'repetitive_generic': [],
        'inappropriate_seasonal': [],
        'poor_descriptive': [],
        'good_names': []
    }

    # Track name frequency for repetition detection
    name_counts = {}

    for context_key, name in cache.items():
        # Extract location from context key (format: time|duration|day|weekend|location|...)
        context_parts = context_key.split('|')
        context_location = context_parts[4] if len(context_parts) > 4 else ''

        # Extract actual name part (after date prefix)
        name_parts = name.split(' - ', 1)
        event_name = name_parts[1] if len(name_parts) > 1 else name

        # Count occurrences
        if event_name in name_counts:
            name_counts[event_name] += 1
        else:
            name_counts[event_name] = 1

        # Check for geographic nonsense
        inappropriate_locations = ['Paris', 'Vegas', 'NYC', 'Hawaii', 'Beach']
        if any(loc in name for loc in inappropriate_locations) and context_location == 'Edmonton':
            issues['geographic_nonsense'].append(f"{name} (context: {context_location})")

        # Check for seasonal inappropriateness
        if 'Beach' in name and ('winter' in context_key.lower() or 'November' in name or 'December' in name or 'January' in name or 'February' in name):
            issues['inappropriate_seasonal'].append(f"{name} (appears to be winter)")

        # Check for poor descriptive quality
        generic_terms = ['Walk', 'Outing', 'Event', 'Activity']
        if any(term in event_name for term in generic_terms) and len(event_name.split()) <= 2:
            issues['poor_descriptive'].append(name)

        # Identify potentially good names
        good_indicators = ['Party', 'Trip', 'Vacation', 'Birthday', 'Wedding', 'Graduation', 'Christmas', 'Holiday']
        if any(indicator in name for indicator in good_indicators):
            issues['good_names'].append(name)

    # Identify repetitive names (appearing more than 3 times)
    for event_name, count in name_counts.items():
        if count > 3:
            issues['repetitive_generic'].append(f"{event_name} (appears {count} times)")

    return issues

def compare_naming_quality(old_cache: Dict[str, str], new_cache: Dict[str, str]) -> Dict[str, any]:
    """Compare naming quality between old and new caches."""
    old_issues = analyze_naming_issues(old_cache)
    new_issues = analyze_naming_issues(new_cache)

    comparison = {
        'old_cache_size': len(old_cache),
        'new_cache_size': len(new_cache),
        'issues_comparison': {},
        'improvement_summary': {},
        'sample_changes': []
    }

    # Compare issue counts
    for issue_type in old_issues.keys():
        old_count = len(old_issues[issue_type])
        new_count = len(new_issues[issue_type])
        comparison['issues_comparison'][issue_type] = {
            'old': old_count,
            'new': new_count,
            'change': new_count - old_count,
            'improvement': old_count > new_count
        }

    # Calculate overall improvement
    total_old_issues = sum(len(issues) for key, issues in old_issues.items() if key != 'good_names')
    total_new_issues = sum(len(issues) for key, issues in new_issues.items() if key != 'good_names')

    comparison['improvement_summary'] = {
        'total_old_issues': total_old_issues,
        'total_new_issues': total_new_issues,
        'net_improvement': total_old_issues - total_new_issues,
        'improvement_percentage': ((total_old_issues - total_new_issues) / max(total_old_issues, 1)) * 100
    }

    # Find sample changes
    common_contexts = set(old_cache.keys()) & set(new_cache.keys())
    changes = []
    for context in list(common_contexts)[:10]:  # Sample first 10 changes
        if old_cache[context] != new_cache[context]:
            changes.append({
                'context': context,
                'old_name': old_cache[context],
                'new_name': new_cache[context]
            })

    comparison['sample_changes'] = changes

    return comparison

def print_analysis_report(issues: Dict[str, List[str]], title: str = "Naming Analysis"):
    """Print a detailed analysis report."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")

    print(f"\n📊 Issue Summary:")
    print(f"   Geographic nonsense: {len(issues['geographic_nonsense'])}")
    print(f"   Repetitive/generic: {len(issues['repetitive_generic'])}")
    print(f"   Inappropriate seasonal: {len(issues['inappropriate_seasonal'])}")
    print(f"   Poor descriptive: {len(issues['poor_descriptive'])}")
    print(f"   Good names: {len(issues['good_names'])}")

    # Show examples of each issue type
    for issue_type, examples in issues.items():
        if examples and issue_type != 'good_names':
            print(f"\n❌ {issue_type.replace('_', ' ').title()} Examples:")
            for example in examples[:5]:  # Show first 5 examples
                print(f"   • {example}")
            if len(examples) > 5:
                print(f"   ... and {len(examples) - 5} more")

    # Show good examples
    if issues['good_names']:
        print(f"\n✅ Good Name Examples:")
        for example in issues['good_names'][:5]:
            print(f"   • {example}")

def print_comparison_report(comparison: Dict[str, any]):
    """Print a comparison report between old and new naming."""
    print(f"\n{'='*60}")
    print("NAMING QUALITY COMPARISON")
    print(f"{'='*60}")

    print(f"\n📈 Overall Improvement:")
    summary = comparison['improvement_summary']
    print(f"   Old issues: {summary['total_old_issues']}")
    print(f"   New issues: {summary['total_new_issues']}")
    print(f"   Net improvement: {summary['net_improvement']}")
    print(f"   Improvement percentage: {summary['improvement_percentage']:.1f}%")

    print(f"\n📋 Issue Type Breakdown:")
    for issue_type, data in comparison['issues_comparison'].items():
        if issue_type == 'good_names':
            continue
        status = "✅ IMPROVED" if data['improvement'] else "❌ WORSE" if data['change'] > 0 else "➖ SAME"
        print(f"   {issue_type.replace('_', ' ').title()}: {data['old']} → {data['new']} ({status})")

    if comparison['sample_changes']:
        print(f"\n🔄 Sample Name Changes:")
        for change in comparison['sample_changes'][:5]:
            print(f"   Context: {change['context'][:50]}...")
            print(f"     Old: {change['old_name']}")
            print(f"     New: {change['new_name']}")
            print()

def main():
    """Main analysis function."""
    print("🔍 Event Naming Quality Analyzer")

    # File paths
    original_cache = Path("data/event_naming_cache_backup_original.json")
    current_cache = Path("data/event_naming_cache.json")

    # Load caches
    print(f"\n📂 Loading caches...")
    original_names = load_naming_cache(original_cache)
    current_names = load_naming_cache(current_cache)

    print(f"   Original cache: {len(original_names)} entries")
    print(f"   Current cache: {len(current_names)} entries")

    # Analyze original cache
    if original_names:
        original_issues = analyze_naming_issues(original_names)
        print_analysis_report(original_issues, "ORIGINAL NAMING ANALYSIS")

    # Analyze current cache if different
    if current_names and current_names != original_names:
        current_issues = analyze_naming_issues(current_names)
        print_analysis_report(current_issues, "CURRENT NAMING ANALYSIS")

        # Compare the two
        comparison = compare_naming_quality(original_names, current_names)
        print_comparison_report(comparison)
    else:
        print("\n📝 Current cache is same as original - no changes to compare")

    # Recommendations
    print(f"\n💡 Recommendations:")
    print("   1. Clear cache if too many geographic nonsense names")
    print("   2. Improve LLM prompts if repetitive/generic names")
    print("   3. Add seasonal awareness if inappropriate seasonal names")
    print("   4. Enhance content analysis if poor descriptive quality")

if __name__ == "__main__":
    main()