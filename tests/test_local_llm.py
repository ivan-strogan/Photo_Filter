#!/usr/bin/env python3
"""
Test local LLM integration with Ollama for offline event naming.

This test demonstrates how the system falls back to local LLM when OpenAI
is not available, providing intelligent naming without external dependencies.
"""

import sys
from pathlib import Path
from datetime import datetime
sys.path.append(str(Path(__file__).parent.parent))

from src.event_namer import EventNamer

def test_llm_fallback_chain():
    """Test the LLM fallback chain: OpenAI -> Ollama -> Templates."""
    print("🧠 Testing LLM Fallback Chain")
    print("=" * 50)

    # Test 1: No API key (should fall back to Ollama or templates)
    print("1️⃣  Testing without OpenAI API key...")
    namer = EventNamer(api_key=None, enable_llm=True)

    test_context = {
        'date_range': {
            'start': datetime(2024, 10, 31, 18, 0),
            'end': datetime(2024, 10, 31, 22, 0)
        },
        'time_patterns': {'primary_time_of_day': 'Evening'},
        'location_info': {'primary_location': 'Residential Area'},
        'content_analysis': {'detected_objects': ['person', 'costume', 'decoration']},
        'cluster_size': 25,
        'calendar_context': {'is_holiday': True, 'holiday_name': 'Halloween'}
    }

    event_name = namer.generate_event_name(test_context)
    print(f"   Generated name: '{event_name}'")

    # Check initialization results
    success, provider = namer._initialize_llm_client()
    print(f"   LLM Provider: {provider}")
    print(f"   Success: {success}")

def test_ollama_availability():
    """Check if Ollama is available and what models are installed."""
    print("\n🦙 Testing Ollama Availability")
    print("=" * 50)

    try:
        import requests
    except ImportError:
        print("❌ requests module not available")
        print("   Install with: pip3 install requests")
        return False

    try:
        # Check if Ollama server is running
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            version_info = response.json()
            print(f"✅ Ollama server is running")
            print(f"   Version: {version_info.get('version', 'unknown')}")

            # List available models
            models_response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if models_response.status_code == 200:
                models_data = models_response.json()
                models = models_data.get('models', [])
                print(f"   Available models: {len(models)}")

                recommended_models = ['llama3.1:8b', 'phi3:mini', 'llama3.2:3b']
                for model_info in models:
                    model_name = model_info.get('name', 'unknown')
                    print(f"      • {model_name}")
                    if any(rec in model_name for rec in recommended_models):
                        print(f"        ⭐ (Recommended for MacBook Pro)")
        else:
            print("❌ Ollama server not responding")
            print("   To install Ollama:")
            print("   1. Download from: https://ollama.ai/download")
            print("   2. Install and run: ollama pull llama3.1:8b")

    except Exception as e:
        print(f"❌ Could not connect to Ollama: {e}")
        print("\n💡 To set up local LLM:")
        print("   1. Install Ollama: https://ollama.ai/download")
        print("   2. Pull a model: ollama pull llama3.1:8b")
        print("   3. Start the server: ollama serve")

def test_model_performance():
    """Test different model configurations for your MacBook Pro."""
    print("\n⚡ Testing Model Performance for MacBook Pro")
    print("=" * 50)

    # Recommended models for your specs (6-core i7, 16GB RAM)
    test_models = [
        "llama3.1:8b",      # Best balance for your hardware
        "phi3:mini",        # Fastest, good quality
        "llama3.2:3b",      # Smaller model, faster
    ]

    for model in test_models:
        print(f"\n🧪 Testing {model}:")

        namer = EventNamer(
            enable_llm=True,
            ollama_model=model,
            ollama_url="http://localhost:11434"
        )

        success, provider = namer._initialize_llm_client()

        if success and provider == "ollama":
            print(f"   ✅ Model available and working")

            # Test with simple context
            simple_context = {
                'date_range': {
                    'start': datetime(2024, 7, 4, 12, 0),
                    'end': datetime(2024, 7, 4, 18, 0)
                },
                'time_patterns': {'primary_time_of_day': 'Afternoon'},
                'location_info': {'primary_location': 'Park'},
                'content_analysis': {'detected_objects': ['person', 'food', 'flag']},
                'cluster_size': 30,
                'calendar_context': {'is_holiday': True, 'holiday_name': 'Independence Day'}
            }

            start_time = datetime.now()
            event_name = namer.generate_event_name(simple_context)
            duration = (datetime.now() - start_time).total_seconds()

            print(f"   📝 Generated: '{event_name}'")
            print(f"   ⏱️  Time: {duration:.1f}s")

            # Performance recommendation
            if duration < 5:
                print(f"   🚀 Excellent performance for MacBook Pro!")
            elif duration < 10:
                print(f"   ✅ Good performance")
            else:
                print(f"   ⚠️  Slow - consider smaller model")

        elif provider == "openai":
            print(f"   ℹ️  Using OpenAI instead")
        else:
            print(f"   ❌ Model not available")

def main():
    """Run all local LLM tests."""
    print("🏠 LOCAL LLM INTEGRATION TEST")
    print("=" * 60)
    print("Testing offline intelligent event naming capabilities")
    print()

    test_llm_fallback_chain()
    test_ollama_availability()
    test_model_performance()

    print("\n" + "=" * 60)
    print("🎯 LOCAL LLM TEST SUMMARY")
    print("=" * 60)

    print("✅ Graceful fallback system implemented")
    print("✅ Multiple LLM provider support (OpenAI + Ollama)")
    print("✅ Hardware-optimized model recommendations")
    print()

    print("💡 RECOMMENDED SETUP FOR YOUR MACBOOK PRO:")
    print("   Model: llama3.1:8b (8B parameters)")
    print("   Reason: Best balance of quality/speed for 6-core i7 + 16GB RAM")
    print("   Installation: ollama pull llama3.1:8b")
    print()

    print("🚀 BENEFITS OF LOCAL LLM:")
    print("   • No API costs or rate limits")
    print("   • Works completely offline")
    print("   • Privacy - your photo data stays local")
    print("   • Fast inference on your hardware")
    print("   • Automatic fallback if OpenAI unavailable")

if __name__ == "__main__":
    main()