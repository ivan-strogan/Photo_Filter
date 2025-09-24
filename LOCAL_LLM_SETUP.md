# Local LLM Setup Guide

Your Photo Filter app now supports **local LLM** for intelligent event naming without needing OpenAI API keys! This guide shows you how to set it up on your MacBook Pro.

## 🎯 Why Local LLM?

- **No API costs** - Zero usage fees
- **Complete privacy** - Your photos never leave your machine
- **Works offline** - No internet required for AI naming
- **No rate limits** - Process as many photos as you want
- **Automatic fallback** - Works when OpenAI is unavailable

## 🖥️ Perfect for Your MacBook Pro

Your specs (6-Core Intel i7, 16GB RAM) are ideal for running local LLMs efficiently!

## 📥 Installation Steps

### 1. Install Ollama

Download and install Ollama (the easiest way to run local LLMs):
```bash
# Install via Homebrew (recommended):
brew install ollama

# Or visit https://ollama.ai/download and download for macOS
```

### 2. Start Ollama Server

**IMPORTANT:** Start the server before pulling models!

**Option A: Manual start (in Terminal 1)**
```bash
ollama serve
```
Keep this terminal window open - the server needs to stay running.

**Option B: Background service (recommended)**
```bash
# Start as macOS background service
brew services start ollama

# Server will now start automatically on boot
```

### 3. Pull a Recommended Model

**In a new terminal window** (or after starting background service):

For your MacBook Pro specs, we recommend starting with **Llama 3.2 3B**:
```bash
ollama pull llama3.2:3b
```

**Model Options (choose one to start):**

```bash
# RECOMMENDED: Best balance for MacBook Pro (start here)
ollama pull llama3.2:3b     # 3B params, ~2GB, 3-5s response

# UPGRADE: Higher quality, slower (try this next)
ollama pull llama3.1:8b     # 8B params, ~4.7GB, 8-12s response

# LIGHTWEIGHT: Fastest option (if 3B is too slow)
ollama pull phi3:mini       # 3.8B params, ~2.3GB, 2-3s response
```

**💡 Recommendation**: Start with `llama3.2:3b`, then upgrade to `llama3.1:8b` if you want higher quality event names.

### 4. Test Your Setup

Verify everything works:
```bash
# List installed models
ollama list

# Test a simple query
ollama run llama3.2:3b "Say hello"

# Exit the chat with: /bye
```

## 🚀 How It Works

The app automatically tries providers in this order:

1. **OpenAI** (if API key available)
2. **Ollama** (if running locally) ⭐
3. **Template-based naming** (fallback)

## 🧪 Testing Your Setup

Run the local LLM test:
```bash
python3 tests/test_local_llm.py
```

This will:
- Check if Ollama is running
- List available models
- Test event name generation
- Show performance metrics

## ⚡ Performance Expectations

On your MacBook Pro:

| Model | Speed | Quality | Memory | Use Case |
|-------|--------|---------|---------|----------|
| `phi3:mini` | ~2-3s | Good | ~2GB | Fastest option |
| `llama3.2:3b` | ~3-5s | Excellent | ~3GB | **Recommended start** |
| `llama3.1:8b` | ~8-12s | Superior | ~5GB | **Quality upgrade** |

**Recommended Path:**
1. Start with `llama3.2:3b` for excellent quality and speed
2. Upgrade to `llama3.1:8b` if you want the best possible event names

## 🎨 Example Event Names

With local LLM, you'll get intelligent names like:

**Instead of:** `2024_10_31 - Evening Event`
**You get:** `2024_10_31 - Halloween Costume Party`

**Instead of:** `2024_07_04 - Afternoon Activity`
**You get:** `2024_07_04 - Independence Day BBQ`

## 🔧 Configuration

You can configure the local LLM in your code:

```python
from src.event_namer import EventNamer

# Use specific model
namer = EventNamer(
    enable_llm=True,
    ollama_model="llama3.2:3b",
    ollama_url="http://localhost:11434"
)
```

## 🔄 Automatic Fallbacks

The system gracefully handles:
- Ollama not running → Falls back to templates
- Model not available → Tries other providers
- Network issues → Uses offline naming
- Long response times → Timeout and fallback

## 📊 Usage in Photo Filter

Once set up, your photo organization will automatically use local LLM:

```bash
# With virtual environment and dependencies:
python3 main.py process --source "Sample_Photos/iPhone Automatic"
```

The app will:
1. Try OpenAI (if API key set)
2. **Use Ollama for intelligent naming** ⭐
3. Fall back to templates if needed

## 🛠️ Troubleshooting

### Ollama not starting?
```bash
# Check if it's already running
ps aux | grep ollama

# Kill existing process if needed
killall ollama

# Start fresh
ollama serve
```

### Model too slow?
Try the faster `phi3:mini` model:
```bash
ollama pull phi3:mini
```

### Out of memory?
Your 16GB RAM is plenty, but if you have other heavy apps running:
- Close unused applications
- Use smaller models (`phi3:mini`)

## 💡 Tips for Best Results

1. **Keep Ollama running** - Start it when you boot your Mac
2. **Use SSD space** - Models are 2-6GB each
3. **Monitor performance** - Check Activity Monitor if slow
4. **Update models** - `ollama pull` to get latest versions

## 🎉 Benefits Recap

✅ **Cost**: Zero API fees
✅ **Privacy**: All processing stays local
✅ **Speed**: Optimized for your hardware
✅ **Quality**: Same intelligent naming as OpenAI
✅ **Reliability**: No internet dependency

Your Photo Filter now has true AI intelligence that runs entirely on your MacBook Pro! 🚀