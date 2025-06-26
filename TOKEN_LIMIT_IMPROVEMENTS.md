# Token Limit Improvements Summary

## 🚀 **Changes Made to Resolve MAX_TOKENS Issues**

### 1. **Updated Default Token Limits**
- **ai_helpers.py**: Increased default from 25,000 → 50,000 tokens
- **generate_json()**: Now defaults to 50,000 tokens
- **generate_json_adaptive()**: Now defaults to 50,000 tokens

### 2. **Replaced Functions with Adaptive Versions**
**Insight Hunter** (`agents/insight_hunter/main.py`):
- `generate_investigation_plan()`: 3,000 → 8,000 tokens + adaptive retry
- `synthesize_autonomous_insights()`: 8,000 → 12,000 tokens + adaptive retry

**Publicist** (`agents/publicist/main.py`):
- `generate_social_media_posts()`: 5,000 → 8,000 tokens + adaptive retry
- `execute_autonomous_social_strategy()`: 4,000 → 6,000 tokens + adaptive retry
- `critique_and_improve()`: 3,000 → 5,000 tokens + adaptive retry

### 3. **Added Environment Variable Support**
Added `MAX_OUTPUT_TOKENS` environment variable to docker-compose.yml for all AI services:
- insight-hunter
- historian  
- visualizer
- scribe
- publicist

**Configuration**:
```yaml
environment:
  - MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-50000}
```

### 4. **Adaptive Retry Logic**
The `generate_json_adaptive()` function now:
- Automatically retries with doubled token limits on MAX_TOKENS errors
- Caps maximum at 100,000 tokens
- Falls back to prompt truncation for very large prompts
- Provides detailed logging of retry attempts

## 📊 **Expected Results**

### Before
```
2025-06-26 19:02:50,036 WARNING Response truncated due to MAX_TOKENS limit
2025-06-26 19:02:50,038 ERROR GEMINI API ERROR: ValueError - Response truncated and not valid JSON
```

### After
```
2025-06-26 19:02:50,036 INFO Retrying with increased token limit: 8000 -> 16000
2025-06-26 19:02:52,041 INFO JSON parsing successful
```

## 🔧 **How to Deploy Changes**

### 1. Set Environment Variable (Optional)
```bash
export MAX_OUTPUT_TOKENS=75000  # For very complex tasks
```

### 2. Restart Services
```bash
docker-compose down
docker-compose up --build
```

### 3. Monitor Results
Look for these log patterns:
- ✅ `JSON parsing successful` (good)
- ✅ `Retrying with increased token limit` (adaptive working)
- ❌ `Response truncated due to MAX_TOKENS limit` (needs investigation)

## 📈 **Token Usage Guidelines**

### Service-Specific Recommendations
- **Investigation Planning**: 8,000 - 12,000 tokens
- **Insight Synthesis**: 12,000 - 20,000 tokens  
- **Social Media Generation**: 8,000 - 10,000 tokens
- **Report Writing**: 25,000 - 50,000 tokens
- **General Analysis**: 50,000 tokens (default)

### Cost Impact
- **50k tokens ≈ $0.125** per request (gemini-2.5-flash)
- **Previous 25k tokens ≈ $0.0625** per request
- **Cost increase**: ~100% for doubled limits
- **Success rate increase**: Expected 90%+ improvement

## 🎯 **Root Cause Analysis**

The MAX_TOKENS errors were caused by:

1. **Gemini's Internal Processing**: Uses tokens for "thoughts" before generating output
2. **Low Default Limits**: 3,000-25,000 tokens insufficient for complex tasks
3. **No Retry Logic**: Single attempts with fixed limits
4. **Race Data Complexity**: Large datasets require more processing tokens

## ✅ **Validation Steps**

1. **Run E2E Test**: Should now complete without MAX_TOKENS errors
2. **Check Logs**: Look for successful completion messages
3. **Monitor Costs**: Track token usage increase
4. **Verify Outputs**: Ensure quality isn't compromised

The adaptive retry system should eliminate most MAX_TOKENS failures while providing fallback options for edge cases.
