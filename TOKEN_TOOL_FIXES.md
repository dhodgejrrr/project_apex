# Token Limit & Tool Discovery Fixes Summary

## 🎯 **Problems Resolved**

### 1. **Token Limit Issues**
- Investigation plan generation was using only 3000 tokens → increased to 8000 with adaptive retry
- Default token limits were 25000 → increased to 50000 
- Several specific functions used low limits → updated to use adaptive functions

### 2. **Tool Discovery Issues**  
- ToolCaller was using POST requests for `/tools/capabilities` endpoint
- Core-analyzer capabilities endpoint only accepts GET requests → fixed HTTP method mismatch

## 🔧 **Changes Made**

### Token Limit Improvements

#### Updated `agents/common/ai_helpers.py`:
```python
# Before: generate_json(prompt, temperature=0.7, max_output_tokens=25000)
# After:  generate_json(prompt, temperature=0.7, max_output_tokens=50000)

# Before: generate_json_adaptive(prompt, temperature=0.7, max_output_tokens=25000)  
# After:  generate_json_adaptive(prompt, temperature=0.7, max_output_tokens=50000)
```

#### Updated `agents/insight_hunter/main.py`:
```python
# Investigation plan generation:
# Before: ai_helpers.generate_json(prompt, temperature=0.7, max_output_tokens=3000)
# After:  ai_helpers.generate_json_adaptive(prompt, temperature=0.7, max_output_tokens=8000)

# Synthesis generation:
# Before: ai_helpers.generate_json(prompt, temperature=0.6, max_output_tokens=8000)
# After:  ai_helpers.generate_json_adaptive(prompt, temperature=0.6, max_output_tokens=12000)
```

#### Updated `agents/publicist/main.py`:
```python
# Tweet generation:
# Before: ai_helpers.generate_json(prompt, temperature=0.8, max_output_tokens=5000)
# After:  ai_helpers.generate_json_adaptive(prompt, temperature=0.8, max_output_tokens=8000)
```

#### Updated `docker-compose.yml`:
Added `MAX_OUTPUT_TOKENS=${MAX_OUTPUT_TOKENS:-50000}` environment variable to all AI-using services:
- insight-hunter
- historian  
- visualizer
- scribe
- publicist

### Tool Discovery Improvements

#### Updated `agents/common/tool_caller.py`:
```python
# Before: Used invoke_cloud_run() (POST) for capabilities discovery
# After:  Use requests.get() directly for capabilities endpoint

def discover_capabilities(self, agent_name: str) -> Dict[str, Any]:
    # Now uses GET request for /tools/capabilities endpoint
    if os.getenv("PUBSUB_EMULATOR_HOST"):
        response = requests.get(capabilities_url, timeout=300)
        response.raise_for_status()
        return response.json()
```

#### Updated `docker-compose.yml`:
Added comprehensive service discovery environment variables:
```yaml
- TOOL_REGISTRY_URL=http://tool-registry:8080
- PERFORMANCE_MONITOR_URL=http://performance-monitor:8080  
- TASK_ROUTER_URL=http://task-router:8080
- CORE_ANALYZER_URL=http://core-analyzer:8080
- VISUALIZER_URL=http://visualizer:8080
- HISTORIAN_URL=http://historian:8080
- PUBLICIST_URL=http://publicist:8080
- SCRIBE_URL=http://scribe:8080
- ARBITER_URL=http://arbiter:8080
```

Added Tool Registry, Performance Monitor, and Task Router services to docker-compose.

## 📊 **Expected Results**

### Token Limit Issues:
- ✅ Investigation plan generation should complete successfully
- ✅ Complex AI responses should no longer hit MAX_TOKENS limits
- ✅ Adaptive retry logic should handle remaining edge cases
- ✅ Default 50000 token limit provides more headroom

### Tool Discovery Issues:
- ✅ ToolCaller can successfully discover core-analyzer capabilities
- ✅ stint_analysis, sector_analysis, and other tools should be found and executable  
- ✅ Investigation workflow should complete successfully
- ✅ Autonomous insights generation should work end-to-end

## 🧪 **Testing Validation**

Expected log pattern after fixes:
```
INFO Generated investigation plan with 4 tasks
INFO Step 2: Executing investigation plan...
INFO Invoking http://core-analyzer:8080/tools/capabilities via GET (local)...
INFO Successfully discovered 4 tools for core_analyzer
INFO Executing tool: stint_analysis with params: {'run_id': '...', 'car_number': '10'}
INFO Tool execution successful: stint_analysis
INFO Step 3: Synthesizing final insights...
INFO Synthesized 8 autonomous insights
```

## 🚀 **Production Deployment**

### Environment Variables:
```bash
export MAX_OUTPUT_TOKENS=50000
export DEBUG_AI_RESPONSES=true  # For monitoring
export LOG_PROMPT_CONTENT=false # Security
```

### Docker Compose Command:
```bash
MAX_OUTPUT_TOKENS=50000 DEBUG_AI_RESPONSES=true docker-compose up
```

The system should now complete the full autonomous insight generation workflow without token limit or tool discovery errors!
