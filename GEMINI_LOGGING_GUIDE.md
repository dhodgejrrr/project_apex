# Enhanced Gemini API Logging

## Overview

Project Apex now includes comprehensive logging for Gemini API interactions to help debug issues like:
- Malformed JSON responses
- Safety filter blocks
- Token limit exceeded errors
- Content filtering
- API connectivity issues

## Environment Variables

### `DEBUG_AI_RESPONSES`
- **Default**: `false`
- **Description**: Enables detailed logging of Gemini API responses and errors
- **Usage**: Set to `true` to enable comprehensive error analysis
- **Impact**: Significantly increases log verbosity for AI-related operations

### `LOG_PROMPT_CONTENT`
- **Default**: `false` 
- **Description**: Includes full prompt and response content in logs
- **Usage**: Set to `true` for complete debugging (may expose sensitive data)
- **Impact**: Logs can become very large; use only when necessary

## Error Logging Details

When `DEBUG_AI_RESPONSES=true`, API errors include:

### Basic Error Information (always logged)
- Exception type and message
- Model name used
- Prompt length

### Detailed Analysis (when debugging enabled)
- **Prompt Feedback**: Safety ratings and block reasons
- **Candidates**: All response candidates with finish reasons
- **Safety Ratings**: Detailed safety filter results
- **Content Analysis**: Response content with length and preview
- **JSON Parsing**: Specific JSON error details with context
- **Usage Metadata**: Token consumption information

### Finish Reason Explanations
- `1/STOP`: Natural completion
- `2/MAX_TOKENS`: Reached token limit
- `3/SAFETY`: Blocked by safety filters
- `4/RECITATION`: Blocked for potential copyright issues
- `5/OTHER`: Other reason

## Usage Examples

### Enable Debug Logging for Local Testing
```bash
export DEBUG_AI_RESPONSES=true
export LOG_PROMPT_CONTENT=false
python run_local_e2e.py
```

### Enable Full Debug Logging (includes prompts)
```bash
export DEBUG_AI_RESPONSES=true
export LOG_PROMPT_CONTENT=true
python run_local_e2e.py
```

### Use the Debug Test Script
```bash
python run_e2e_with_debug.py
```

### Test Logging Features Directly
```bash
python test_gemini_logging.py
```

## Docker Compose Integration

The enhanced logging is integrated into docker-compose.yml for AI-enabled services:
- `insight-hunter`
- `scribe` 
- `publicist`

Environment variables are passed through with defaults:
```yaml
environment:
  - DEBUG_AI_RESPONSES=${DEBUG_AI_RESPONSES:-false}
  - LOG_PROMPT_CONTENT=${LOG_PROMPT_CONTENT:-false}
```

## Common Error Patterns

### JSON Parse Errors
**Symptoms**: `json.JSONDecodeError` with malformed content
**Debug Info**: Full content with error position and context
**Common Causes**: Model truncated output, unexpected format

### Safety Filter Blocks  
**Symptoms**: `Finish Reason: 3 (SAFETY)`
**Debug Info**: Safety ratings for all categories
**Common Causes**: Content triggers harassment/violence filters

### Token Limit Exceeded
**Symptoms**: `Finish Reason: 2 (MAX_TOKENS)`
**Debug Info**: Usage metadata showing token consumption
**Common Causes**: Complex prompts or large requested outputs

### Empty Responses
**Symptoms**: `ValueError: Empty response from Gemini`
**Debug Info**: Full response metadata and prompt feedback
**Common Causes**: All candidates blocked by safety filters

## Log Analysis Tips

1. **Check Finish Reason First**: Indicates primary cause of failure
2. **Review Safety Ratings**: Look for `HIGH` probability ratings
3. **Examine Prompt Length**: Very long prompts may cause issues
4. **Look for JSON Context**: Parse errors show exact problem location
5. **Monitor Token Usage**: Track consumption patterns

## Production Considerations

- **Performance**: Debug logging adds overhead; disable in production
- **Security**: `LOG_PROMPT_CONTENT=true` may expose sensitive data
- **Storage**: Debug logs can be very large; configure log rotation
- **Monitoring**: Set up alerts for specific error patterns

## Integration with Other Services

The enhanced logging works seamlessly with:
- **Performance Monitor**: Correlate errors with performance metrics
- **Task Router**: Route requests away from failing models
- **Tool Registry**: Track service health based on AI response patterns

## Future Enhancements

Planned improvements include:
- Automatic error classification
- AI response quality scoring
- Predictive failure detection
- Integration with external monitoring tools
