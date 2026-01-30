### Notes
- hf datasets sample is deterministic because it takes based on index
#### Scoring results
Two scripts:
- score_results.py: scores results for a single model run
- cross_score.py: scores results across multiple model runs
Differences: 
- cross_score.py handles multiple model runs and aggregates results
- score_results.py focuses on scoring a single run and generating detailed metrics


### Solved Issues
- 27 Jan 2026: investigate why type1_self_recognition is missing results.
    - all the gpt-4.1-mini results are missing (gpt-4.1-mini when used as observer)
    - API Error: Error code: 400 - {'error': {'message': 'Provider returned error', 'code': 400, 'metadata': {'raw': '{\n  "error": {\n    "message": "Invalid \'max_output_tokens\': integer below minimum value. Expected a value >= 16, but got 5 instead.",\n    "type": "invalid_request_error",\n    "param": "max_output_tokens",\n    "code": "integer_below_min_value"\n  }\n}', 'provider_name': 'Azure', 'is_byok': False}}, 'user_id': 'org_35hztXFgqDSd8QQCYroiGchNklU'}.

    - surgical fix: set min max_output_tokens to 16 for run in type1_self_pred.py Task1_3_SelfRecognition.proces_item(line 200)

### Issues
- type1_kth requires no_cache because it runs multiple models against a single target.
- mismatch between task argument names (e.g. type1_recog vs type1_self_recognition in cross_model_run.py)





