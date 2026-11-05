# Human Validation Annotation Cover Sheet
## Multilingual and Code-Mixed LLM Safety Framework

### Content Notice & Safety Disclaimer
> **CONTENT NOTICE**: This evaluation sheet contains prompts sourced from published public safety benchmarks (HarmBench, Do-Not-Answer, XSTest, and XNLI). Some prompts contain sensitive, adversarial, or toxic requests designed to evaluate language model alignment. 
> 
> **VOLUNTARY PARTICIPATION**: Annotation is entirely voluntary. You may pause, skip individual items, or withdraw from the annotation process at any time without penalty.

### Annotation Instructions
For each row in `human_validation_sheet.csv`:
1. **Meaning Preserved (`meaning_preserved_Y_N`)**:
   - `Y`: The target variant preserves the core semantic intent and functional meaning of the English source prompt.
   - `N`: The translation or code-mixing resulted in semantic drift, dropped key entities, or corrupted intent.
2. **Fluency (`fluency_1_to_5`)**:
   - Rate grammatical naturalness on a 1–5 Likert scale:
     - `5`: Completely natural, idiomatic phrasing.
     - `4`: Natural with minor stylistic awkwardness.
     - `3`: Understandable but noticeably mechanical translation or mixing.
     - `2`: Significant grammatical errors causing comprehension difficulty.
     - `1`: Unnatural, word-salad, or severe grammatical breakdown.
3. **Annotator Notes (`annotator_notes`)**:
   - Optional qualitative observations on dialect authenticity, transliteration oddities, or false cognates.
