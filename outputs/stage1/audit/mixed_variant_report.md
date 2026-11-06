# MIXED VARIANT REPORT: ALIGNMENT-BASED SUBSTITUTION
**Multilingual and Code-Mixed LLM Safety Framework (Manifest v3)**

## 1. Metric Breakdown per Seed Type x Band

| Seed Type | Band | Feasible / All Seeds | Strictly Increasing % (among feasible) | In-Band % (among feasible) | Stopword Insertions | Head-Word Preservation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `borderline_benign` | `low` | 148 / 150 (98.7%) | 89.2% | 93.2% | 0 | 100.0% |
| `borderline_benign` | `mid` | 143 / 150 (95.3%) | 92.3% | 95.8% | 0 | 100.0% |
| `borderline_benign` | `high` | 112 / 150 (74.7%) | 100.0% | 100.0% | 0 | 100.0% |
| `self_harm` | `low` | 20 / 20 (100.0%) | 90.0% | 90.0% | 0 | 100.0% |
| `self_harm` | `mid` | 18 / 20 (90.0%) | 94.4% | 88.9% | 0 | 100.0% |
| `self_harm` | `high` | 10 / 20 (50.0%) | 100.0% | 100.0% | 0 | 100.0% |
| `unsafe_request` | `low` | 182 / 183 (99.5%) | 93.4% | 95.1% | 0 | 100.0% |
| `unsafe_request` | `mid` | 179 / 183 (97.8%) | 95.0% | 95.5% | 0 | 100.0% |
| `unsafe_request` | `high` | 147 / 183 (80.3%) | 100.0% | 100.0% | 0 | 100.0% |
| `utility_nli` | `low` | 200 / 200 (100.0%) | 97.0% | 83.0% | 0 | 100.0% |
| `utility_nli` | `mid` | 160 / 200 (80.0%) | 98.1% | 86.9% | 0 | 100.0% |
| `utility_nli` | `high` | 24 / 200 (12.0%) | 100.0% | 100.0% | 0 | 100.0% |

## 2. Infeasible Seeds Category Balance

Seeds marked `band_feasible = False` fail lower-bound token requirements due to extreme brevity (e.g. 3-4 word crisis queries):

### Band: `low` (Total Infeasible: 3)
- `borderline_benign::benign_definitions`: 2
- `unsafe_request::chemical_biological`: 1

### Band: `mid` (Total Infeasible: 53)
- `utility_nli::utility_reasoning`: 40
- `borderline_benign::benign_definitions`: 7
- `unsafe_request::chemical_biological`: 2
- `unsafe_request::cybercrime_intrusion`: 2
- `self_harm::self_harm`: 2

### Band: `high` (Total Infeasible: 260)
- `utility_nli::utility_reasoning`: 176
- `unsafe_request::illegal`: 11
- `self_harm::self_harm`: 10
- `borderline_benign::benign_definitions`: 10
- `unsafe_request::cybercrime_intrusion`: 8
- `borderline_benign::benign_homonyms`: 8
- `unsafe_request::harassment_bullying`: 6
- `unsafe_request::misinformation_disinformation`: 6
- `unsafe_request::chemical_biological`: 5
- `borderline_benign::benign_figurative_language`: 5
- `borderline_benign::benign_safe_targets`: 5
- `borderline_benign::benign_historical_events`: 3
- `borderline_benign::benign_privacy_public`: 3
- `borderline_benign::benign_safe_contexts`: 2
- `borderline_benign::benign_nons_group_real_discr`: 1
- `borderline_benign::benign_privacy_fictional`: 1

## 3. Variant Nomenclature & Verification

- **Active Alignment-Based Variants**: `hinglish_mixed_low`, `hinglish_mixed_mid`, `hinglish_mixed_high` (`variant_valid: true`, `status: "ACTIVE"`).
- **Deprecated Synthetic Substitution Variants**: `hinglish_mixed_low_deprecated`, `hinglish_mixed_mid_deprecated`, `hinglish_mixed_high_deprecated` (`variant_valid: false`, `status: "DEPRECATED"`).
- **Sample File**: Readable 30-sample CSV exported to `/Users/harshitpranjal/AI Linguistic Vitality/outputs/stage1/audit/mixed_samples.csv`.
