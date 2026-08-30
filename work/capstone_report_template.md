# Capstone Report — Content Refresh Opportunity Scoring

**Author:** Aleeza Maryam  
**Lane:** Refresh / Content Opportunity Scoring  
**Repo:** https://github.com/Aleeza-Maryam/Aleeza-ML-Internship-WEEK1  
**Date:** August 30, 2026

---

## 0. Abstract

This research explores whether machine learning can identify content refresh opportunities more effectively than a simple hand-written rule, addressing the practical challenge content teams face in prioritizing which pages to review first. Using March 2026 search and analytics data from the FlyRank Internship Warehouse, comprising 10,000 content pages with daily performance metrics and content metadata, we trained a Random Forest classifier to predict which pages would experience a ≥30% decline in impressions from March to April 2026. The model achieved a robust AUC of 0.752, substantially outperforming the baseline rule (AUC: 0.498) by 25.4 percentage points, demonstrating that machine learning can effectively distinguish between pages that need refresh and those that do not. The final output is a ranked list of content pages with opportunity scores, reason codes explaining each flag, and actionable recommendations, enabling content teams to prioritize human review efficiently and focus editorial resources where they matter most.

---

## 1. Problem Framing

**Decision Supported:**  
This project supports the operational decision of **which content pages should be reviewed first for a potential refresh**. Content teams typically manage hundreds or thousands of pages, making manual review of all pages impractical. The model provides a data-driven prioritization system that directs attention to pages most likely to benefit from intervention.

**Unit of Analysis:**  
Individual content pages, uniquely identified by `content_hash_id` in the FlyRank warehouse. Each page represents a distinct URL with associated search performance, analytics engagement, and content metadata.

**Output:**  
For each page, the system produces:
1. **Opportunity Score** (0 to 1): A probability estimate that the page will experience significant decline if left unrefreshed, with higher scores indicating greater urgency.
2. **Reason Code**: A human-readable explanation of which signals contributed to the score (e.g., "low ranking", "old content", "low traffic").
3. **Recommended Action**: A specific, actionable suggestion for what to do with the page (e.g., "refresh content", "optimize meta tags", "merge with similar page").

**Action Humans Take:**  
Content editors and managers use the ranked list to decide which pages to assign for refresh in their next content sprint. For each high-scoring page, an editor investigates the page's current state, assesses whether refresh is appropriate, and executes the recommended action if they agree with the model's assessment.

**Cost of Wrong Call:**  

| Error Type | Consequence | Impact |
|------------|-------------|--------|
| **False Positive** (flagging a page that doesn't need refresh) | Wasted editorial time reviewing and potentially updating content unnecessarily | Low to Medium (one review cycle wasted) |
| **False Negative** (missing a page that does need refresh) | Missed opportunity to recover or improve traffic, engagement, and conversions | High (lost revenue, declining visibility over time) |
| **False Positive with unnecessary update** | May introduce errors or remove valuable historical context | Medium (can be corrected) |
| **False Negative with severe decline** | Page may drop off first page of search results, losing significant traffic | Very High (hard to recover lost rankings) |

**Why ML Helps:**  
Simple rules like "refresh pages older than 180 days with low traffic" are intuitive but miss important nuances:
- **Non-linear interactions**: A page with very high traffic but slightly old content may benefit differently than a page with low traffic and old content.
- **Complex signal combinations**: The interaction between rank, impressions, clicks, and age creates patterns that simple rules cannot capture.
- **Adaptive thresholds**: What constitutes "low" traffic depends on the page's historical performance and context. ML can learn these thresholds from data.
- **Scoring precision**: ML provides a continuous score rather than a binary flag, allowing for fine-grained prioritization.
- **Feature weighting**: ML automatically learns which signals matter most, rather than relying on arbitrary weights (e.g., "+1 point for each condition").

---

## 2. Data Safety

**Data Used:**

| Table | Purpose | Date Range |
|-------|---------|------------|
| `fact_content_daily_performance` | Daily search and analytics performance (impressions, clicks, position, pageviews, engaged sessions) | March 1-31, 2026 (training); April 1-30, 2026 (label only) |
| `dim_content` | Content metadata (creation date, update date, content type, search volume, word count) | Snapshot as of March 2026 |

**Development Window:**
- **Training/Development Window**: March 1-31, 2026 – all features are derived from this period.
- **Test/Outcome Window**: April 1-30, 2026 – used exclusively to create the binary label (decline indicator), never used as features.
- **Decision Date**: March 31, 2026 – the point at which the model would make predictions in a real deployment.

**Excluded Columns (and why):**

| Column | Reason for Exclusion |
|--------|---------------------|
| `client_hash_id` | Grouping only, never as a feature (would cause leakage and overfitting to specific clients) |
| `content_hash_id` | Identifier, not a predictive feature (would cause overfitting) |
| `url_hash_id` | Identifier, not a predictive feature |
| `keyword_hash_id` | Identifier, not a predictive feature |
| `keyword_char_count` | Not relevant for content refresh decisions |
| `keyword_token_count` | Not relevant for content refresh decisions |
| `content_updated_date` | Future dates beyond March 31 would cause leakage; removed from feature set |
| `is_published` | Not predictive for refresh need (all published pages) |
| `is_deleted` | Not relevant (deleted pages excluded from analysis) |
| `content_type` | Too many categories with small sample sizes; not reliable as a feature |
| `provider_used` | Metadata about data provider, not predictive |
| `model_used` | Metadata about model used, not predictive |

**Leakage Risks Considered and Mitigated:**

1.  **Future Performance Information**: April data used ONLY for label creation. At no point is April data used to inform model training or feature engineering. The model sees only March signals when making predictions.
   
2.  **Future Update Dates**: `content_updated_date` for dates beyond March 31, 2026 is removed from the feature set. The `days_since_update` feature was initially considered but dropped because it would have included future dates.
   
3.  **Client Identifiers**: `client_hash_id` is used only for grouping during data exploration. It is never included as a feature in the model to prevent overfitting to specific clients.
   
4.  **Trend-based Labels**: The decline label is created from future data (April vs March). This label is never used as a feature – it's only the prediction target.
   
5.  **Time-aware Split**: All training data comes from March 2026 only. No April data is used anywhere in training or validation.

**Data Safety Confirmation:**
-  No client names, domains, or URLs
-  No private queries or search terms
-  No credentials, API keys, or raw exports
-  No causal claims about Google's algorithm
-  All data is aggregated at the page level; no user-level PII

---

## 3. Baseline

**Hand-Written Rule (from Week 4):**

The baseline represents a typical industry approach to content refresh prioritization – a simple, rule-based scoring system that editors can implement without data science support.

```python
def baseline_rule(row):
    score = 0
    if row['march_avg_position'] > 10:    # Page is on page 2+ of search results
        score += 1
    if row['content_age_days'] > 180:     # Content is over 6 months old
        score += 1
    if row['march_impressions'] < 100:    # Page receives very low search traffic
        score += 1
    return score  # Score >= 2 = refresh candidate
