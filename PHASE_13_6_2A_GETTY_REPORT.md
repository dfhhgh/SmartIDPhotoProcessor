# Phase 13.6.2a — Getty Images Source Feasibility Report

## 1. Objective

Evaluate Getty Images as a potential additional source for the celebrity identity dataset. Determine whether Getty Images is technically, legally, and practically suitable as an additional source for acquiring high-quality identity-labeled celebrity photographs. Execute a small 6-person pilot only if feasibility is confirmed.

## 2. Official Getty API Feasibility

### API Version and Base URI

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

- API version: v3
- Base URI: `https://api.gettyimages.com/v3/`
- Swagger documentation: `https://api.gettyimages.com/swagger`

### Search Endpoints

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

- Creative images: `GET /v3/search/images/creative?phrase=<query>`
- Editorial images: `GET /v3/search/images/editorial?phrase=<query>`
- Pagination: `page` (default 1) and `page_size` (valid: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 25, 30, 50, 60, 75, 100)
- Display sizes: `fields=display_sizes` returns thumbnail/comp URIs
- Download sizes: `fields=downloads` returns download URIs (requires access token)

### Response Format

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

```json
{
  "result_count": 867845,
  "images": [
    {
      "id": "1199241887",
      "asset_family": "creative",
      "caption": "description text",
      "license_model": "royaltyfree",
      "max_dimensions": {"height": 4480, "width": 6720},
      "display_sizes": [{"name": "thumb", "uri": "https://..."}],
      "title": "Image title"
    }
  ]
}
```

## 3. Authentication Requirements

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

### API Key

- Required for all requests
- Sent via `Api-Key` HTTP header
- Only available to Getty Images customers with a paid subscription
- Must contact a Getty Images sales representative to obtain

### OAuth2 Access Token

- Optional for search (enhanced privileges)
- Required for download operations
- Obtained via client credentials grant: `POST https://authentication.gettyimages.com/oauth2/token`
- Parameters: `client_id=<API_KEY>&client_secret=<API_SECRET>&grant_type=client_credentials`
- Token lifetime: 1800 seconds (30 minutes)
- Sent via `Authorization: Bearer <ACCESS_TOKEN>` header

### Environment Variables Required

```
GETTY_API_KEY=<your-api-key>
GETTY_API_SECRET=<your-api-secret>
```

## 4. Licensing Findings

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

### Getty Images Content License Agreement (Last Updated: April 2026)

The Getty Images Content License Agreement explicitly addresses machine learning and AI usage:

#### Section 3.11 — No Machine Learning, AI, or Biometric Technology Use

> Unless explicitly authorized in a Getty Images invoice, sales order confirmation or license agreement, you may not use content (including any caption information, keywords or other metadata associated with content) for any machine learning and/or artificial intelligence purposes, or for any technologies designed or intended for the identification of natural persons.

Key restrictions:

1. **No ML training**: Content may not be used to train, fine-tune, optimize, evaluate, or create ML/AI models
2. **No biometric identification**: Content may not be used for technologies designed to identify natural persons
3. **No metadata exploitation**: Caption information, keywords, and metadata cannot be used separately from content
4. **No standalone file redistribution**: Content cannot be redistributed as standalone files
5. **Editorial content**: Cannot be used for commercial purposes without additional license
6. **Attribution required**: Editorial use requires photo credit

#### Exceptions (Limited)

- AI may be used for internal indexing/searching/sorting of creative (non-editorial) content
- AI may be used for permitted editing of licensed creative content
- These exceptions explicitly do **not** allow training, fine-tuning, or data ingestion

#### Data Licensing Pathway

Getty Images offers a separate **Data Licensing** product specifically for ML training:

- Custom datasets purpose-built for training
- Rights-cleared content with transparent data practices
- Requires contacting Getty Images sales team
- Enterprise-level licensing with negotiated terms
- Separate from standard content licensing

## 5. Dataset/ML Usage Findings

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

### Our Intended Use Case

Our project requires:
1. Downloading celebrity photographs
2. Storing them locally as a dataset
3. Generating face embeddings (InsightFace ArcFace)
4. Building a FAISS similarity index
5. Using the index for reverse face search
6. Potentially fine-tuning face recognition models

### License Analysis

Our use case **directly conflicts** with Section 3.11 of the Getty Images Content License Agreement:

| Our Use | License Restriction | Permitted? |
|---------|-------------------|------------|
| Download photos | Requires paid license | Only with purchase |
| Store locally | Requires license | Only with purchase |
| Generate face embeddings | "Technologies designed for identification of natural persons" | **NO** |
| Build similarity index | ML/AI purpose | **NO** |
| Train face models | "Training" definition | **NO** |
| Use in production pipeline | Commercial use of editorial content | **NO** (editorial) |

### Getty's Official Position

Getty Images explicitly states:

> "If you have any content training needs, please reach out to your Getty Images' representative."

This indicates they have a dedicated enterprise licensing pathway for ML use cases, but it requires direct negotiation and separate licensing.

## 6. Technical Source Architecture

**CONFIRMED FROM CODE**

### Existing ImageSource Abstraction

```python
class ImageSource(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def search(self, query: str, max_results: int) -> Iterator[SearchResult]: ...

    @abstractmethod
    def download_url(self, url: str) -> bytes | None: ...

    def close(self) -> None: ...
```

### GettySource Implementation

Created `dataset_acquisition/sources/getty.py` implementing the ImageSource interface:

- **name**: `"getty_images"`
- **search()**: Hits `/v3/search/images/creative`, paginates, yields SearchResult objects
- **download_url()**: Downloads from Getty display URIs with retry logic
- **Authentication**: OAuth2 client credentials grant
- **Rate limiting**: Configurable delay + exponential backoff on 429

### Integration Points

- Uses existing `Downloader` class (no duplication)
- Uses existing Single-Face Gate (InsightFace)
- Uses existing SHA-256 deduplication
- Uses existing rejection telemetry
- Uses existing splitter for reference/query division
- Compatible with existing manifest/report generation

## 7. Search Capabilities

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

### Creative Search

- `GET /v3/search/images/creative?phrase=<query>`
- Returns royalty-free and rights-managed creative content
- Supports phrase-based search
- No person-specific filter (must use name in query)

### Editorial Search

- `GET /v3/search/images/editorial?phrase=<query>`
- Returns editorial/news/sports content
- Better for celebrity/event photography
- Requires additional license for commercial use

### Available Filters (from API docs)

- `phrase`: Search text
- `page`: Page number (default 1)
- `page_size`: Results per page (1-100)
- `fields`: Response fields (`display_sizes`, `downloads`, etc.)
- `sort_order`: Result ordering

### Missing Filters for Our Use Case

- **No person-specific filter**: Must search by name in phrase
- **No face-count filter**: Cannot pre-filter single-person images
- **No headshot/portrait filter**: Must rely on query phrasing
- **No orientation filter**: Cannot filter landscape/portrait
- **No number-of-people filter**: Cannot exclude group photos

## 8. Filters Used

**NOT APPLICABLE — PILOT BLOCKED**

No filters were applied because the pilot was blocked by licensing restrictions.

Had the pilot proceeded, the following search queries would have been used:

### Actors (Tom Hanks, Scarlett Johansson, Denzel Washington)

- `"<name> portrait"`
- `"<name> headshot"`
- `"<name> interview"`
- `"<name> red carpet"`
- `"<name> premiere"`

### Football Players (Lionel Messi, Cristiano Ronaldo, Kylian Mbappé)

- `"<name> portrait"`
- `"<name> match action"`
- `"<name> interview"`
- `"<name> press conference"`
- `"<name> celebration"`

## 9. Pilot Identities

**NOT EXECUTED — BLOCKED BY LICENSING**

Planned identities:

| Identity | Category | Getty Coverage Expected |
|----------|----------|------------------------|
| Tom Hanks | Actor | High (decades of coverage) |
| Scarlett Johansson | Actor | High (Marvel, red carpet) |
| Denzel Washington | Actor | High (decades of coverage) |
| Lionel Messi | Football | High (global coverage) |
| Cristiano Ronaldo | Football | High (global coverage) |
| Kylian Mbappé | Football | High (recent years) |

## 10. Candidate Counts

**NOT APPLICABLE — PILOT BLOCKED**

No candidates were searched because the pilot was blocked by licensing restrictions.

## 11. Accepted Counts

**NOT APPLICABLE — PILOT BLOCKED**

No images were accepted because the pilot was blocked by licensing restrictions.

## 12. Rejection Counts

**NOT APPLICABLE — PILOT BLOCKED**

No images were rejected because the pilot was blocked by licensing restrictions.

## 13. Rejection Reasons

**NOT APPLICABLE — PILOT BLOCKED**

No rejection telemetry was collected because the pilot was blocked by licensing restrictions.

## 14. Per-Person Statistics

**NOT APPLICABLE — PILOT BLOCKED**

No per-person statistics were collected because the pilot was blocked by licensing restrictions.

## 15. API Errors

**NOT APPLICABLE — PILOT BLOCKED**

No API calls were made because the pilot was blocked by licensing restrictions.

Expected API error behaviors (from documentation):

| HTTP Code | Meaning | Handling |
|-----------|---------|----------|
| 401 | Unauthorized (bad/missing API key) | Stop, report GETTY_API_ACCESS_UNAVAILABLE |
| 403 | Forbidden (no permission) | Stop, report GETTY_API_ACCESS_UNAVAILABLE |
| 429 | Rate limited | Retry with exponential backoff |
| 500 | Server error | Retry after brief delay |

## 16. Rate Limits

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

- Rate limits are per-API-key, configured at account setup
- Default: queries per second (QPS) varies by account
- 429 response includes `X-Error-Detail: Account Over Queries Per Second Limit`
- Rate limits can be adjusted by Getty support on request
- Download URLs expire after ~24 hours

## 17. Runtime

**NOT APPLICABLE — PILOT BLOCKED**

No runtime was measured because the pilot was blocked by licensing restrictions.

Expected runtime characteristics (from API docs):

- Search latency: ~100-500ms per request
- Download latency: ~1-5s per image (depends on size)
- Rate limit delay: Account-configured QPS
- Total estimated for 6-person pilot: ~10-30 minutes (if licensed)

## 18. Accepted Images/Minute

**NOT APPLICABLE — PILOT BLOCKED**

No images were collected because the pilot was blocked by licensing restrictions.

## 19. Dataset Quality

**NOT APPLICABLE — PILOT BLOCKED**

No dataset was created because the pilot was blocked by licensing restrictions.

Expected quality characteristics (inference from Getty's content library):

- **Positive**: High-quality professional photography
- **Positive**: Extensive metadata (titles, captions, keywords)
- **Positive**: Diverse poses and contexts (editorial coverage)
- **Positive**: High resolution (up to 6720x4480)
- **Negative**: Group photos likely common (events, red carpet)
- **Negative**: Watermarked comp images (full resolution requires purchase)
- **Negative**: Editorial content requires separate license for commercial use

## 20. Visual Review Observations

**NOT APPLICABLE — PILOT BLOCKED**

No visual review was performed because the pilot was blocked by licensing restrictions.

## 21. Deduplication

**NOT APPLICABLE — PILOT BLOCKED**

No deduplication analysis was performed because the pilot was blocked by licensing restrictions.

Expected deduplication behavior:

- SHA-256 dedup would catch exact duplicates within Getty results
- Cross-person dedup would check against existing Wikimedia datasets
- Getty editorial coverage likely has many similar/same-event images

## 22. Retrieval Smoke Test

**NOT APPLICABLE — PILOT BLOCKED**

No retrieval test was performed because the pilot was blocked by licensing restrictions.

## 23. Comparison with Wikimedia

**INFERENCE (no Getty data collected)**

| Metric | Wikimedia | Getty (Expected) |
|--------|-----------|------------------|
| API access | Free, no key required | Paid subscription required |
| Licensing | CC-BY-SA / Public Domain | Strict commercial license |
| ML/AI training | Permitted (with attribution) | **BLOCKED** (Section 3.11) |
| Person search | Via category system | Phrase-based only |
| Face count filter | Not available | Not available |
| Image quality | Variable (amateur to pro) | High (professional) |
| Metadata quality | Variable | High (structured) |
| Watermarks | None | Comp images watermarked |
| Download cost | Free | Per-image or subscription |
| Rate limits | Configurable | Account-configured QPS |

### Key Differences

1. **Licensing**: Wikimedia permits ML training; Getty explicitly prohibits it
2. **Cost**: Wikimedia is free; Getty requires paid subscription + per-image fees
3. **Quality**: Getty has consistently higher professional quality
4. **Metadata**: Getty has richer, more structured metadata
5. **Access**: Wikimedia has lower barrier to entry

## 24. Limitations

### Licensing Limitations (BLOCKING)

- **Section 3.11 prohibition**: ML/AI use explicitly prohibited without authorization
- **No free trial for API**: API access requires paid subscription
- **Editorial license restrictions**: Commercial use requires additional license
- **No standalone redistribution**: Cannot redistribute downloaded images

### Technical Limitations (if licensing were resolved)

- **No person-specific filter**: Must rely on text search phrases
- **No face-count filter**: Cannot pre-filter single-person images
- **No headshot/portrait filter**: Must rely on query phrasing
- **Comp images watermarked**: Full resolution requires purchase
- **Download URLs expire**: Must download within ~24 hours

### Access Limitations

- **API key required**: Must contact Getty sales representative
- **Paid subscription required**: No free API access available
- **Account setup required**: Rate limits configured at account creation

## 25. Recommendation

### Primary Finding: BLOCKED BY LICENSING

**GETTY_PILOT_BLOCKED_BY_LICENSING**

The Getty Images Content License Agreement (Section 3.11) explicitly prohibits using content for:

> "any machine learning and/or artificial intelligence purposes, or for any technologies designed or intended for the identification of natural persons"

Our project's core use case — building a face recognition dataset for reverse search — falls directly under this prohibition. The license also explicitly states that "training" (defined as using content to develop, build, improve, fine-tune, optimize, evaluate or otherwise create or enhance any ML/AI technology) is prohibited.

### Secondary Finding: API Access Unavailable

Even if licensing were resolved, API access requires:

- Paid subscription to Getty Images
- API key from Getty sales representative
- OAuth2 client credentials for download operations

We do not have these credentials and cannot obtain them without a commercial relationship with Getty Images.

### Alternative Pathway: Data Licensing

Getty Images offers a **Data Licensing** product specifically for ML training:

- Custom datasets purpose-built for training
- Rights-cleared content with transparent data practices
- Enterprise-level licensing with negotiated terms
- Requires contacting Getty Images sales team

This pathway would require:
1. Establishing a commercial relationship with Getty Images
2. Negotiating a data licensing agreement
3. Paying enterprise-level licensing fees
4. Potentially receiving a custom dataset (not API access)

### Recommendation

**Do not implement GettySource for production use.** The licensing restrictions are clear and unambiguous. Instead:

1. **Continue with Wikimedia Commons** as the primary free source
2. **Consider Getty Data Licensing** if budget allows enterprise-level access
3. **Keep the GettySource implementation** for potential future use if licensing is resolved
4. **Focus on improving Wikimedia yield** through better query strategies

### Implementation Status

Despite the licensing block, the GettySource implementation is complete and tested:

- `dataset_acquisition/sources/getty.py` — Full ImageSource implementation
- `tests/test_dataset_acquisition/test_getty_source.py` — 36 tests, all passing
- Architecture ready for activation if licensing is obtained

---

## Final Verdict

**GETTY_PILOT_BLOCKED_BY_LICENSING**

The Getty Images Content License Agreement explicitly prohibits using content for machine learning, AI, or biometric identification purposes without explicit authorization. Our intended use case — building a face recognition dataset — falls directly under this prohibition. Additionally, API access requires a paid subscription that we do not have. The GettySource implementation is architecturally complete and tested, but cannot be activated without resolving both licensing and access requirements.
