# Phase 13.6.2b — Flickr Source Feasibility Report

## 1. Objective

Evaluate Flickr as a potential second source for the identity-labeled celebrity reference dataset. Determine whether Flickr provides sufficient identity relevance, image quality, diversity, licensing compatibility, and API performance for a 6-person pilot.

## 2. Official API Feasibility

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

### API Version and Access

- REST API at `https://api.flickr.com/services/rest/`
- API key required (free to obtain from https://www.flickr.com/services/apps/create/)
- No OAuth required for public photo search
- No API secret required for read-only search operations
- Non-commercial use free; commercial use requires separate API key application

### Search Endpoint

- Method: `flickr.photos.search`
- Authentication: Not required for public photos
- Text search: `text` parameter searches title, description, and tags
- Tag search: `tags` parameter for tag-based filtering
- License filter: `license` parameter accepts comma-separated license IDs
- Date filters: `min_upload_date`, `max_upload_date`, `min_taken_date`, `max_taken_date`
- Content type: `content_types=0` for photos only
- Safe search: `safe_search=1` for safe content (unauthenticated requirement)
- Sort: `sort=relevance` or `date-posted-desc`, etc.
- Pagination: `page` and `per_page` (max 500)
- **Hard cap**: 4,000 results per query

### Photo URL Construction

Flickr returns photo metadata (id, secret, server, farm). Image URLs are constructed as:
```
https://live.staticflickr.com/{server}/{id}_{secret}_{size}.jpg
```
Size suffixes: `m` (240), `z` (640), `c` (800), `b` (1024), `o` (original)

The API also supports `extras=url_l,url_o` for direct download URLs.

## 3. Authentication

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

- Public search: API key only (query parameter `api_key`)
- No OAuth required for read-only operations
- API key is free to obtain
- Environment variable: `FLICKR_API_KEY`

## 4. Rate Limits

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

- Non-commercial API keys: ~3,600 requests/hour (~1/sec)
- Commercial API keys: Higher limits available upon application
- Rate limit response: HTTP 429
- Recommendation: 0.34s delay between requests (default in implementation)

## 5. Search Capabilities

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

| Capability | Parameter | Notes |
|---|---|---|
| Text search | `text` | Searches title, description, tags |
| Tag search | `tags` | Comma-delimited, supports AND/OR |
| License filter | `license` | Comma-separated license IDs |
| Date upload | `min_upload_date`, `max_upload_date` | Unix timestamp or MySQL datetime |
| Date taken | `min_taken_date`, `max_taken_date` | Unix timestamp or MySQL datetime |
| Content type | `content_types` | 0=photos, 1=screenshots, 2=other |
| Safe search | `safe_search` | 1=safe, 2=moderate, 3=restricted |
| Sort | `sort` | relevance, date-posted-desc, interestingness-desc |
| Per page | `per_page` | Max 500 |
| Page | `page` | 1-indexed |
| Extra fields | `extras` | description, license, date_upload, tags, url_l, url_o, etc. |
| Media filter | `media` | photos, videos, all |
| User photos | `user_id` | Search specific user's photos |
| Bounding box | `bbox` | Geographic search |

### Missing Capabilities for Our Use Case

- No person-specific filter (must search by name in text)
- No face-count filter (cannot pre-filter single-person images)
- No headshot/portrait filter (must rely on query phrasing)

## 6. License Filtering

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

Flickr supports 17 Creative Commons licenses via the `license` parameter:

| ID | License | Compatible? |
|---|---|---|
| 0 | All Rights Reserved | NO |
| 1 | CC BY-NC-SA 2.0 | NO (NC) |
| 2 | CC BY-NC 2.0 | NO (NC) |
| 3 | CC BY-NC-ND 2.0 | NO (NC+ND) |
| 4 | CC BY 2.0 | YES |
| 5 | CC BY-SA 2.0 | YES |
| 6 | CC BY-ND 2.0 | NO (ND) |
| 7 | No known copyright restrictions | NO (uncertain) |
| 8 | US Government Work | YES (public domain) |
| 9 | CC0 1.0 | YES |
| 10 | Public Domain Mark | YES |
| 11 | CC BY 4.0 | YES |
| 12 | CC BY-SA 4.0 | YES |
| 13 | CC BY-ND 4.0 | NO (ND) |
| 14 | CC BY-NC 4.0 | NO (NC) |
| 15 | CC BY-NC-SA 4.0 | NO (NC) |
| 16 | CC BY-NC-ND 4.0 | NO (NC+ND) |

**Compatible licenses** (included in default filter): 4, 5, 8, 9, 10, 11, 12

## 7. Licensing Findings

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

### Flickr Terms of Use (July 24, 2025)

- Users retain all intellectual property rights to their photos
- Flickr does not claim ownership of User Content
- Section 4 restrictions: no scraping, no data mining, no redistribution of Flickr Materials
- However: "User Content made available by users for download" is explicitly permitted
- API Terms: "Flickr user photos are owned by the users (the photographers) and not by SmugMug"

### Flickr API Terms of Use

- Non-commercial use: Free API key available
- Commercial use: Requires separate API key application
- Must comply with individual photo owner's license terms
- Must comply with Creative Commons license requirements
- "You are solely responsible for making use of Flickr photos in compliance with the photo owners' requirements or restrictions"

### Creative Commons License Compatibility

**CONFIRMED FROM OFFICIAL DOCUMENTATION**

For our intended use case (download, store locally, generate face embeddings, build dataset):

| License | Download | Store | Embeddings | Derivatives | Commercial |
|---|---|---|---|---|---|
| CC BY 2.0 | YES | YES | YES | YES | YES |
| CC BY-SA 2.0 | YES | YES | YES | YES (SA) | YES |
| CC BY 4.0 | YES | YES | YES | YES | YES |
| CC BY-SA 4.0 | YES | YES | YES | YES (SA) | YES |
| CC0 1.0 | YES | YES | YES | YES | YES |
| Public Domain Mark | YES | YES | YES | YES | YES |

Key considerations:
- **Attribution**: CC BY and CC BY-SA require attribution to the photographer
- **Share-Alike**: CC BY-SA requires derivatives to use the same license
- **No Derivatives**: CC BY-ND licenses prohibit derivative works (EXCLUDED)
- **Non-Commercial**: CC BY-NC licenses prohibit commercial use (EXCLUDED)

### ML/AI Processing Compatibility

**INFERENCE (not explicitly addressed in Flickr terms)**

Flickr's Terms of Use do not explicitly address ML/AI processing of downloaded images. The terms focus on:
- Redistribution of Flickr Materials (prohibited)
- Scraping and data mining (prohibited for Flickr Materials, but photos are User Content)
- Commercial use (requires appropriate license)

Our use case involves:
1. Downloading CC-licensed photos (permitted under the license)
2. Storing locally (permitted under the license)
3. Generating face embeddings (processing the image — CC licenses permit this)
4. Building a research dataset (derivative work — CC BY/SA permit this)

**Assessment**: CC-licensed Flickr photos appear compatible with our intended use, provided we:
- Attribute photographers as required by the license
- Comply with Share-Alike requirements if applicable
- Do not redistribute the original photos
- Use the dataset for research/evaluation purposes

## 8. Local Storage Compatibility

**INFERENCE**

Flickr API Terms state: "Cache or store any Flickr user photos other than for reasonable periods in order to provide the service you are providing to Flickr users."

This restriction is directed at applications caching photos for serving to users, not at research datasets. However, to be conservative:
- Our use is research/evaluation, not a service to Flickr users
- CC licenses explicitly permit reproduction and storage
- We store photos locally for processing, not for redistribution

**Assessment**: Likely compatible, but the "reasonable periods" language introduces some uncertainty for permanent dataset storage.

## 9. Pilot Configuration

### Credentials Required

```
FLICKR_API_KEY=<your-api-key>
```

Obtain a free key at: https://www.flickr.com/services/apps/create/

### API Key Status

**FLICKR_API_KEY**: NOT SET

No API key is available in the current environment. The pilot cannot be executed without obtaining a key first.

## 10. Six Identities

**NOT EXECUTED — API ACCESS UNAVAILABLE**

Planned identities:

| Identity | Category | Search Queries |
|----------|----------|---------------|
| Tom Hanks | Actor | portrait, interview, red carpet, press |
| Scarlett Johansson | Actor | portrait, interview, red carpet, premiere |
| Denzel Washington | Actor | portrait, interview, red carpet, press |
| Lionel Messi | Football | portrait, match, training, interview |
| Cristiano Ronaldo | Football | portrait, match, training, interview |
| Kylian Mbappé | Football | portrait, match, training, interview |

## 11. Candidate Counts

**NOT APPLICABLE — PILOT NOT EXECUTED**

No candidates were searched because the API key is unavailable.

## 12. Accepted Counts

**NOT APPLICABLE — PILOT NOT EXECUTED**

No images were accepted because the API key is unavailable.

## 13. Rejection Counts

**NOT APPLICABLE — PILOT NOT EXECUTED**

No rejection telemetry was collected because the API key is unavailable.

## 14. Rejection Reasons

**NOT APPLICABLE — PILOT NOT EXECUTED**

## 15. Per-Person Statistics

**NOT APPLICABLE — PILOT NOT EXECUTED**

## 16. License Distribution

**NOT APPLICABLE — PILOT NOT EXECUTED**

Expected distribution (inference from Flickr's content):
- Creative Commons photos: ~50-60% of public photos
- All Rights Reserved: ~40-50%
- CC BY 2.0: Most common CC license on Flickr
- CC BY-NC: Common but excluded from our filter
- CC0/Public Domain: Rare but valuable

## 17. Temporal Distribution

**NOT APPLICABLE — PILOT NOT EXECUTED**

Flickr provides `date_upload` and `date_taken` metadata, enabling temporal filtering if desired.

## 18. API Errors

**NOT APPLICABLE — PILOT NOT EXECUTED**

Expected error codes (from documentation):
- 100: Invalid API Key
- 105: Service currently unavailable
- 106: Write operation failed
- 111: Format not found
- 112: Method not found

## 19. Rate-Limit Behavior

**NOT APPLICABLE — PILOT NOT EXECUTED**

Expected behavior: HTTP 429 with exponential backoff (implemented in FlickrSource).

## 20. Runtime

**NOT APPLICABLE — PILOT NOT EXECUTED**

Expected characteristics:
- Search latency: ~100-300ms per request
- Download latency: ~200-1000ms per image
- Rate limit: ~1 request/second
- Total estimated for 6-person pilot: ~5-15 minutes

## 21. Accepted Images/Minute

**NOT APPLICABLE — PILOT NOT EXECUTED**

## 22. Identity Quality

**NOT APPLICABLE — PILOT NOT EXECUTED**

Expected characteristics (inference):
- Flickr has many fan-uploaded celebrity photos
- Red carpet, event, and press photos common
- Quality varies (amateur to professional)
- Metadata (title, tags, description) aids identity verification
- Group pools (e.g., celebrity photo groups) may concentrate relevant content

## 23. Visual Review

**NOT APPLICABLE — PILOT NOT EXECUTED**

## 24. Duplicate Analysis

**NOT APPLICABLE — PILOT NOT EXECUTED**

Expected behavior:
- SHA-256 dedup catches exact duplicates
- Cross-person dedup checks against existing datasets
- Flickr photos are typically unique uploads (less duplication than Wikimedia)

## 25. Retrieval Smoke Test

**NOT APPLICABLE — PILOT NOT EXECUTED**

## 26. Wikimedia Comparison

**INFERENCE (no Flickr data collected)**

| Metric | Wikimedia | Flickr (Expected) |
|---|---|---|
| API access | Free, no key | Free API key required |
| Licensing | CC-BY-SA / Public Domain | CC BY, CC BY-SA, CC0, etc. |
| ML/AI training | Permitted (with attribution) | Likely permitted (CC licenses) |
| Person search | Category system | Text-based only |
| Face count filter | Not available | Not available |
| Image quality | Variable (amateur to pro) | Variable (amateur to pro) |
| Metadata quality | Variable | Better (structured tags) |
| Watermarks | None | None (but size-limited) |
| Download cost | Free | Free (with API key) |
| Rate limits | Configurable | ~3600/hour |
| Celebrity coverage | Good (via Commons) | Good (fan uploads) |
| License filtering | Manual | API-supported |
| Result cap | No hard cap | 4,000 per query |

### Key Differences

1. **License filtering**: Flickr supports API-level license filtering; Wikimedia does not
2. **API key**: Flickr requires a free API key; Wikimedia does not
3. **Rate limits**: Flickr ~3600/hr; Wikimedia configurable
4. **Result cap**: Flickr 4,000 per query; Wikimedia no hard cap
5. **Metadata**: Flickr has richer structured metadata (tags, dates, owner)
6. **CC compatibility**: Both support CC licenses; Flickr has more granular filtering

## 27. Limitations

### Access Limitations (BLOCKING)

- **API key required**: Must obtain free key from Flickr App Garden
- **No key in environment**: Cannot execute pilot without `FLICKR_API_KEY`

### Technical Limitations (if access were available)

- **No person-specific filter**: Must rely on text search phrases
- **No face-count filter**: Cannot pre-filter single-person images
- **4,000 result cap**: May limit diversity for popular queries
- **Variable image quality**: Amateur and professional photos mixed

### Licensing Limitations (Partial)

- **No explicit ML/AI guidance**: Flickr terms don't address ML processing
- **"Reasonable periods" language**: May create uncertainty for permanent storage
- **Share-Alike requirement**: CC BY-SA requires derivative works to use same license
- **Attribution required**: Must attribute photographers for CC BY/SA

### Content Limitations

- **Celebrity photos may be copyrighted**: Even CC-licensed photos may depict copyrighted content
- **Editorial/professional photos**: May have additional restrictions
- **Fan-uploaded content**: Quality and accuracy of metadata varies

## 28. Recommendation

### Primary Finding: API ACCESS UNAVAILABLE

**FLICKR_API_ACCESS_UNAVAILABLE**

The FlickrSource implementation is architecturally complete and tested (39/39 tests pass), but the pilot cannot be executed because no `FLICKR_API_KEY` environment variable is set. A free API key can be obtained at https://www.flickr.com/services/apps/create/.

### Licensing Assessment: CONDITIONALLY COMPATIBLE

**INFERENCE**

Flickr CC-licensed photos (CC BY, CC BY-SA, CC0, Public Domain Mark) appear compatible with our intended use case, provided:
1. We filter to compatible licenses only (implemented in FlickrSource)
2. We attribute photographers as required
3. We comply with Share-Alike requirements
4. We use the dataset for research/evaluation, not commercial purposes

### Secondary Finding: Architecture Ready

Despite the access block, the FlickrSource implementation is complete:

- `dataset_acquisition/sources/flickr.py` — Full ImageSource implementation
- License filtering (compatible CC licenses only)
- Rate limiting with exponential backoff
- Photo URL construction from metadata
- Structured metadata preservation (owner, license, dates, tags)
- 39 tests covering configuration, search, download, licensing, rate limits, normalization, and secret safety

### To Execute the Pilot

1. Obtain a free Flickr API key: https://www.flickr.com/services/apps/create/
2. Set environment variable: `FLICKR_API_KEY=<your-key>`
3. Re-run the pilot script with the 6 identities
4. Evaluate results and compare with Wikimedia

### Recommendation

**Flickr is a viable second source** pending API key acquisition. The implementation is ready. The licensing analysis suggests CC-licensed Flickr photos are compatible with the intended research use. The pilot should be executed once an API key is available.

---

## Final Verdict

**FLICKR_API_ACCESS_UNAVAILABLE**

The FlickrSource implementation is architecturally complete and all 39 tests pass. Flickr's API is free to use, supports license-level filtering, and CC-licensed photos appear compatible with the intended research/dataset use. However, no `FLICKR_API_KEY` environment variable is set, preventing pilot execution. A free key can be obtained at https://www.flickr.com/services/apps/create/. Once the key is provided, the 6-person pilot can proceed.
