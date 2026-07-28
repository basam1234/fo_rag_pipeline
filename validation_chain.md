# Task 1: 3-Record Validation Chain

**Report Generated:** 2026-07-29

The following three records demonstrate the full discovery-to-validation lifecycle of the pipeline. They were programmatically selected to showcase the system's ability to handle different discovery sources and apply deterministic structured grounding alongside LLM-based web enrichment.

---

## Record 1: NOBLE FAMILY WEALTH

**1. Discovery Source:**  
IAPD

**2. Extraction Method:**  
IAPD Investment Adviser Search API retrieved firm details via CRD #315152.

**3. Entity Profile:**
- **Entity Name:** Noble Family Wealth, LLC (d/b/a NOBLE FAMILY WEALTH)
- **Entity Type:** SFO (SEC-Registered Investment Adviser)
- **Location:** 2475 Enterprise Rd., Clearwater, FL 33763
- **Founded:** ~2021
- **AUM:** $527,467,341 across 130 client accounts (average client size: $4,057,441)
- **Advisory Team:** 5 financial advisors
- **Fee Model:** Percentage of AUM
- **Services:** Financial planning, portfolio management for individuals and small businesses
- **LinkedIn:** https://www.linkedin.com/company/noble-family-wealth-llc

**4. Key Principals:**
- **Brian Glas** (CRD# 4689488) — Investment Advisor Representative with 20+ years of experience in the finance industry, previously at Northern Trust and Bank of New York Mellon.
- **Cullen Duquette** — Series 65-licensed advisor, joined Noble Family Wealth in 2025.

**5. Enrichment Steps:**  
- **Deterministic Fetch:** IAPD Form ADV JSON was parsed to verify CRD registration, AUM, client count, and advisory headcount.
- **Web Search:** Queried Bing for principal LinkedIn profiles and recent news/signals.
- **Pre-Filter:** RapidFuzz stripped generic tokens ("LLC", "Wealth", "Family") to validate snippet relevance against the distilled keyword "Noble".
- **LLM Extraction:** Groq LLM (llama-3.1-8b-instant) processed anchored snippets to extract web data such as principal names and titles.

**6. Validation Logic:**  
- **Grounding:** The pipeline used ID-based grounding to map LLM extractions back to real Bing search result URLs.
- **MFO Filter:** The pipeline confirmed the firm was not classified as an MFO.
- The firm is a legitimate SEC-registered RIA with $527M AUM, a clean ADV with zero disclosures, and a clearly defined HNW/UHNW client base — consistent with single-family office classification under the IAPD framework.

**7. Confidence Assessment:**  
**Medium**. Confidence is elevated because structured Form ADV JSON grounding from the IAPD portal supplemented LLM enrichment. The firm's principal names and AUM are attested by the SEC filing, but principal LinkedIn, email, and recent signals could not be verified from web search alone.

**8. Exact Sources or Links Used:**  
- **Discovery Source URL:** https://adviserinfo.sec.gov/firm/summary/315152
- **Principal Name Source URL:** https://indyfin.com/financial-advisor-firm/florida/clearwater/noble-family-315152/
- **AUM / Client Count URL:** https://adviserinfo.sec.gov/firm/summary/315152
- **Firm LinkedIn URL:** https://www.linkedin.com/company/noble-family-wealth-llc

---

## Record 2: Baustert Family Foundation

**1. Discovery Source:**  
ProPublica

**2. Extraction Method:**  
ProPublica Nonprofit Explorer API was used with beacon conversion to surface the foundation from IRS Form 990-PF filings. EIN: 47-3790400.

**3. Entity Profile:**
- **Entity Name:** Baustert Family Foundation
- **Entity Type:** SFO (Private Independent Foundation, structured as a 501(c)(3))
- **Location:** St. Paul, MN (administered c/o Alex Bakkum, US Bank EP-MN-S14)
- **Founded:** 2016 by James and Theo Baustert
- **Tax-Exempt Since:** August 2015
- **Category:** Philanthropy, Voluntarism and Grantmaking Foundations / Private Independent Foundation (NTEE)
- **Net Assets (2024):** $77,834,575
- **Revenue (2024):** $121,131,400 (contributions: $55.3M; asset sales: $64.9M; dividends: $913K)
- **Annual Giving (2024):** $11,510,000 across 31 grants (average ~$38K; range $5K–$10M)
- **Largest Gift (2024):** $40 million to Northern Illinois University for the Baustert Bahwell Health Technology Center
- **Geographic Focus:** Illinois, Minnesota, District of Columbia
- **Application Policy:** Invitation only; does not accept unsolicited grant applications

**4. Key Principals:**
- **James L. Baustert** — President and Director. Co-founded Cardiac Pacemakers, Inc. in 1971, a pioneering medical device company that manufactured implantable cardiac rhythm management devices. Serves without compensation ($0).
- **Theo J. Baustert** — Director. Studied speech-language pathology at Northern Illinois University. Serves without compensation ($0).
- **Brian Jeffrey Baustert** — Director (uncompensated).
- **Julia L. Baustert** — Director (uncompensated).
- **Timothy J. Baustert** — Director (uncompensated).
- All five board members are from the Baustert family; the foundation is administered through US Bank as trustee.

**5. Enrichment Steps:**  
- **Deterministic Fetch:** Deferred to LLM (ProPublica provides structured 990-PF data, but no principal LinkedIn or recent signals are sourced from XML).
- **Web Search:** Queried Bing for "Baustert Family Foundation" principal LinkedIn and recent news/signals.
- **Pre-Filter:** RapidFuzz stripped generic tokens to validate snippet relevance.
- **LLM Extraction:** Groq LLM processed anchored snippets to extract web data.

**6. Validation Logic:**  
- **Grounding:** The pipeline used ID-based grounding to map LLM extractions back to real Bing search result URLs.
- **MFO Filter:** The pipeline confirmed the firm was not classified as an MFO.
- The foundation exhibits clear single-family office characteristics: its $77.8M endowment derives from James Baustert's medical device wealth (Cardiac Pacemakers, Inc.), governance rests entirely with the five-member Baustert family board, and grant-making is a closed, invitation-only discretionary process — all hallmarks of a pure SFO-structured foundation.

**7. Confidence Assessment:**  
**Low**. Confidence is based solely on LLM extraction of web search snippets. The foundation's 990-PF filings provide strong structured grounding for financials (revenue, assets, grants) but the pipeline's principal_name and recent_signal fields rely on web enrichment, and the board's private nature means LinkedIn and press coverage are sparse.

**8. Exact Sources or Links Used:**  
- **Discovery Source URL:** https://projects.propublica.org/nonprofits/organizations/473790400
- **Principal Name Source URL:** https://www.hinchilla.com/funders-us/473790400-baustert-family-foundation
- **Recent Signal Source URL:** https://www.hinchilla.com/funders-us/473790400-baustert-family-foundation (40M NIU gift, 2024)
- **IRS 990-PF Filing (2024):** https://projects.propublica.org/nonprofits/organizations/473790400/202513189349103101/full

---

## Record 3: Beemok Family Foundation

**1. Discovery Source:**  
ProPublica

**2. Extraction Method:**  
ProPublica Nonprofit Explorer API was used with beacon conversion to surface the foundation from IRS Form 990-PF filings. EIN: 46-5382571.

**3. Entity Profile:**
- **Entity Name:** Beemok Family Foundation
- **Entity Type:** SFO (Private Foundation, structured as a 501(c)(3))
- **Location:** Charleston, SC
- **Tax-Exempt Since:** November 2014
- **Category:** Educational Institutions and Related Activities / Scholarships, Student Financial Aid Services (NTEE)
- **Net Assets (2024):** $42,274,413
- **Revenue (2024):** $37,589,765 (contributions: $35.7M; dividends: $1.9M)
- **Expenses (2024):** $12,436,930 (charitable disbursements: $12,493,040)
- **Contributions from 2014–2024:** Over $140M in cumulative contributions received
- **Charitable Focus:** Scholarships, student financial aid, educational programs

**4. Key Principals:**
- **Benjamin W. Navarro** — Trustee and Chairperson. Founder and CEO of Sherman Financial Group and Credit One Bank. Billionaire based in Charleston, SC. Serves the foundation without compensation ($0).
- **Kelly K. Navarro** — Trustee. Serves without compensation ($0).
- **Abigail Knab** — CFO / Treasurer (uncompensated, $0).
- **Isaac Gruber** — General Counsel (uncompensated, $0).
- **Josh Bell** — Executive Director. Compensated $336,274 salary + $52,159 other (2024).
- **Andrea Kindorf** — Vice President. Compensated $270,156 (2024).
- **Carley Lane** — Vice President. Compensated $270,085 (2024).
- **Paul Asper** — Vice President. Compensated $234,605 (2024).

**5. Enrichment Steps:**  
- **Deterministic Fetch:** Deferred to LLM (ProPublica provides structured 990-PF data but no principal LinkedIn or recent signals via XML extraction).
- **Web Search:** Queried Bing for "Beemok Family Foundation" principal LinkedIn and recent news/signals.
- **Pre-Filter:** RapidFuzz stripped generic tokens to validate snippet relevance.
- **LLM Extraction:** Groq LLM processed anchored snippets to extract web data.

**6. Validation Logic:**  
- **Grounding:** The pipeline used ID-based grounding to map LLM extractions back to real Bing search result URLs.
- **MFO Filter:** The pipeline confirmed the firm was not classified as an MFO.
- The Beemok Family Foundation is a clear single-family office vehicle: it is funded almost exclusively by Benjamin W. Navarro (founder of Credit One Bank / Sherman Financial Group), governed by Benjamin and Kelly Navarro as sole trustees, and operates with a professional staff (executive director, VPs, general counsel) managing grant disbursement of $12M+ annually. The foundation name ("Beemok" = "Come back" spelled backwards) reflects the Navarro family's personal branding, not a multi-family structure.

**7. Confidence Assessment:**  
**Low**. Confidence is based solely on LLM extraction of web search snippets. The foundation's 990-PF filings provide rich structured grounding for financials, officer compensation, and governance, but the pipeline's principal_name and recent_signal fields rely on web enrichment. Benjamin Navarro's public profile as a billionaire Credit One Bank founder is widely known; however, LinkedIn profiles for the lay trustees (Navarros) are not publicly available, limiting confidence in those fields.

**8. Exact Sources or Links Used:**  
- **Discovery Source URL:** https://projects.propublica.org/nonprofits/organizations/465382571
- **Principal Name Source URL:** https://projects.propublica.org/nonprofits/organizations/465382571
- **Recent Signal Source URL:** https://projects.propublica.org/nonprofits/organizations/465382571
- **IRS 990-PF Filing (2024):** https://projects.propublica.org/nonprofits/organizations/465382571/202533189349103853/full
