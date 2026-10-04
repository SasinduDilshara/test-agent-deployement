# Documents

These are sample documents for **Lumora Systems**, a fictional company. They are not real policies or legal advice.

## `hr/`: HR policy documents (indexed into Pinecone)

| doc_id | Title | Category |
|---|---|---|
| `leave-policy` | Leave Policy | leave |
| `working-hours-and-remote-work` | Working Hours and Remote Work Policy | workplace |
| `compensation-and-payroll` | Compensation and Payroll Policy | compensation |
| `benefits` | Employee Benefits Policy | benefits |
| `code-of-conduct` | Code of Conduct | conduct |
| `grievance-and-disciplinary` | Grievance and Disciplinary Policy | conduct |
| `onboarding-and-probation` | Onboarding and Probation Policy | employment-lifecycle |
| `resignation-and-offboarding` | Resignation and Offboarding Policy | employment-lifecycle |
| `performance-management` | Performance Management Policy | performance |
| `travel-and-expense` | Travel and Expense Policy | finance |

### Document format (required)

```markdown
---
doc_id: leave-policy          # required; lowercase letters, digits and hyphens; must equal the file name
title: Leave Policy           # required
category: leave               # required; usable as a search filter
version: "2.1"                # bumped automatically by the agent on every update
effective_date: "2026-04-01"
last_updated: "2026-04-01"    # set automatically by the agent on every update
owner: People & Culture
applies_to: ...
---

# Leave Policy

## Annual Leave              # '##' sections are the unit the agent can update
...
### Optional sub-heading
```

To add a document, create `hr/<doc_id>.md` in this format and run `hr-ingest ingest` from `rag-ingestion/`.

## `company/company_profile.json`: company metadata (not HR policy)

This file holds the company's legal details, headquarters, offices, headcount, leadership, departments, products, certifications, key contacts, working calendar and social links. The agent reads it directly with its `get_company_details` tool. It is **not** indexed into Pinecone.
