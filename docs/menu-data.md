# Eataly AI — Enriched menu knowledge base and 50-query evaluation set

## Files
- `eataly_menu_enriched.jsonl`: 160 menu records; **one JSON object per line**, suitable for LangChain document ingestion.
- `eataly_menu_enriched.json`: the same records as an array.
- `eataly_eval_50.jsonl` / `.json` / `.csv`: 50 manually designed retrieval/answer test cases with expected source item IDs and target answers.

## Menu enrichment fields
- `guest_friendly_description`: one short, appealing suggested guest-facing sentence based on listed ingredients; not verified restaurant copy.
- `allergen_assessment.inferred_contains`: flags inferred from **explicitly listed ingredients** or highly conventional implied recipe components; these are **NOT confirmed allergen declarations**.
- `allergen_assessment.possible_unconfirmed`: additional possible allergens from commonly used sauces, recipes, ingredients or cross-label uncertainty; **not proof of presence**.
- `allergen_assessment.assessment_status`, `verified_by_restaurant=false`, `needs_staff_confirmation=true` explicitly track source limitations.
- `search_text_enriched`: formatted text to embed; avoid putting unverified inferred allergens in strict metadata filters for food safety.

**Important:** Every item requires staff verification before allergen advice. The menus do not supply full recipes, supplier packaging, processing information or cross-contact controls. No record can be assumed allergen-free. Inferred flags are not exhaustive. A positive inferred flag must not be represented as a certified declaration, and missing flags are never evidence of absence.

Some items are beverages, add-ons and complete set offers, not individual prepared dishes. Guest-friendly copy for wine intentionally avoids inventing flavour notes. Prices reflect menu print dates, not verified current availability. Promotion availability may be time-limited.

## Evaluation design
- `expected_item_ids` is the required relevant **retrieval evidence**, not necessarily the only correct result. Some multi-item questions have legitimate additional supporting records.
- Use `Recall@5` or `Hit@5` over retrieved item IDs and evaluate factual answer content against `expected_answer`.
- For empty IDs, test whether the assistant avoids unsupported guarantees; don't compute standard recall for those questions.
- For food-allergy cases, require uncertainty / referral to staff where appropriate; never accept an unqualified allergen-free assurance.
- Run the same cases under Chroma vector-only and hybrid retrieval, then compare results.
- Menu sources: EEC_PIPA_Menu_Main_2026.10.22.pdf; ECAN_PIPA_Menu_Lunch_2026.09.17.pdf; EEC_PIPA_Menu_Dessert_2026.10.01.pdf; seasoned for fall.pdf.
