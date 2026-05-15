Architecture diagram

 ┌────────────────────────────────────────────────────────────────────────────┐
 │                         MouseHash DANDI Chat Agent                         │
 └────────────────────────────────────────────────────────────────────────────┘
                                 ▲           │
                 user text ──────┘           └────── suggestions, evidence,
                                                        artifact links
                                        │
             ┌──────────────────────────┴───────────────────────────┐
             │       SmoLAgents loop  (LiteLLM • Claude 4.6)        │
             │       system prompt: "narrate evidence, not guesses" │
             └──────────────────────────┬───────────────────────────┘
                                        │   tool dispatch
                                        ▼
   ┌────────────────────────────────────────────────────────────────────────┐
   │   inspect_dandiset · parse_nwb_manifest · show_role_manifest           │
   │   suggest_analyses  · propose_transformation_plan                      │
   │   confirm_and_run_move · explain_blocked_tools · list_paper_evidence   │
   └──┬──────────────────────┬───────────────────────┬────────────────────┬─┘
      │                      │                       │                    │
      ▼                      ▼                       ▼                    ▼
  ┌────────────┐     ┌────────────────┐    ┌──────────────────┐
 ┌──────────────┐
  │  Parser    │     │  Readiness     │    │  AnalysisMove    │   │  BlahML
   │
  │ orchestr.  │     │  engine        │    │  builder         │   │  dialogue
   │
  │            │     │                │    │  (transform +    │   │  + param
   │
  │ ManifestPa-│     │ score_move +   │    │  view + tool +   │   │  validation
   │
  │ rser  ⊕    │     │ rank_suggest.  │    │  validation)     │   │
   │
  │ apply_rules│     │                │    │                  │   │
   │
  └─────┬──────┘     └───────┬────────┘    └────────┬─────────┘
 └──────┬───────┘
        │                    │                      │                    │
        ▼                    ▼                      ▼                    ▼

 ┌────────────────────────────────────────────────────────────────────────────┐
  │  Typed core  (pydantic)
 │
  │  EvidenceItem · EvidenceBackedRoleManifest · TransformationSpec · ToolSpec
 │
  │  AnalysisView · AnalysisMove · ToolReadinessReport
 │

 └────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼

 ┌────────────────────────────────────────────────────────────────────────────┐
  │  DataJoint schema  dandi_agent
 │
  │  Dandiset (M) → NWBAsset (M) → RoleManifest (C)
 │
  │      ↓
 │
  │   AnalysisView (C) ← TransformationSpec (L) + TransformationRun (C)
 │
  │      ↓
 │
  │   ToolRun (C) ← ToolSpec (L)  →  AnalysisArtifact (C)
 │

 └────────────────────────────────┬───────────────────────────────────────────┘
                                   │
                                   ▼
             ┌─────────────────────┴─────────────────────┐
             │ DANDI archive  •  local NWB cache         │
             │ MVP-3:  Crossref / bioRxiv abstracts      │
             └───────────────────────────────────────────┘

        Roles  →  Transformations  →  AnalysisView  →  Tool  →  Artifact
        (what exists)   (how it becomes analyzable)  (the question)  (audit)
