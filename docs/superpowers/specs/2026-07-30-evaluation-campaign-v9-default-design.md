# Evaluation Campaign v9 Default Design

## Goal

New evaluation campaigns default to Agentic v9 so an omitted or untouched
version selection cannot silently create a v8 campaign.

## Scope

- Change the Evaluation Setup UI's initial Agentic execution version to `v9`.
- Change evaluation campaign creation/configuration defaults to `v9`.
- Keep explicit `v8` requests valid.
- Keep existing campaign snapshots and stored results unchanged.
- Keep historical-data parsing fallbacks at `v8` where absence means the
  execution predates explicit version recording.
- Do not change the daily Agentic Chat runtime or its defaults.

## Data Flow

1. Evaluation Setup opens with `v9` selected.
2. When Agentic mode is selected, the UI runs the existing v9 preflight.
3. The create request explicitly records `agentic_execution_version: "v9"`.
4. If another evaluation client omits the field, the backend campaign request
   and config models also default to `v9`.
5. Explicit v8 identities continue to require and preserve `v8`.

## Compatibility

This is a creation-default change, not a migration. Stored campaign snapshots,
run identities, analytics fallbacks, trace schemas, and database projections
must not be rewritten from v8 to v9.

## Verification

- Frontend test proves untouched Agentic selection submits v9 and performs v9
  preflight.
- Backend schema tests prove omitted creation/config version becomes v9.
- Backend tests prove explicit v8 identities remain valid.
- Existing historical compatibility tests remain unchanged and pass.

