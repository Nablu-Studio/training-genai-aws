# Sync plan

- source: s3://genai-kb-docs/
- metadata: source, updated_at, sensitivity, owner
- sync mode: nightly + manual after critical update
- verification: doc_count, sentinel docs, retrieval smoke tests
