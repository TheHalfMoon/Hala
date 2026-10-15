# Small experimental adapter archive

This isolated branch is an archival destination authorized for Hala experiments. It is never merged into main or the foundation PR and does not represent a qualified model release. Base weights, customer data, credentials and executable pickle snapshots are excluded. Only small adapter safetensors, bounded configuration/metadata and exact SHA256 manifests may be retained after independent source review.

The initial roundtrip record is a harmless storage and retrieval probe. It contains no trained adapter and proves no inference or training result. An immutable commit and independently retrieved byte hash must be recorded before a later training experiment relies on this destination. A future actual adapter requires its own verified manifest and fresh-session download/load/evaluation evidence.
