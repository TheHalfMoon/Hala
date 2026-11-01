# H317 — canonical metadata identifiers and strict JSON key custody

This stacked source-only guard extends H315/H316 without reading image or
application bytes. It **never authorizes TRAIN, DEV, or OOD admission**.

The input metadata envelope now rejects duplicate JSON keys at every nesting
level, instead of accepting a parser's last value. A `source_repo` must be
an unambiguous GitHub-style `owner/repository` identifier (not a URL/ref).
An `asset_path` must be a canonical POSIX-relative identifier (not a
filesystem location), with no absolute path, parent/current-directory segment,
empty segment, backslash, query/fragment, escape sequence, or control/format
character. Unicode letters, including Arabic file names, remain supported.

The validator does not follow identifiers, resolve paths, fetch sources,
validate legal title, or assert real per-asset rights. Conservative rejection
can exclude otherwise valid unusual filenames; a rejected record stays
quarantined for independent review, never auto-normalized or accepted.

Invented-only regression tests exercise paths, repositories, duplicate JSON
keys, output sanitization, and continued no-admission behavior:

```sh
python -m unittest discover -s tests -p 'test_h317_*.py' -v
```

Official H199 TRAIN admission remains empty; H173 scored OOD stays sealed;
H250/H251 external-source assets remain quarantined; H293/H294 safety blocks
remain untouched. No real media, model weights, GPU use, paid inference, merge,
release or deployment.
