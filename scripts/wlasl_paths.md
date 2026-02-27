# Expected WLASL Paths and Formats

## Dataset root
`DATA_ROOT` is the base folder containing video files referenced by annotations.

## Labels mapping
`LABELS_PATH` should point to the mapping used by training/export.
Supported forms:
- JSON list: `['label_a', 'label_b', ...]`
- JSON dict (id->label): `{'0':'label_a', '1':'label_b', ...}`
- JSON dict (label->id): `{'label_a':0, 'label_b':1, ...}`

For WLASL100 sanity, mapping must resolve to exactly 100 contiguous IDs `[0..99]`.

## Annotation split file
`ANN_PATH` can be:
- `.json` (primary/expected in PR#4)
- `.csv` (optional, unsupported by default in PR#4 unless `ALLOW_CSV=True`)

Required per-sample information:
- video path key: one of `video_relpath`, `video_path`, `video`, `path`, `file`, `url`, `filepath`, `filename`
- plus either:
  - label key (`label`, `gloss`, `word`, ...), or
  - class id key (`class_id`, `label_id`, `id`, ...)

Optional:
- `split` key; script filters by `SPLIT` when present.
