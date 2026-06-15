# Future SafeLens Dataset Collection Prompt

Review `SafeLens-Agentic-HSE-Combined-Document.md` and the SafeLens bundle's
dataset manifest, reference catalog, taxonomy, and research files. Connect the
Google Drive plugin and use this restricted destination:

`https://drive.google.com/drive/folders/1TQwO-wIOqSdJ_clfe0-QmjGgq9M6mX8V?usp=drive_link`

Before downloading, inspect the Drive folder and produce a source/license/size
inventory. Reverify every license from the primary publisher page. Acquire only
CC0, CC BY, Apache-2.0, MIT, or explicitly authorized company-owned media.
Exclude NC, AGPL, unclear-license, and provenance-uncertain media from production
data.

Ask before using authenticated Roboflow, Kaggle, or similar accounts and before
transferring more than 5 GB. Preserve source archives, licenses, attribution,
checksums, acquisition dates, versions, and original annotations. Deduplicate
overlapping datasets, validate boxes, normalize labels to the SafeLens taxonomy,
and split by source/site/video rather than individual image.

For licensed videos, retain raw video only when useful and extract
representative frames with source and sequence identifiers to prevent leakage.
Create YOLO and COCO exports, class/source statistics, a rejected-source report,
and a reproducible acquisition log. Never commit media to Git or place
private/customer images in a public link.

The official source pages previously reviewed reported CC BY 4.0 for the
principal Roboflow, Hugging Face, and SHEL5K sources, but verify those terms
again at acquisition time.
