# Data Contract

This pipeline never stores raw images, private manifests, checkpoints, or results in Git.

## Required manifest format

Each CSV manifest must contain:

| Column | Description |
|---|---|
| path | Absolute or container-mounted path to an image |
| label | Class label |
| source | Optional data source, device, farm, camera, or IoT site identifier |

Example:

```csv
path,label,source
/data/site_1/image_001.png,Sound,site_1
/data/site_2/image_002.png,Fusarium,site_2

