# Data center buildout workflow

Tilebox workflow that ranks data center sites by visible Sentinel-2 change between a before and after date.

The root task is `tilebox.com/datacenters/RankDataCenterBuildout@v1.14`. It accepts:

```json
{
  "csv_url": "https://docs.google.com/spreadsheets/d/1JJ6kcVo-NjlAYtznwHOki2DVl4WWV6lhy-eXhFCdKKU/export?format=csv&gid=386766486",
  "max_sites": 3,
  "random_seed": 1337,
  "before_date": "2024-05-01",
  "after_date": "2026-05-01",
  "window_days": 60,
  "crop_size_m": 3000,
  "scene_cloud_cover_max": 30.0,
  "crop_cloud_cover_max": 1.0,
  "status_filter": [
    "Approved/Permitted/Under construction",
    "Expanding",
    "Proposed"
  ]
}
```

If `status_filter` is omitted or set to `null`, it defaults to the three statuses shown above. The workflow applies this filter before merging datapoints into sites.

For every merged site, the workflow selects a before and after Sentinel-2 L2A scene, reads cropped assets from the Copernicus archive, caches raw lossless cropped bands as `.npz`, writes an RGB `preview.png`, computes CVA connected components, construction-specific masks, SSIM structural change, and patch-level Clay embedding distance components, and stores `outputs/ranking.json` in the job cache.
