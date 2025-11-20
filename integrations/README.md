# Vision Service Integration

This directory contains integration modules for uploading training data and visualizations to the Vision Service API.

## Modules

### `vision_service.py`
Core integration functions for training runs, epochs, configs, and test results.

**Functions:**
- `create_training()` - Create a new training run
- `send_epoch_results_from_file()` - Upload epoch metrics from JSON
- `create_config()` - Register training configuration
- `send_test_results_from_file()` - Upload test results
- `upload_visualization()` - Upload individual visualization file

### `visualization_uploader.py`
Specialized module for uploading visualization images to the Vision API.

**Functions:**
- `upload_visualization()` - Upload a single visualization file
- `upload_all_visualizations_for_image()` - Upload all viz types for one image
- `get_epoch_uuid_from_model_path()` - Extract epoch UUID from checkpoint filename

### `training_logger.py`
Training integration that automatically logs metrics during training.

## Visualization Upload

The visualization script (`visualize.py`) now supports automatic upload to the Vision API.

### Usage

**Basic visualization (no upload):**
```bash
python visualize.py -c config/zod/config_1.json -p zod_dataset/visualizations.txt
```

**With automatic upload:**
```bash
python visualize.py -c config/zod/config_1.json -p zod_dataset/visualizations.txt --upload
```

The script will:
1. Auto-detect the epoch UUID from the model checkpoint filename
2. Generate all visualization types (segment, overlay, compare, correct_only)
3. Upload each visualization to the Vision API via MinIO
4. Create database records for each uploaded visualization

**Specify epoch UUID manually:**
```bash
python visualize.py -c config/zod/config_1.json --upload --epoch-uuid abc123-def456-...
```

### Visualization Types

The script generates and uploads 4 types of visualizations per image:

- **segment**: Model prediction segmentation map with class colors
- **overlay**: Original image overlaid with prediction
- **compare**: Red (incorrect) / Green (correct) comparison mask
- **correct_only**: Segmentation showing only correctly predicted pixels

### Model Checkpoint Format

For auto-detection to work, model checkpoints should follow this naming:
```
epoch_{epoch_num}_{epoch_uuid}.pth
```

Example: `epoch_9_19f7fbfc-88ac-4944-b8c1-7cc2b96ed1c3.pth`

### API Endpoints Used

The visualization upload process uses these Vision API endpoints:

1. **POST** `/api/visualizations/upload-url` - Get signed upload URL
2. **PUT** `{signed_url}` - Upload file to MinIO
3. **POST** `/api/visualizations` - Create visualization record

### Error Handling

The script includes comprehensive error handling:
- Missing files are logged but don't stop processing
- Failed uploads are counted and reported
- Network errors are caught and logged
- Final summary shows upload success/failure counts

## Configuration

### API Base URL

Set in both `vision_service.py` and `visualization_uploader.py`:
```python
VISION_API_BASE_URL = "https://vision-api.tumbaland.eu/api"
```

For local development, change to:
```python
VISION_API_BASE_URL = "http://localhost:4010/api"
```

### Supported Image Formats

- PNG (`.png`)
- JPEG (`.jpg`, `.jpeg`)

Default mimetype: `image/png`

## Example Integration

```python
from integrations.visualization_uploader import upload_all_visualizations_for_image

# Upload all visualizations for a single image
results = upload_all_visualizations_for_image(
    epoch_uuid="abc123-def456",
    output_base="logs/zod/config_1/visualizations",
    image_name="image_001.png"
)

# Check results
for viz_type, result in results.items():
    if result:
        print(f"✓ {viz_type} uploaded successfully")
    else:
        print(f"✗ {viz_type} upload failed")
```

## Troubleshooting

**Issue: "Could not auto-detect epoch UUID"**
- Ensure model checkpoint follows naming convention
- Use `--epoch-uuid` flag to specify manually

**Issue: Upload fails with timeout**
- Check network connection to Vision API
- Verify API is accessible: `curl https://vision-api.tumbaland.eu/api/health`

**Issue: "File not found"**
- Verify visualization files were generated
- Check output directory: `logs/{dataset}/{config}/visualizations/`

**Issue: 409 Conflict error**
- Visualization with same UUID already exists
- This is normal if re-running upload for same epoch

## OpenAPI Specification

Full API documentation: `integrations/openapi.yml`

View interactive docs at:
- Production: https://vision-api.tumbaland.eu/docs
- Development: http://localhost:4010/docs
