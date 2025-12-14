# Carotid Plaque Segmentation and Classification Project

## Project Overview

This project implements a semi-supervised carotid plaque segmentation and classification system. It adopts a Mean Teacher framework to perform multi-task learning on carotid ultrasound images (longitudinal and transverse views):

- Segmentation task: identify vessel and plaque regions (3-class segmentation: background, vessel, plaque)
- Classification task: determine plaque risk level (RADS2 vs RADS3-4, binary classification)

## Project Structure

```
CSV-Challenge-ISBI2026/
├── baseline/                          # Baseline model training code
│   ├── Dockerfile                     # Docker image configuration
│   ├── Dataset.py                     # Dataset loading and data augmentation
│   ├── model.py                       # Model definition
│   ├── train.py                       # Semi-supervised training script
│   ├── test.py                        # Testing and evaluation script
│   ├── augmentations/                 # Data augmentation modules
│   │   ├── __init__.py
│   │   └── ctaugment.py               # CTAugment augmentation strategy
│   └── utils/                         # Utility functions
│       ├── eval_utils.py              # Evaluation metrics (Dice, NSD, etc.)
│       ├── losses.py                  # Loss functions
│       ├── ramps.py                   # Consistency weight scheduling
│       ├── util.py                    # General utilities
│       └── vis_utils.py               # Visualization utilities
│
├── docker_submission_template/        # Docker submission template
│   ├── run_infer.py                   # Inference script
│   ├── Dockerfile                     # Docker image configuration
│   ├── requirements.txt               # Python dependencies
│   ├── models/                        # Directory to place model code (e.g. model.py)
│   └── weights/                       # Directory to place model weights (.pth files)
│
└── Readme.md                          # Project documentation
```

## Usage Guide

### 1. Get the Project Files

```bash
git clone https://github.com/xxx/xxx.git
```

The repository contains two Docker-based components: `baseline` and `docker_submission_template`, which correspond to the baseline model and the submission template, respectively.

### 2. Train the Model

#### Data Preparation

The data should be organized in the following structure:

```
data/
└── train/
    ├── train.list              # List of training samples
    ├── img/                    # Image files (.h5 format)
    │   ├── case_001.h5         # Contains long_img and trans_img datasets
    │   └── ...
    └── mask/                   # Annotation files (.h5 format)
        ├── case_001_label.h5   # Contains long_mask, trans_mask, cls datasets
        └── ...
```

#### Training Parameter Configuration

Edit the `Args` class in `baseline/train.py`:

```python
class Args:
    root_path = "/path/to/data"           # Path to the data
    max_iterations = 30000                # Maximum number of iterations
    batch_size = 24                       # Total batch size
    labeled_bs = 8                        # Batch size for labeled samples
    num_labeled = 200                     # Number of labeled samples
    base_lr = 0.01                        # Initial learning rate
    ema_decay = 0.99                      # EMA decay rate
    seg_consistency = 0.1                 # Segmentation consistency weight
    cls_consistency = 0.1                 # Classification consistency weight
```

#### Baseline Model Training

```bash
# Enter the baseline directory
cd baseline

# Run training
python train.py
```

**Note**: Make sure you have modified the `root_path` in the `Args` class to point to your actual data directory before running the training.

During training:

- Training logs are printed every 250 iterations
- Validation is performed every 500 iterations
- Checkpoints are saved every 3000 iterations
- The best Dice model and the best classification Score model are saved automatically
- Logs and TensorBoard records are saved under `result/experiment_name/`

### 3. Model Testing

#### Configure Testing Parameters

Edit the `Args` class in `baseline/test.py`:

```python
class Args:
    root_path = "/path/to/data"            # Path to test data
    model_path = "/path/to/model.pth"      # Path to model weights
    batch_size = 1                         # Test batch size
    save_vis = "/path/to/save/results"     # Directory to save results
    num_vis_cases = 5                      # Number of cases to visualize
```

#### Run Testing

```bash
python test.py
```

### 4. About model.py ⭐

`model.py` is the core file. You can freely modify its internal implementation, but you must adhere to the following interface specification:

#### Required Interface

```python
class Model(nn.Module):
    def __init__(self, in_chns=1, seg_classes=2, cls_classes=2):
        """Initialize model"""
        pass
    
    def forward(self, x_long, x_trans):
        """
        Perform inference
        
        Args:
            data_root: Input data root directory (/input/ in Docker)
            output_dir: Output directory (/output/ in Docker)
            batch_size: Batch size
        """
        pass
```

### 5. About run_infer.py ⭐

`run_infer.py` is the core script for testing within the Docker submission environment. Please do not modify its arguments or interface; otherwise, the evaluation procedure will fail.

## Submission

### Prerequisites

Before submission, please ensure:
- Docker is installed on your system ([Install Docker](https://docs.docker.com/get-docker/))
- You have a Docker Hub account ([Register here](https://hub.docker.com/signup))
- You have logged in with `docker login`

#### What to Submit

- Name your model file `model.py` and place it in `docker_submission_template/models/`
- Name your weight file `best_model.pth` and place it in `docker_submission_template/weights/`

### Step 1: Build Docker Image

```bash
# Enter the docker_submission_template directory
cd docker_submission_template

# Build the Docker image
docker build -t my-submission:latest .

# Verify the build was successful
docker images | grep my-submission
```

**Tip**: Building may take several minutes. If build fails, check the Dockerfile and requirements.txt.

### Step 2: Test Locally (Recommended)

Before submitting, test your Docker image locally to ensure it works correctly:

```bash
# Run the container with test data
docker run --rm \
  -v /path/to/test/data:/data \
  -v /path/to/output:/output \
  my-submission:latest

# Check the output results
ls /path/to/output/preds
```

**Expected Output Structure:**
```
/path/to/output/preds/
├── case_001.h5
├── case_002.h5
└── ...
```

Each output `.h5` file should contain:
- `pred_long`: Longitudinal view segmentation prediction (H×W, uint8, values: 0=background, 1=vessel, 2=plaque)
- `pred_trans`: Transverse view segmentation prediction (H×W, uint8, values: 0=background, 1=vessel, 2=plaque)
- `cls_pred`: Classification prediction (scalar, uint8, values: 0=RADS2, 1=RADS3-4)

**Note**: 
- Input data directory should be mounted to `/data`
- Output predictions will be saved to `/output/preds/` inside the container
- [预期输出内容的详细说明待补充]

### Step 3: Push to Docker Hub

```bash
# Login to Docker Hub
docker login

# Tag your image (replace YOUR_USERNAME with your Docker Hub username)
docker tag my-submission:latest YOUR_USERNAME/my-submission:latest

# Push to Docker Hub
docker push YOUR_USERNAME/my-submission:latest
```

**Note**: 
- Pushing may take 10-30 minutes depending on image size and network speed
- The image should be kept under 5GB for faster evaluation
- From mainland China, you may need to configure a proxy or mirror

### Step 4: Set Image Visibility

After pushing, ensure your image is publicly accessible:
1. Go to [Docker Hub](https://hub.docker.com/)
2. Navigate to your repository
3. Go to **Settings** → Set visibility to **Public**

### Step 5: Submit on Platform

Submit your Docker image link on the **CSV Challenge Platform**:
- Go to the CSV Challenge Platform submission page
- Submit your Docker image link in the format: `YOUR_USERNAME/my-submission:latest`

### Verification

To verify your submission was successful, you can pull your image from Docker Hub:

```bash
docker pull YOUR_USERNAME/my-submission:latest
```

If you encounter any issues, please contact: csv2026_challenge@163.com

## FAQ

### Q1: Docker image push failed or very slow?

Answer:
- Check your network connection
- From mainland China, configure Docker mirror acceleration (e.g., Alibaba Cloud, Tencent Cloud mirrors)
- Use a VPN if necessary
- If push timeout occurs, try `docker push` again (it will resume from where it stopped)

### Q2: How to reduce Docker image size?

Answer:
- Remove unnecessary dependencies from requirements.txt
- Use multi-stage builds in Dockerfile
- Clean up cache and temporary files in Dockerfile

### Q3: How to debug if the Docker container fails during evaluation?

Answer:
- Test locally first using the command in Step 2
- Check container logs: `docker logs <container_id>`
- Ensure your model.py and weights are correctly placed
- Verify run_infer.py interface hasn't been modified
