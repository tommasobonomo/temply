# [Project Name] (a Temply project)

This project ...
It uses the [tommasobonomo/temply](https://github.com/tommasobonomo/temply) template for quick Deep Learning-based experimentation.

## Install

If you just want to play around with the code, maybe run a quick evaluation, but you do not want to start building Docker images and running full training sessions, you should first install the requirements: 
```bash
pip install -r requirements.txt
```

Otherwise, you can build a Docker image which takes care of running these steps in the build process:
```bash
docker build -t <project name> .
```

## Usage

Usage happens through the `fit_and_evaluate` script, located at `scripts/fit_and_evaluate.py`.
It is used both for training and evaluation and it can be easily configured through [Hydra](https://hydra.cc/docs/intro/) CLI options.

The Hydra configuration is stored as a Python object (more specifically `@dataclass`) named `Config` in the `settings.py` file.
The root configuration object (`Config`) holds options to control the script behaviour and some integrations (for example [Weights&Biases](https://wandb.ai/)).
It also holds two sub-configuration objects, `ModelConfig` and `TrainerConfig`.
The first holds fields that modify the model that is being used and related settings, while the second one will be deconstructed as arguments to initialise a `pytorch_lightning.Trainer`, so one should take care not to add fields that are not compatible with the `Trainer`'s initialisation arguments.

The options can be changed directly in the `settings.py` file or through [Hydra's CLI syntax](https://hydra.cc/docs/advanced/override_grammar/basic/#modifying-the-config-object).
For example, one could decide to fine-tune a different model than the default and not use Weights&Biases for training tracking with the following syntax:
```bash
export PYTHONPATH=. 
python scripts/fit_and_evaluate.py \
    model.transformer_model=bert-base-uncased \
    enable_wandb=false
```

The above script can be used also with the Docker image, just prepending `docker run <project name>`.

### Training on Google Cloud Platform

It's also very easy to run a custom training job provisioned on Google Cloud Platform's Vertex AI service. If you have the `gcloud` utility installed and configured, you just need to run:
```bash
export JOB_NAME=<project name>_$(date +%Y%m%d_%H%M%S)
```
```bash
gcloud ai custom-jobs create $JOB_NAME \
    --config=gcp_training_config.yaml
```

Make sure you have built and uploaded to GCP's Container Registery the correct Docker image, then pass that image's URI as `<DOCKER_IMAGE_URI>` in `gcp_training_config.yaml`.

For more options and configs, see [docs](https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create).