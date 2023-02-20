FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime

WORKDIR /app

COPY requirements.txt requirements.txt

# Remove torch from requirements as it is already installed within the base Docker image
RUN sed -i '/^torch/d' requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV WANDB_API_KEY=<YOUR_API_KEY_HERE>

ENTRYPOINT ["python", "-m", "scripts.fit_and_evaluate"]