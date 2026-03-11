PROJECT_ID ?= your-gcp-project-id
REGION     ?= us-central1
IMAGE_NAME  = churn-prevention-app
IMAGE_TAG  ?= latest
REPO        = $(REGION)-docker.pkg.dev/$(PROJECT_ID)/churn-prevention/$(IMAGE_NAME)

# ── Local development ─────────────────────────────────────────────────────────

up:
	cd docker && docker compose up --build

down:
	cd docker && docker compose down

logs:
	cd docker && docker compose logs -f app

test:
	pytest tests/ -v

# Log a mock MLFlow run (useful to verify MLFlow UI)
mlflow-log:
	cd docker && docker compose exec app python -m app.ml.train

# ── GCP deployment ────────────────────────────────────────────────────────────

# Step 1: Authenticate Docker with Artifact Registry
docker-auth:
	gcloud auth configure-docker $(REGION)-docker.pkg.dev

# Step 2: Build and push the image
docker-push:
	docker build -t $(REPO):$(IMAGE_TAG) -f docker/Dockerfile .
	docker push $(REPO):$(IMAGE_TAG)

# Step 3: Deploy infrastructure with Terraform
tf-init:
	cd terraform && terraform init

tf-plan:
	cd terraform && terraform plan -var="project_id=$(PROJECT_ID)"

tf-apply:
	cd terraform && terraform apply -var="project_id=$(PROJECT_ID)" -auto-approve

tf-destroy:
	cd terraform && terraform destroy -var="project_id=$(PROJECT_ID)" -auto-approve

# Full GCP deploy in one shot
deploy: docker-push tf-apply
