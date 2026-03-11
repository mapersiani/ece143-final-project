locals {
  image_uri = "${var.region}-docker.pkg.dev/${var.project_id}/churn-prevention/${var.image_name}:${var.image_tag}"
  db_url    = "postgresql://churn_user:${var.db_password}@${google_sql_database_instance.main.public_ip_address}:5432/churn_db"
}

# MLFlow Cloud Run service
resource "google_cloud_run_v2_service" "mlflow" {
  name     = "churn-mlflow"
  location = var.region

  template {
    service_account = google_service_account.app_sa.email

    containers {
      image = "python:3.11-slim"
      command = [
        "bash", "-c",
        "pip install mlflow psycopg2-binary --quiet && mlflow server --host 0.0.0.0 --port 8080 --backend-store-uri ${local.db_url} --default-artifact-root gs://${google_storage_bucket.mlflow_artifacts.name}/artifacts"
      ]
      ports { container_port = 8080 }

      env {
        name  = "PORT"
        value = "8080"
      }

      resources {
        limits = { cpu = "1", memory = "512Mi" }
      }
    }
  }

  depends_on = [google_project_service.services]
}

# Main app Cloud Run service
resource "google_cloud_run_v2_service" "app" {
  name     = "churn-prevention-app"
  location = var.region

  template {
    service_account = google_service_account.app_sa.email

    containers {
      image = local.image_uri

      env {
        name  = "DATABASE_URL"
        value = local.db_url
      }
      env {
        name  = "MLFLOW_TRACKING_URI"
        value = google_cloud_run_v2_service.mlflow.uri
      }
      env {
        name = "GOOGLE_API_KEY"
        value_source {
          secret_key_ref {
            secret  = google_secret_manager_secret.google_api_key.secret_id
            version = "latest"
          }
        }
      }
      env {
        name  = "MAX_DEBATE_ROUNDS"
        value = tostring(var.max_debate_rounds)
      }
      env {
        name  = "CONSENSUS_THRESHOLD"
        value = tostring(var.consensus_threshold)
      }

      ports { container_port = 8000 }

      resources {
        limits = { cpu = "2", memory = "2Gi" }
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_sql_database_instance.main,
    google_cloud_run_v2_service.mlflow,
  ]
}

# Allow unauthenticated access (remove for production)
resource "google_cloud_run_v2_service_iam_member" "app_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.app.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}
