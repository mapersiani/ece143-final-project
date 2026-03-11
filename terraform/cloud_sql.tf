# Cloud SQL PostgreSQL instance
resource "google_sql_database_instance" "main" {
  name             = "churn-postgres"
  database_version = "POSTGRES_15"
  region           = var.region

  settings {
    tier              = "db-f1-micro"  # smallest tier for course project
    availability_type = "ZONAL"
    disk_size         = 10
    disk_type         = "PD_SSD"

    backup_configuration {
      enabled = false  # disable for cost savings
    }

    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        value = "0.0.0.0/0"  # restrict in production
        name  = "all"
      }
    }
  }

  deletion_protection = false
  depends_on          = [google_project_service.services]
}

resource "google_sql_database" "churn_db" {
  name     = "churn_db"
  instance = google_sql_database_instance.main.name
}

resource "google_sql_user" "app_user" {
  name     = "churn_user"
  instance = google_sql_database_instance.main.name
  password = var.db_password
}

# Cloud Storage bucket for MLFlow artifacts
resource "google_storage_bucket" "mlflow_artifacts" {
  name          = "${var.project_id}-mlflow-artifacts"
  location      = var.region
  force_destroy = true

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }
  }
}

# Store secrets in Secret Manager
resource "google_secret_manager_secret" "db_password" {
  secret_id  = "churn-db-password"
  depends_on = [google_project_service.services]
  replication { auto {} }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = var.db_password
}

resource "google_secret_manager_secret" "google_api_key" {
  secret_id  = "churn-google-api-key"
  depends_on = [google_project_service.services]
  replication { auto {} }
}

resource "google_secret_manager_secret_version" "google_api_key" {
  secret      = google_secret_manager_secret.google_api_key.id
  secret_data = var.google_api_key
}
