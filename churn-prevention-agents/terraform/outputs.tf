output "app_url" {
  description = "Cloud Run app URL"
  value       = google_cloud_run_v2_service.app.uri
}

output "mlflow_url" {
  description = "MLFlow server URL"
  value       = google_cloud_run_v2_service.mlflow.uri
}

output "postgres_ip" {
  description = "Cloud SQL public IP"
  value       = google_sql_database_instance.main.public_ip_address
}

output "artifact_bucket" {
  description = "GCS bucket for MLFlow artifacts"
  value       = google_storage_bucket.mlflow_artifacts.name
}

output "docker_push_command" {
  description = "Command to push Docker image to Artifact Registry"
  value       = "docker push ${var.region}-docker.pkg.dev/${var.project_id}/churn-prevention/${var.image_name}:${var.image_tag}"
}
