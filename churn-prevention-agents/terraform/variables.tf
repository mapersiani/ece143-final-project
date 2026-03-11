variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "image_name" {
  description = "Docker image name in Artifact Registry"
  type        = string
  default     = "churn-prevention-app"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "db_password" {
  description = "Cloud SQL postgres password"
  type        = string
  sensitive   = true
}

variable "google_api_key" {
  description = "Gemini API key"
  type        = string
  sensitive   = true
}

variable "max_debate_rounds" {
  description = "Max debate rounds before escalation"
  type        = number
  default     = 5
}

variable "consensus_threshold" {
  description = "Critic rating threshold for approval"
  type        = number
  default     = 7
}
