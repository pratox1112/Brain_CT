output "cluster_name" {
  value = google_container_cluster.autopilot.name
}

output "region" {
  value = var.region
}

output "image_repository" {
  description = "Use as image.repository in infra/helm/values-gke.yaml (without the tag)."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.images.repository_id}/brainct-api"
}

output "get_credentials" {
  description = "Points kubectl at the new cluster."
  value       = "gcloud container clusters get-credentials ${google_container_cluster.autopilot.name} --region ${var.region} --project ${var.project_id}"
}

output "configure_docker" {
  description = "Lets docker push to the registry."
  value       = "gcloud auth configure-docker ${var.region}-docker.pkg.dev"
}
