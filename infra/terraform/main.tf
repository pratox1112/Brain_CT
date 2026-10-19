# Phase 8: the smallest GKE setup for the BrainCT API.
#   - GKE Autopilot cluster: Google manages the nodes; billing follows the pods' CPU/memory
#     requests. The free tier covers the management fee of one Autopilot cluster per account.
#   - Artifact Registry repository for the Docker image.
#
# Cost: see docs/results.md (Phase 8). Destroy after every session:  terraform destroy

resource "google_project_service" "apis" {
  for_each = toset([
    "container.googleapis.com",
    "artifactregistry.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false # leave the APIs enabled; turning them off can break other things
}

resource "google_artifact_registry_repository" "images" {
  location      = var.region
  repository_id = var.repository_id
  format        = "DOCKER"
  description   = "BrainCT API images (research use only)"

  depends_on = [google_project_service.apis]
}

# Permissions for the account the cluster's nodes run as. Autopilot nodes use the project's
# Compute Engine default service account, which newer projects no longer give broad roles.
# Without these, image pulls fail with "403 Forbidden" (seen in session 1, 2026-10-19).
data "google_project" "this" {}

locals {
  node_service_account = "serviceAccount:${data.google_project.this.number}-compute@developer.gserviceaccount.com"
}

# Read-only access to this one repository, so nodes can pull the API image.
resource "google_artifact_registry_repository_iam_member" "nodes_pull_images" {
  project    = var.project_id
  location   = google_artifact_registry_repository.images.location
  repository = google_artifact_registry_repository.images.name
  role       = "roles/artifactregistry.reader"
  member     = local.node_service_account
}

# GKE's recommended minimum role for node service accounts: writing logs and metrics
# (needed for Phase 9 monitoring) and other node basics.
resource "google_project_iam_member" "nodes_default_role" {
  project = var.project_id
  role    = "roles/container.defaultNodeServiceAccount"
  member  = local.node_service_account

  depends_on = [google_project_service.apis]
}

resource "google_container_cluster" "autopilot" {
  name             = var.cluster_name
  location         = var.region
  enable_autopilot = true

  release_channel {
    channel = "REGULAR"
  }

  # Allow `terraform destroy` to delete the cluster (the provider protects it by default).
  deletion_protection = false

  depends_on = [google_project_service.apis]
}
