# Running BeyondRAG on AWS

Everything BeyondRAG needs on AWS, as CloudFormation, and a compose file to check it on your own
machine with the [AWS Emulator](https://github.com/aws/aws-emulator), AWS's local emulator
released in November 2026. No AWS account needed to try it.

```bash
docker compose -f infra/docker-compose.aws.yml up --build --exit-code-from deployer
```

About a minute later the infrastructure has been deployed into the emulator and checked, and
`infra/results/` says what happened.

The infrastructure is deployed **empty**: every service exists, nothing runs yet. Putting the
app into it (building the image, pushing it, rolling it out) is the job of CI/CD, a follow-up.

## What runs where

| Runs locally (docker-compose) | Runs on AWS |
| :--- | :--- |
| `beyondrag-app` | ECS on Fargate behind an Application Load Balancer, with auto scaling; images in ECR |
| Postgres | RDS for PostgreSQL 16, its password managed in Secrets Manager |
| Elasticsearch | Amazon OpenSearch Service 2 |
| Redis | ElastiCache for Valkey 8 |
| Grafana | Amazon Managed Grafana, reading CloudWatch |
| `workspace/uploads` | S3 |
| Ollama | not included; `ModelEndpoint` points the app at a model |

## The templates (`cloudformation/`)

Four stacks, deployed in this order, each exporting what the next one needs:

1. **`network.yaml`**: a VPC across two Availability Zones. Public subnets for the load balancer, private subnets for the app (internet only through NAT), isolated subnets for the data stores (no route out). Each security group accepts only the tier in front of it: internet → load balancer → app → data stores.
2. **`data.yaml`**: one encryption key; PostgreSQL, OpenSearch and Valkey in the isolated subnets, with a standby in a second zone in production; the uploads bucket (versioned, private, TLS only); AWS Backup daily, kept 35 days, copied to the recovery Region.
3. **`observability.yaml`**: the app's logs, an alarm topic, Grafana and its read-only role.
4. **`app.yaml`**: the image repository, the ECS cluster and task definition (settings from the other stacks, passwords from Secrets Manager), the load balancer, the service, scaling on CPU (60%) and on requests per task (50 a minute), and alarms (server errors, slow answers, unhealthy tasks, database CPU, search cluster health).

The app stack has two switches:

- **`ServiceState`**: `Empty` creates everything and runs no tasks; `Live` runs the service.
- **`Capacity`**: `Small` runs 1 to 3 tasks, `Production` 2 to 10.

`params/` holds three environments: `dev`, `prod` (us-east-1) and `dr` (us-west-2), a small, empty copy of production that takes over if us-east-1 is lost.

## What the compose file does

Two emulators stand in for the two Regions. A deployer container then:

| Step | What it does |
| :--- | :--- |
| `lint` | checks every template with cfn-lint |
| `deploy` | deploys the four stacks into each Region through change sets, as a pipeline would |
| `verify` | reads each resource back from its service and checks it against the design (48 checks) |
| `inventory` | lists every resource the stacks created and checks it exists |
| `drills` | goes live, scales on high CPU, redeploys with no change, detects manual changes, and brings the recovery Region to production size |

To look around yourself, keep the emulators running and point the AWS CLI at them:

```bash
docker compose -f infra/docker-compose.aws.yml up -d aws-emulator aws-emulator-dr
docker compose -f infra/docker-compose.aws.yml run --rm deployer deploy prod
eval "$(infra/scripts/aws-env.sh primary)"   # or: dr
aws cloudformation describe-stacks --query "Stacks[].[StackName,StackStatus]"
aws ecs describe-services --cluster beyondrag-prod --services app
```

## Results

- The whole infrastructure in about **10 seconds**: 69 resources in us-east-1, 67 in us-west-2.
- **42 of 48 checks pass** in each Region.
- Going live changes exactly **2 resources**, the service and its scaling target.
- CPU at 90% against the 60% target took the service from **2 to 3 tasks**.
- The recovery Region reached production size in about **2 seconds**.

The emulator is new, and a few services aren't fully covered yet, which is where the remaining
checks stop. Where a template needed to work around that, it says so in a comment, and
`results/findings_first_runs.json` lists the details.

## Before the app runs on AWS

These are app changes, for the CI/CD follow-up:

- **Search**: OpenSearch needs a search adapter in the app, because BeyondRAG uses Elasticsearch 8 features (its client, `int8_hnsw`, RRF).
- **Database**: the app reads `POSTGRES_URI`. The task gets the parts instead (`POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`), so the app needs to build the URI from them.
- **Uploads**: the app writes to the local workspace today; `RAG_UPLOADS_BUCKET` is passed for the move to S3.
- **Grafana sign-in** needs IAM Identity Center or a SAML provider in the account.
- **HTTPS**: set `CertificateArn` to serve HTTPS and redirect HTTP to it.
