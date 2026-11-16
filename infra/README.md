# BeyondRAG on AWS: the infrastructure, as code

CloudFormation for everything BeyondRAG needs on AWS, deployed **empty** and verified on the
[AWS Emulator](https://github.com/aws/aws-emulator) in Docker. No AWS account, no bill.
Filling the services (building the image, pushing it, rolling it out) is the CI/CD follow-up.

```bash
docker compose -f infra/docker-compose.aws.yml up --build --exit-code-from deployer
```

About a minute later `infra/results/` holds what happened. To look around yourself while the
emulators run (`up -d aws-emulator aws-emulator-dr`, then `run --rm deployer deploy prod`):

```bash
eval "$(infra/scripts/aws-env.sh primary)"   # us-east-1; use "dr" for us-west-2
aws cloudformation describe-stacks --query "Stacks[].[StackName,StackStatus]"
aws ecs describe-services --cluster beyondrag-prod --services app
```

## What runs where

| Local (docker-compose) | AWS (`cloudformation/`) |
| :--- | :--- |
| `beyondrag-app` | ECS on Fargate behind an Application Load Balancer, with Application Auto Scaling; images in ECR |
| Postgres (LangGraph state) | RDS for PostgreSQL 16, password managed by RDS in Secrets Manager |
| Elasticsearch 8.15 | Amazon OpenSearch Service 2.17 |
| Redis 8 | ElastiCache for Valkey 8.0 |
| Grafana (otel-lgtm) | Amazon Managed Grafana reading CloudWatch |
| `workspace/uploads` | S3 |
| Ollama | not provisioned; `ModelEndpoint` is a parameter |

Four stacks, deployed in order, each exporting what the next one needs:

1. **`network.yaml`**: VPC over two Availability Zones; public subnets for the load balancer, private subnets for the app (internet only through NAT), isolated subnets for the data stores (no route out); security groups that only accept the tier in front: internet → load balancer → app → data stores.
2. **`data.yaml`**: one KMS key; PostgreSQL, OpenSearch and Valkey in the isolated subnets, all encrypted, Multi-AZ in production; the uploads bucket (versioned, private, TLS only); AWS Backup daily, kept 35 days, copied to the recovery Region.
3. **`observability.yaml`**: the app's log group, an alarm topic, the Grafana workspace and its read-only role.
4. **`app.yaml`**: ECR repository, ECS cluster, task definition (settings from the other stacks, secrets from Secrets Manager), load balancer, service, auto scaling on CPU (60%) and on requests per task (50 a minute), alarms for 5xx, p95 latency, unhealthy tasks, database CPU and a red search cluster.

`ServiceState` is the switch between the two deploys: `Empty` creates all of it and runs no tasks (there is no image yet); `Live` runs the `Capacity` profile, `Small` (1 to 3 tasks) or `Production` (2 to 10).

`params/` holds three environments: `dev`, `prod` (us-east-1) and `dr`, a pilot-light copy in us-west-2, deployed small and empty, that a failover turns live.

## What the deployer checks (`scripts/infra.py`)

| Step | What it does |
| :--- | :--- |
| `lint` | cfn-lint on every template |
| `deploy dr`, `deploy prod` | every stack through a change set, the way a pipeline would |
| `verify` | 48 checks that read each resource back **from its service**, not from the template: the data tier has no route out, each security group accepts only the tier in front, PostgreSQL is Multi-AZ, encrypted and private, OpenSearch is zone-aware, the bucket is versioned and private, the backups are copied to us-west-2, the tasks get secrets from Secrets Manager and run in private subnets, the alarms notify the topic |
| `inventory` | for every resource CloudFormation reports as created, asks the owning service whether it exists |
| `drills` | go live; a CPU breach scales the service; a no-op redeploy; two changes made by hand and drift detection; the recovery Region's app tier brought to production size |

## Results (AWS Emulator 1.0.385)

- Lint: 0 findings. Empty deploy of all four stacks: **69 resources in 10.4 s** (us-east-1), 67 in us-west-2.
- Verify: **42 of 48 checks pass** in each Region; the other 6 are blocked by emulator gaps (below); 0 fail.
- Inventory: of 69 resources CloudFormation reported as created, **67 exist** in their services; the 2 ElastiCache resources do not.
- Go live: a change set of exactly 2 changes (service, scaling target), 2 tasks wanted, bounds 2 to 10.
- Scaling: CPU at 90% against the 60% target took the service from 2 to 3 tasks in 9 s; at 15% it stayed at 3, inside the 300 s scale-in cooldown. The metric is handed to the alarm by the drill: no task runs in the emulator, so CloudWatch has no CPU data of its own.
- No-op redeploy: no changes. Recovery Region to production size: 2.2 s.

`results/findings_first_runs.json` lists the 15 things the emulator rejected or revealed while the templates were written. cfn-lint passed every one of those templates.

## Emulator gaps, and what the templates do about them

The templates are written for AWS. Where the emulator differs, the workaround is one that is also valid on AWS, or it lives in the deployer, never in a resource's behaviour.

| Gap in the emulator (1.0.385) | Workaround |
| :--- | :--- |
| A value from a Number parameter or a mapping reaches the service as a string ("expected number, got string") | Literal numbers: port 8090, capacity profiles, search volume sizes chosen by conditions |
| ElastiCache resources in a template are reported as created but not created | None; `verify` and `inventory` report it. ElastiCache called directly works |
| `!GetAtt Database.MasterUserSecret.SecretArn` is empty | The app stack takes the secret ARN and the cache address as parameters; the deployer passes the data stack's outputs, or the secret RDS reports when the output is empty |
| `MasterUserSecret.KmsKeyId` ignored (the secret uses `aws/secretsmanager`) | None; `verify` reports it |
| An existing task definition cannot be updated | Going live leaves the task definition alone; new images are the CI/CD step |
| `ListBackupPlans` not implemented | `verify` reads the plan by its ID |
| No r7g OpenSearch instance types | `r6g.large.search` |
| Drift detection missed an ingress rule added by hand | None; reported as a drill result |
| ECS runs no images, the load balancer forwards only to servers the emulator runs | Not needed: the deploy is empty by design |

## Before this runs on AWS

- **OpenSearch is not a drop-in for Elasticsearch 8.** BeyondRAG uses the Elasticsearch 8 client, `int8_hnsw` and RRF; it needs a search adapter for OpenSearch (or Elasticsearch run elsewhere).
- **PostgreSQL settings.** The app reads `POSTGRES_URI`; the task gets `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`, `POSTGRES_USER` and `POSTGRES_PASSWORD`, so the app needs to build the URI from them.
- **Uploads** go to the local workspace today; `RAG_UPLOADS_BUCKET` is passed for the S3 change.
- **Valkey** works for the exact caches; the semantic cache's vector search (Redis 8 query engine, off by default) is untested on Valkey.
- **Grafana sign-in** needs IAM Identity Center (`AWS_SSO`) or a SAML provider in the account.
- **TLS**: pass `CertificateArn` to get HTTPS with a redirect from HTTP.
- Real AWS will find what the emulator cannot: IAM policies are not evaluated in the emulator, quotas are not enforced, and nothing here proves the app runs.
