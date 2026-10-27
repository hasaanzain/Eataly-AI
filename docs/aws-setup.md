# Eataly AI on AWS

How the app is deployed, how to update it, and how to stop paying for it.

```
Browser ──HTTP──▶ Application Load Balancer (open to everyone, no login)
                        │  health check: GET /api/health (200 only once the PDF index is built)
                        ▼
                  ECS Fargate task: FastAPI (backend/app.py) on port 8501
                        ├─ serves the UI (frontend/index.html) and /api/*
                        ├─ downloads PDFs from S3 documents/ and indexes them at startup
                        ├─ writes feedback to S3 feedback/YYYY/MM/DD/<uuid>.json
                        ├─ API keys injected from Secrets Manager (eataly-ai/app)
                        └─ logs to CloudWatch (/ecs/eataly-ai)

git push (main) ──▶ GitHub Actions ──▶ ECR image tagged with the commit SHA ──▶ ECS rolling deploy
```

Region: `ca-central-1`. AWS CLI profile: `hasaan` (sign in with `aws login --profile hasaan`).

## The two CloudFormation stacks

| Stack | Template | What it holds | Cost |
|---|---|---|---|
| `eataly-ai-foundation` | [infra/aws/foundation.yaml](../infra/aws/foundation.yaml) | Private S3 bucket, ECR repo, secret + KMS key, log group, container roles, GitHub deploy role (OIDC) | ~$1-2/month |
| `eataly-ai-app` | [infra/aws/app.yaml](../infra/aws/app.yaml) | ECS cluster/service/task definition, load balancer, security groups, health alarm | ~$60-80/month |

The foundation stack is long-lived. The app stack can be deleted to stop paying and recreated later.

Current values:

| Thing | Value |
|---|---|
| App URL | http://eataly-LoadB-Im6ruFOAYdDe-1295347466.ca-central-1.elb.amazonaws.com |
| Bucket | `eataly-ai-foundation-databucket-fs1wcpnicpyp` |
| Secret | `eataly-ai/app` (keys `OPENAI_API_KEY`, `LANGSMITH_API_KEY`) |
| ECR repo | `272423021667.dkr.ecr.ca-central-1.amazonaws.com/eataly-ai` |
| GitHub secret | `AWS_DEPLOY_ROLE_ARN` = the foundation stack's `GitHubDeployRoleArn` output |

## Everyday tasks

**Deploy code changes**: push to `main`. The workflow builds the image, pushes it to ECR, and rolls the service. The new task only receives traffic after `/api/health` returns 200 (PDFs indexed); if it never gets healthy, ECS rolls back automatically. Watch it under the repo's Actions tab.

**Update the PDFs**:

```bash
aws s3 sync eataly_ai_knowledge_base/ s3://eataly-ai-foundation-databucket-fs1wcpnicpyp/documents/ --delete --profile hasaan
aws ecs update-service --cluster eataly-ai --service eataly-ai --force-new-deployment --profile hasaan
```

The restarted task re-downloads and re-indexes everything before taking traffic (zero downtime). S3 versioning keeps replaced files for 90 days.

**Change API keys**: edit `eataly-ai/app` in Secrets Manager, then run the `--force-new-deployment` command above.

**Read feedback**:

```bash
aws s3 ls s3://eataly-ai-foundation-databucket-fs1wcpnicpyp/feedback/ --recursive --profile hasaan
```

Each file holds the question, answer, cited documents/pages, rating and conversation id. No names or personal data are collected.

**See logs/errors**: CloudWatch, log group `/ecs/eataly-ai`, or `aws logs tail /ecs/eataly-ai --follow --profile hasaan`.

## Changing the app stack

The deploy workflow registers new task-definition revisions outside CloudFormation. When updating `eataly-ai-app`, always pass the currently deployed image tag, or the stack rolls the app back to an older image:

```bash
TAG=$(aws ecs describe-task-definition --task-definition eataly-ai --query 'taskDefinition.containerDefinitions[0].image' --output text --profile hasaan | cut -d: -f2)
aws cloudformation deploy --stack-name eataly-ai-app --template-file infra/aws/app.yaml \
  --parameter-overrides ImageTag=$TAG --profile hasaan
```

Prefer creating a change set first and checking that nothing is replaced unexpectedly.

## Access and abuse protection

The app has no login and is open to everyone (owner decision; the documents are already public). Protection lives in the API:

- 20 chat questions per visitor (IP) per 5 minutes (`CHAT_LIMIT`)
- 2000 chat questions per day in total (`DAILY_CHAT_LIMIT`)
- Input size limits on every request

Also set a monthly spending limit in the OpenAI dashboard. To restrict access again, update the stack's `AllowedCidr` parameter (e.g. `x.x.x.x/32`). Adding a real login (Cognito) needs HTTPS, which needs a domain name.

## Stop paying / start again

- **Stop**: CloudFormation console, delete stack `eataly-ai-app`. Documents, keys and images stay in the foundation stack.
- **Start**: create `eataly-ai-app` again from `infra/aws/app.yaml` with `VpcId=vpc-09b93fdac5449896d`, `SubnetIds=subnet-0b8ebdebbb1e54b91,subnet-073a2169225cd79ee,subnet-00e2e4e59dd49b971`, `AllowedCidr=0.0.0.0/0` and `ImageTag=<latest commit SHA in ECR>`. The URL will change.
