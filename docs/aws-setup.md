# Deploying Eataly AI to AWS (current architecture)

This guide deploys the **existing** Streamlit app to AWS without changing how it works. When the container starts it downloads the PDFs from a private S3 bucket and builds the in-memory Chroma index, as it does today on Streamlit Cloud. Later phases replace that with a pre-built, versioned index.

```
Staff browser ──HTTPS──▶ Application Load Balancer ──▶ ECS Fargate task (Docker: Streamlit app)
                                                          │   ├── reads PDFs from S3 (private)
                                                          │   ├── reads API keys from Secrets Manager
                                                          │   └── writes logs to CloudWatch
GitHub (push to main) ──GitHub Actions──▶ ECR (image tagged with Git SHA) ──▶ ECS redeploy
```

Region used throughout: **`ca-central-1` (Canada Central)**. If you pick another region, change it everywhere, including `.github/workflows/deploy.yml`.

Names used throughout (keep them unless you have a reason to change them; the workflow file expects them):

| Thing | Name |
|---|---|
| S3 bucket | `eataly-ai-<your-initials>-<random>` (bucket names are globally unique) |
| Secret | `eataly-ai/app` |
| ECR repository | `eataly-ai` |
| ECS cluster / service / task family / container | `eataly-ai` |
| CloudWatch log group | `/ecs/eataly-ai` |

Rough cost while running: about **$60–80 USD/month** (1 Fargate task at 1 vCPU / 2 GB, 1 load balancer, public IPv4 addresses, a secret, logs). OpenAI usage is extra. Check numbers in the [AWS Pricing Calculator](https://calculator.aws/). Tear everything down when you don't need it (see the end of this guide).

---

## Step 0: Account safety (do this first)

1. **Turn on MFA for the root user**: AWS console, then your account name, then *Security credentials*.
2. **Create a day-to-day admin login.** Don't use root for daily work. The simplest option is **IAM Identity Center**: enable it, create a user, and give it the `AdministratorAccess` permission set.
3. **Set a budget alert**: open *Billing → Budgets → Create budget → Monthly cost budget*, choose e.g. $50, and enter your email. This only alerts you; it doesn't stop spending.
4. **Install the AWS CLI** on your Mac and log in:
   ```bash
   brew install awscli
   aws configure sso          # if you used IAM Identity Center
   aws sts get-caller-identity  # should print your account ID
   ```
   Note your **12-digit account ID**. It appears below as `ACCOUNT_ID`.

---

## Step 1: Private S3 bucket for the PDFs

```bash
BUCKET=eataly-ai-hm-$RANDOM            # pick once, write it down
aws s3api create-bucket --bucket $BUCKET --region ca-central-1 \
  --create-bucket-configuration LocationConstraint=ca-central-1
aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-versioning --bucket $BUCKET --versioning-configuration Status=Enabled

# Upload the knowledge base
aws s3 sync eataly_ai_knowledge_base/ s3://$BUCKET/documents/
```

Layout used now and in later phases:

```
s3://BUCKET/documents/   approved PDFs (now)
s3://BUCKET/feedback/    feedback events (Phase 4)
s3://BUCKET/indexes/     versioned Chroma indexes (Phase 3)
```

---

## Step 2: Store API keys in Secrets Manager

Console: *Secrets Manager → Store a new secret → Other type of secret*. Add key/value pairs:

| Key | Value |
|---|---|
| `OPENAI_API_KEY` | your OpenAI key |
| `LANGSMITH_API_KEY` | your LangSmith key |

Name it `eataly-ai/app`. Copy the **secret ARN** (it ends in a random 6-character suffix such as `-AbC123`).

> The Google Sheets query log reads `st.secrets`, which isn't available in the container, so it will be skipped and logged to CloudWatch. That's expected. It is being replaced by S3 feedback events.

---

## Step 3: ECR repository (where Docker images live)

```bash
aws ecr create-repository --repository-name eataly-ai --region ca-central-1 \
  --image-tag-mutability IMMUTABLE --image-scanning-configuration scanOnPush=true
```

Push a first image by hand, so ECS has something to run before GitHub Actions is set up:

```bash
REGISTRY=ACCOUNT_ID.dkr.ecr.ca-central-1.amazonaws.com
aws ecr get-login-password --region ca-central-1 | docker login --username AWS --password-stdin $REGISTRY
docker build --platform linux/amd64 -t $REGISTRY/eataly-ai:manual-1 .
docker push $REGISTRY/eataly-ai:manual-1
```

(`--platform linux/amd64` is required on Apple-silicon Macs. Without it the image is built for ARM and won't start on standard Fargate.)

---

## Step 4: IAM roles for the container

ECS uses two roles:

**a) Task execution role:** used by ECS itself to pull the image, write logs and fetch secrets.

Console: *IAM → Roles → Create role → AWS service → Elastic Container Service → Elastic Container Service Task*. Attach `AmazonECSTaskExecutionRolePolicy`. Name it `eataly-ai-execution-role`. Then add an inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "secretsmanager:GetSecretValue",
    "Resource": "arn:aws:secretsmanager:ca-central-1:ACCOUNT_ID:secret:eataly-ai/app-*"
  }]
}
```

**b) Task role:** used by *your code* (boto3) and limited to the S3 prefixes it needs.

Same trusted service (*Elastic Container Service Task*), no managed policies. Name it `eataly-ai-task-role`. Inline policy:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    { "Effect": "Allow", "Action": "s3:ListBucket", "Resource": "arn:aws:s3:::BUCKET",
      "Condition": { "StringLike": { "s3:prefix": ["documents/*", "indexes/*"] } } },
    { "Effect": "Allow", "Action": "s3:GetObject",
      "Resource": ["arn:aws:s3:::BUCKET/documents/*", "arn:aws:s3:::BUCKET/indexes/*"] },
    { "Effect": "Allow", "Action": "s3:PutObject", "Resource": "arn:aws:s3:::BUCKET/feedback/*" }
  ]
}
```

---

## Step 5: Log group, cluster and task definition

```bash
aws logs create-log-group --log-group-name /ecs/eataly-ai --region ca-central-1
aws logs put-retention-policy --log-group-name /ecs/eataly-ai --retention-in-days 30 --region ca-central-1
aws ecs create-cluster --cluster-name eataly-ai --region ca-central-1
```

Save this as `task-definition.json` (outside the repo, or delete it afterwards) and fill in the placeholders:

```json
{
  "family": "eataly-ai",
  "requiresCompatibilities": ["FARGATE"],
  "networkMode": "awsvpc",
  "cpu": "1024",
  "memory": "2048",
  "runtimePlatform": { "cpuArchitecture": "X86_64", "operatingSystemFamily": "LINUX" },
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/eataly-ai-execution-role",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/eataly-ai-task-role",
  "containerDefinitions": [{
    "name": "eataly-ai",
    "image": "ACCOUNT_ID.dkr.ecr.ca-central-1.amazonaws.com/eataly-ai:manual-1",
    "essential": true,
    "portMappings": [{ "containerPort": 8501, "protocol": "tcp" }],
    "environment": [
      { "name": "KNOWLEDGE_BASE_S3_URI", "value": "s3://BUCKET/documents/" },
      { "name": "AWS_REGION", "value": "ca-central-1" },
      { "name": "LANGSMITH_TRACING", "value": "true" },
      { "name": "LANGSMITH_PROJECT", "value": "eataly-ai" }
    ],
    "secrets": [
      { "name": "OPENAI_API_KEY",    "valueFrom": "SECRET_ARN:OPENAI_API_KEY::" },
      { "name": "LANGSMITH_API_KEY", "valueFrom": "SECRET_ARN:LANGSMITH_API_KEY::" }
    ],
    "logConfiguration": {
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/eataly-ai",
        "awslogs-region": "ca-central-1",
        "awslogs-stream-prefix": "app"
      }
    }
  }]
}
```

```bash
aws ecs register-task-definition --cli-input-json file://task-definition.json --region ca-central-1
```

---

## Step 6: Networking, load balancer and service (console is easiest)

Use the **default VPC** and its public subnets. That avoids a NAT gateway, which costs about $35/month.

1. **Security groups** (*EC2 → Security Groups*):
   - `eataly-ai-alb-sg`: inbound HTTP 80 and HTTPS 443 from `0.0.0.0/0`.
   - `eataly-ai-task-sg`: inbound TCP 8501 **only from `eataly-ai-alb-sg`**.
2. **Target group** (*EC2 → Target Groups → Create*):
   - Target type **IP addresses**, protocol HTTP, port 8501, default VPC.
   - Health check path: `/_stcore/health`.
   - Don't register any targets; ECS does that.
   - Name it `eataly-ai-tg`.
3. **Load balancer** (*EC2 → Load Balancers → Create → Application Load Balancer*):
   - Internet-facing, at least 2 public subnets, security group `eataly-ai-alb-sg`.
   - Listener HTTP:80, forwarding to `eataly-ai-tg`.
   - After creation, *Attributes → Idle timeout → 300 seconds*. Streamlit keeps a websocket open, and the default 60 s can drop sessions.
4. **ECS service** (*ECS → Clusters → eataly-ai → Services → Create*):
   - Launch type **Fargate**, task definition `eataly-ai` (latest revision).
   - Service name `eataly-ai`, desired tasks **1**.
   - Networking: default VPC, public subnets, security group `eataly-ai-task-sg`, **Public IP: ON** (needed to reach OpenAI/S3/ECR without a NAT gateway).
   - Load balancing: choose the existing ALB and target group, container `eataly-ai:8501`.
   - Health check grace period: 120 s.

After a few minutes the target should show **healthy**. Open the ALB's DNS name (`http://eataly-ai-…elb.amazonaws.com`). The **first question takes a while** because the app downloads and embeds all PDFs on the first request. Later ones are fast.

Troubleshooting: *ECS → service → Tasks → (task) → Logs*, or CloudWatch log group `/ecs/eataly-ai`.

### HTTPS and access control (decide before sharing the link)

- **HTTPS needs a domain name.** Request a certificate in *Certificate Manager (ACM)* for e.g. `ai.yourdomain.com`, validate it via DNS, add an HTTPS:443 listener on the ALB using it, and change the HTTP:80 listener to *redirect to HTTPS*. Then point a DNS CNAME at the ALB.
- **The app has no login.** Anyone with the URL can query internal HR/benefits material. The lowest-effort fix needs no code: once HTTPS is on, add an **"Authenticate (Amazon Cognito)"** action to the HTTPS listener, so users must sign in before the ALB forwards them to the app.

---

## Step 7: Automatic deploys from GitHub

1. **Create a private GitHub repo** and push this project. `.gitignore` already excludes the PDFs, `.env` and secrets. Check `git status` before the first commit and make sure no PDF is listed.
2. **Let GitHub log in to AWS without stored keys (OIDC):**
   - *IAM → Identity providers → Add provider → OpenID Connect*.
     - Provider URL: `https://token.actions.githubusercontent.com`
     - Audience: `sts.amazonaws.com`
   - *IAM → Roles → Create role → Web identity*, choose that provider and audience. Restrict it to your repo: GitHub organization = your username, repository = your repo, branch = `main`. Name it `eataly-ai-github-deploy`.
   - Inline policy:
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         { "Effect": "Allow", "Action": "ecr:GetAuthorizationToken", "Resource": "*" },
         { "Effect": "Allow",
           "Action": ["ecr:BatchCheckLayerAvailability", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart",
                      "ecr:CompleteLayerUpload", "ecr:PutImage", "ecr:BatchGetImage"],
           "Resource": "arn:aws:ecr:ca-central-1:ACCOUNT_ID:repository/eataly-ai" },
         { "Effect": "Allow",
           "Action": ["ecs:DescribeTaskDefinition", "ecs:RegisterTaskDefinition"], "Resource": "*" },
         { "Effect": "Allow", "Action": ["ecs:UpdateService", "ecs:DescribeServices"],
           "Resource": "arn:aws:ecs:ca-central-1:ACCOUNT_ID:service/eataly-ai/eataly-ai" },
         { "Effect": "Allow", "Action": "iam:PassRole",
           "Resource": ["arn:aws:iam::ACCOUNT_ID:role/eataly-ai-execution-role",
                        "arn:aws:iam::ACCOUNT_ID:role/eataly-ai-task-role"] }
       ]
     }
     ```
3. **Add the role ARN as a GitHub secret.** In the repo, go to *Settings → Secrets and variables → Actions → New repository secret*:
   - Name: `AWS_DEPLOY_ROLE_ARN`
   - Value: `arn:aws:iam::ACCOUNT_ID:role/eataly-ai-github-deploy`

Every push to `main` now runs `.github/workflows/deploy.yml`. It builds the image, tags it with the commit SHA, pushes it to ECR and rolls the ECS service onto it. You can also trigger it manually from the *Actions* tab.

---

## Step 8: Basic alarms (CloudWatch)

- *CloudWatch → Alarms → Create alarm*, metric `ApplicationELB → Per AppELB, per TG Metrics → UnHealthyHostCount` for `eataly-ai-tg`. Alarm when ≥ 1 for 5 minutes, with email notification.
- Optional: `HTTPCode_Target_5XX_Count` > 5 in 5 minutes.

Latency and request-level metrics come in a later phase, when the app writes structured logs.

---

## Smoke test after each deploy

1. The target group shows healthy.
2. The app loads over the ALB/domain URL.
3. A menu question gets a sensible answer.
4. An out-of-scope question ("is the lasagna available right now?") gets the manager-escalation response.
5. CloudWatch logs show no errors.

---

## Tearing down (to stop paying)

In this order:

1. ECS service: set desired tasks to 0, then delete it.
2. Load balancer, then target group.
3. ECS cluster.
4. Optional:
   - ECR images. Storage is cheap; the repo can stay.
   - The S3 bucket. It costs pennies, and you may want to keep the documents.
   - The secret ($0.40/month).

Recreating it later means repeating Step 6 only.
