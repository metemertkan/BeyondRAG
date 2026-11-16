# The deployer: lints and deploys the CloudFormation stacks, reads them back and runs the drills.
FROM public.ecr.aws/docker/library/python:3.12-slim
RUN pip install --no-cache-dir "boto3==1.40.*" "cfn-lint==1.*" "pyyaml==6.*"
WORKDIR /infra
ENTRYPOINT ["python", "/infra/scripts/infra.py"]
