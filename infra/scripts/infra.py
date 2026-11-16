#!/usr/bin/env python3
"""Lint, deploy, verify and drill BeyondRAG's AWS infrastructure on the AWS Emulator.

  infra.py lint                 cfn-lint on every template
  infra.py deploy [env]         deploy the four stacks in order (env: prod, dr, dev)
  infra.py verify [env]         read every resource back and check it against the design
  infra.py drills               go live, scale, no-op update, drift, rebuild the recovery Region
  infra.py all                  lint, deploy dr, deploy prod, verify both, drills

Runs inside the deployer container (see docker-compose.aws.yml), which shares the emulators'
network namespace: us-east-1 on 127.0.0.1:9297 and us-west-2 on 127.0.0.1:9298. Every step
writes its findings to results/<step>.json.
"""

from __future__ import annotations

import configparser
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, WaiterError

INFRA = Path(__file__).resolve().parents[1]
TEMPLATES = INFRA / "cloudformation"
PARAMS = INFRA / "params"
RESULTS = INFRA / "results"
STACKS = ["network", "data", "observability", "app"]
EMULATORS = {
    "us-east-1": (os.environ.get("PRIMARY_HOME", "/emulator/primary"), 9297),
    "us-west-2": (os.environ.get("DR_HOME", "/emulator/dr"), 9298),
}


# --- connections ------------------------------------------------------------------------

class Region:
    """Clients for one emulator. One emulator stands for one Region."""

    def __init__(self, name: str) -> None:
        home, port = EMULATORS[name]
        folder = Path(home) / str(port)
        creds = configparser.ConfigParser()
        creds.read(folder / "emulator_credentials")
        section = creds[creds.sections()[0]]
        self.name = name
        self._session = boto3.Session(aws_access_key_id=section["aws_access_key_id"],
                                      aws_secret_access_key=section["aws_secret_access_key"], region_name=name)
        self._endpoint = f"https://127.0.0.1:{port}"
        self._ca = str(folder / "ca.pem")
        self._clients: Dict[str, Any] = {}

    def __call__(self, service: str) -> Any:
        if service not in self._clients:
            self._clients[service] = self._session.client(
                service, endpoint_url=self._endpoint, verify=self._ca,
                config=Config(retries={"max_attempts": 3}, inject_host_prefix=False))
        return self._clients[service]

    @property
    def account(self) -> str:
        return self("sts").get_caller_identity()["Account"]


def load_params(env: str) -> Dict[str, Any]:
    return json.loads((PARAMS / f"{env}.json").read_text())


def save(name: str, data: Any) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, indent=1, default=str) + "\n")


def say(msg: str) -> None:
    print(msg, flush=True)


def stack_name(env_name: str, stack: str) -> str:
    return f"beyondrag-{env_name}-{stack}"


# --- lint -------------------------------------------------------------------------------

def lint() -> Dict[str, Any]:
    out = {}
    for stack in STACKS:
        path = TEMPLATES / f"{stack}.yaml"
        run = subprocess.run(["cfn-lint", "--format", "json", str(path)], capture_output=True, text=True)
        findings = json.loads(run.stdout or "[]")
        out[stack] = [{"rule": f["Rule"]["Id"], "level": f["Level"], "message": f["Message"], "line": f["Location"]["Start"]["LineNumber"]}
                      for f in findings]
        say(f"lint {stack}: {len(findings)} finding(s)")
        for f in out[stack]:
            say(f"  {f['level']} {f['rule']} line {f['line']}: {f['message']}")
    save("lint", out)
    return out


# --- deploy -----------------------------------------------------------------------------

def _parameters(values: Dict[str, str], account: str) -> List[Dict[str, str]]:
    return [{"ParameterKey": k, "ParameterValue": str(v).replace("{account}", account)} for k, v in values.items()]


def _wait_change_set(cfn: Any, stack: str, change_set: str) -> Dict[str, Any]:
    while True:
        cs = cfn.describe_change_set(StackName=stack, ChangeSetName=change_set)
        if cs["Status"] in ("CREATE_COMPLETE", "FAILED"):
            return cs
        time.sleep(1)


def _wait_stack(cfn: Any, stack: str) -> str:
    while True:
        status = cfn.describe_stacks(StackName=stack)["Stacks"][0]["StackStatus"]
        if not status.endswith("_IN_PROGRESS"):
            return status
        time.sleep(1)


def apply(region: Region, env_name: str, stack: str, values: Dict[str, str], label: str = "") -> Dict[str, Any]:
    """Create or update one stack through a change set, the way a pipeline would."""
    cfn = region("cloudformation")
    name = stack_name(env_name, stack)
    try:
        status = cfn.describe_stacks(StackName=name)["Stacks"][0]["StackStatus"]
        kind = "UPDATE"
        if status == "ROLLBACK_COMPLETE":
            # A stack whose first create failed cannot be updated, only deleted and created again.
            say(f"  {name} is in ROLLBACK_COMPLETE from a failed create; deleting it first")
            cfn.delete_stack(StackName=name)
            while True:
                try:
                    if cfn.describe_stacks(StackName=name)["Stacks"][0]["StackStatus"] == "DELETE_COMPLETE":
                        break
                except ClientError:
                    break
                time.sleep(1)
            kind = "CREATE"
    except ClientError:
        kind = "CREATE"
    change_set = f"{label or kind.lower()}-{int(time.time() * 1000)}"
    t0 = time.time()
    cfn.create_change_set(StackName=name, ChangeSetName=change_set, ChangeSetType=kind,
                          TemplateBody=(TEMPLATES / f"{stack}.yaml").read_text(),
                          Parameters=_parameters(values, region.account),
                          Capabilities=["CAPABILITY_IAM", "CAPABILITY_NAMED_IAM"])
    cs = _wait_change_set(cfn, name, change_set)
    changes = [{"action": c["ResourceChange"]["Action"], "resource": c["ResourceChange"]["LogicalResourceId"],
                "type": c["ResourceChange"]["ResourceType"], "replacement": c["ResourceChange"].get("Replacement")}
               for c in cs.get("Changes", [])]
    result: Dict[str, Any] = {"stack": name, "kind": kind, "changes": changes}
    if cs["Status"] == "FAILED":
        result.update(status="NO_CHANGES" if "didn't contain changes" in cs.get("StatusReason", "") else "CHANGE_SET_FAILED",
                      reason=cs.get("StatusReason"), seconds=round(time.time() - t0, 1))
        return result
    cfn.execute_change_set(StackName=name, ChangeSetName=change_set)
    status = _wait_stack(cfn, name)
    result.update(status=status, seconds=round(time.time() - t0, 1),
                  resources=len(cfn.describe_stack_resources(StackName=name)["StackResources"]))
    if not status.endswith("COMPLETE") or "ROLLBACK" in status:
        events = cfn.describe_stack_events(StackName=name)["StackEvents"]
        result["failures"] = [f"{e['LogicalResourceId']} {e['ResourceStatus']}: {e.get('ResourceStatusReason', '')}" for e in events
                              if "FAILED" in e["ResourceStatus"] or (e.get("ResourceStatusReason") and "ROLLBACK" in e["ResourceStatus"])][:10]
    return result


def deploy(env: str) -> List[Dict[str, Any]]:
    cfg = load_params(env)
    region = Region(cfg["region"])
    out = []
    t0 = time.time()
    workarounds: List[Dict[str, str]] = []
    for stack in STACKS:
        values = dict(cfg["stacks"][stack])
        if stack == "app":
            values.update(emulator_overrides(region, values["EnvName"], workarounds))
        r = apply(region, values["EnvName"], stack, values)
        out.append(r)
        say(f"deploy {env} {r['stack']}: {r['status']} in {r['seconds']} s, {r.get('resources', '-')} resources")
        for f in r.get("failures", []):
            say(f"  FAILED {f}")
        if r["status"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE", "NO_CHANGES"):
            break
    save(f"deploy_{env}", {"env": env, "region": cfg["region"], "total_seconds": round(time.time() - t0, 1), "stacks": out, "emulator_workarounds": workarounds})
    return out


def emulator_overrides(region: Region, env_name: str, log: List[Dict[str, str]]) -> Dict[str, str]:
    """The data stack values the app stack takes as parameters, as a pipeline passes them.

    On AWS both come from the data stack's outputs. The emulator leaves both outputs empty;
    then the secret comes from RDS itself and the cache gets a placeholder, and the log says so."""
    data = outputs(region, env_name, "data")
    values = {"DatabaseSecretArn": data.get("DatabaseSecretArn", ""), "CacheAddress": data.get("CacheAddress", "")}
    if not values["DatabaseSecretArn"]:
        values["DatabaseSecretArn"] = region("rds").describe_db_instances(DBInstanceIdentifier=f"beyondrag-{env_name}")["DBInstances"][0]["MasterUserSecret"]["SecretArn"]
        log.append({"output": "DatabaseSecretArn", "why": "Database.MasterUserSecret.SecretArn came back empty", "used": "the secret RDS reports for the instance"})
    if not values["CacheAddress"]:
        values["CacheAddress"] = "cache-not-created-by-the-emulator.invalid"
        log.append({"output": "CacheAddress", "why": "the emulator's CloudFormation creates no ElastiCache resources", "used": "a placeholder host name"})
    for w in log:
        say(f"  emulator workaround: {w['output']}: {w['why']}")
    return values


# --- inventory: does each resource the stack reports exist in its service? ---------------

def _exists(region: Region, rtype: str, pid: str, props: Dict[str, Any]) -> Any:
    """Ask the owning service for the resource CloudFormation says it created.

    Returns True or False, or a string saying why it could not be asked."""
    ec2 = region("ec2")
    lookups: Dict[str, Callable[[], Any]] = {
        "AWS::EC2::VPC": lambda: ec2.describe_vpcs(VpcIds=[pid])["Vpcs"],
        "AWS::EC2::Subnet": lambda: ec2.describe_subnets(SubnetIds=[pid])["Subnets"],
        "AWS::EC2::InternetGateway": lambda: ec2.describe_internet_gateways(InternetGatewayIds=[pid])["InternetGateways"],
        "AWS::EC2::VPCGatewayAttachment": lambda: [a for g in ec2.describe_internet_gateways()["InternetGateways"] for a in g.get("Attachments", []) if a.get("State") in ("available", "attached")],
        "AWS::EC2::EIP": lambda: ec2.describe_addresses(AllocationIds=[pid])["Addresses"] if pid.startswith("eipalloc") else ec2.describe_addresses(PublicIps=[pid])["Addresses"],
        "AWS::EC2::NatGateway": lambda: [g for g in ec2.describe_nat_gateways(NatGatewayIds=[pid])["NatGateways"] if g["State"] != "deleted"],
        "AWS::EC2::RouteTable": lambda: ec2.describe_route_tables(RouteTableIds=[pid])["RouteTables"],
        "AWS::EC2::Route": lambda: [r for t in ec2.describe_route_tables(RouteTableIds=[props["RouteTableId"]])["RouteTables"] for r in t["Routes"] if r.get("DestinationCidrBlock") == props.get("DestinationCidrBlock")],
        "AWS::EC2::SubnetRouteTableAssociation": lambda: ec2.describe_route_tables(Filters=[{"Name": "association.route-table-association-id", "Values": [pid]}])["RouteTables"],
        "AWS::EC2::VPCEndpoint": lambda: ec2.describe_vpc_endpoints(VpcEndpointIds=[pid])["VpcEndpoints"],
        "AWS::EC2::SecurityGroup": lambda: ec2.describe_security_groups(GroupIds=[pid])["SecurityGroups"],
        "AWS::RDS::DBInstance": lambda: region("rds").describe_db_instances(DBInstanceIdentifier=pid)["DBInstances"],
        "AWS::RDS::DBSubnetGroup": lambda: region("rds").describe_db_subnet_groups(DBSubnetGroupName=pid)["DBSubnetGroups"],
        "AWS::RDS::DBParameterGroup": lambda: region("rds").describe_db_parameter_groups(DBParameterGroupName=pid)["DBParameterGroups"],
        "AWS::OpenSearchService::Domain": lambda: region("opensearch").describe_domain(DomainName=pid)["DomainStatus"],
        "AWS::ElastiCache::SubnetGroup": lambda: region("elasticache").describe_cache_subnet_groups(CacheSubnetGroupName=pid)["CacheSubnetGroups"],
        "AWS::ElastiCache::ReplicationGroup": lambda: region("elasticache").describe_replication_groups(ReplicationGroupId=pid)["ReplicationGroups"],
        "AWS::S3::Bucket": lambda: region("s3").head_bucket(Bucket=pid),
        "AWS::S3::BucketPolicy": lambda: region("s3").get_bucket_policy(Bucket=props["Bucket"])["Policy"],
        "AWS::KMS::Key": lambda: region("kms").describe_key(KeyId=pid)["KeyMetadata"]["Enabled"],
        "AWS::KMS::Alias": lambda: [a for a in region("kms").list_aliases()["Aliases"] if a["AliasName"] == pid],
        "AWS::IAM::Role": lambda: region("iam").get_role(RoleName=pid)["Role"],
        "AWS::Backup::BackupVault": lambda: region("backup").describe_backup_vault(BackupVaultName=pid),
        "AWS::Backup::BackupPlan": lambda: region("backup").get_backup_plan(BackupPlanId=pid)["BackupPlan"],
        "AWS::Backup::BackupSelection": lambda: region("backup").get_backup_selection(BackupPlanId=props["BackupPlanId"], SelectionId=pid)["BackupSelection"],
        "AWS::SNS::Topic": lambda: region("sns").get_topic_attributes(TopicArn=pid)["Attributes"],
        "AWS::Logs::LogGroup": lambda: [g for g in region("logs").describe_log_groups(logGroupNamePrefix=pid)["logGroups"] if g["logGroupName"] == pid],
        "AWS::Grafana::Workspace": lambda: region("grafana").describe_workspace(workspaceId=pid)["workspace"],
        "AWS::ECR::Repository": lambda: region("ecr").describe_repositories(repositoryNames=[pid])["repositories"],
        "AWS::SecretsManager::Secret": lambda: region("secretsmanager").describe_secret(SecretId=pid)["ARN"],
        "AWS::ECS::Cluster": lambda: [c for c in region("ecs").describe_clusters(clusters=[pid])["clusters"] if c["status"] == "ACTIVE"],
        "AWS::ECS::TaskDefinition": lambda: region("ecs").describe_task_definition(taskDefinition=pid)["taskDefinition"],
        "AWS::ECS::Service": lambda: [s for s in region("ecs").describe_services(cluster=pid.split("/")[-2], services=[pid])["services"] if s["status"] == "ACTIVE"],
        "AWS::ElasticLoadBalancingV2::LoadBalancer": lambda: region("elbv2").describe_load_balancers(LoadBalancerArns=[pid])["LoadBalancers"],
        "AWS::ElasticLoadBalancingV2::TargetGroup": lambda: region("elbv2").describe_target_groups(TargetGroupArns=[pid])["TargetGroups"],
        "AWS::ElasticLoadBalancingV2::Listener": lambda: region("elbv2").describe_listeners(ListenerArns=[pid])["Listeners"],
        "AWS::ApplicationAutoScaling::ScalableTarget": lambda: region("application-autoscaling").describe_scalable_targets(ServiceNamespace="ecs", ResourceIds=[pid.split("|")[0]])["ScalableTargets"],
        "AWS::ApplicationAutoScaling::ScalingPolicy": lambda: [p for p in region("application-autoscaling").describe_scaling_policies(ServiceNamespace="ecs")["ScalingPolicies"] if p["PolicyARN"] == pid],
        "AWS::CloudWatch::Alarm": lambda: region("cloudwatch").describe_alarms(AlarmNames=[pid])["MetricAlarms"],
    }
    if rtype not in lookups:
        return "no lookup written"
    try:
        return bool(lookups[rtype]())
    except ClientError as exc:
        code = exc.response["Error"]["Code"]
        if code in ("NotImplemented", "NotImplementedException"):
            return "service API not implemented"
        return False


def load_template(stack: str) -> Dict[str, Any]:
    """Read a template, turning CloudFormation's short tags (!Ref, !Sub, ...) into the long form."""
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    def construct(loader: Any, suffix: str, node: Any) -> Any:
        name = "Ref" if suffix == "Ref" else f"Fn::{suffix}"
        if isinstance(node, yaml.ScalarNode):
            value = loader.construct_scalar(node)
        elif isinstance(node, yaml.SequenceNode):
            value = loader.construct_sequence(node, deep=True)
        else:
            value = loader.construct_mapping(node, deep=True)
        return {name: value}

    Loader.add_multi_constructor("!", construct)
    return yaml.load((TEMPLATES / f"{stack}.yaml").read_text(), Loader=Loader)


def inventory(env: str) -> Dict[str, Any]:
    """For every resource CloudFormation reports as created, ask the owning service."""
    cfg = load_params(env)
    region = Region(cfg["region"])
    cfn = region("cloudformation")
    n = cfg["stacks"]["network"]["EnvName"]
    rows = []
    for stack in STACKS:
        name = stack_name(n, stack)
        if not outputs(region, n, stack) and stack != "app":
            continue
        body = load_template(stack)
        for res in cfn.describe_stack_resources(StackName=name)["StackResources"]:
            if res["ResourceStatus"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
                continue
            props = {}
            if res["ResourceType"] in ("AWS::EC2::Route", "AWS::S3::BucketPolicy", "AWS::Backup::BackupSelection"):
                props = _resolved_props(cfn, name, res["LogicalResourceId"], body)
            found = _exists(region, res["ResourceType"], res.get("PhysicalResourceId", ""), props)
            rows.append({"stack": stack, "resource": res["LogicalResourceId"], "type": res["ResourceType"], "status": res["ResourceStatus"], "exists": found})
    by_type: Dict[str, Dict[str, int]] = {}
    for r in rows:
        key = "exists" if r["exists"] is True else "missing" if r["exists"] is False else "cannot check"
        by_type.setdefault(r["type"], {"exists": 0, "missing": 0, "cannot check": 0})[key] += 1
    summary = {"reported_complete": len(rows), "exist": sum(r["exists"] is True for r in rows),
               "missing": sum(r["exists"] is False for r in rows), "cannot_check": sum(not isinstance(r["exists"], bool) for r in rows)}
    result = {"env": env, "region": cfg["region"], "summary": summary, "by_type": by_type, "resources": rows}
    save(f"inventory_{env}", result)
    say(f"inventory {env}: {summary}")
    for t, c in sorted(by_type.items()):
        if c["missing"] or c["cannot check"]:
            say(f"  {t}: {c}")
    return result


def _resolved_props(cfn: Any, stack: str, logical: str, body: Dict[str, Any]) -> Dict[str, Any]:
    """The few properties a lookup needs, resolved to physical IDs."""
    props = dict(body.get("Resources", {}).get(logical, {}).get("Properties", {}))
    def phys(ref: Any) -> Any:
        if isinstance(ref, dict) and "Ref" in ref:
            return cfn.describe_stack_resource(StackName=stack, LogicalResourceId=ref["Ref"])["StackResourceDetail"]["PhysicalResourceId"]
        return ref
    return {k: phys(v) for k, v in props.items() if k in ("RouteTableId", "DestinationCidrBlock", "Bucket", "BackupPlanId")}


# --- verify -----------------------------------------------------------------------------

# The app stack's capacity profiles: (minimum, maximum) tasks when Live.
CAPACITY = {"Small": (1, 3), "Production": (2, 10)}

# Checks that cannot pass in the emulator however right the template is.
EMULATOR_GAPS = {
    "elasticache": "the emulator's CloudFormation (1.0.385) reports ElastiCache resources as created without creating them",
}
EMULATOR_GAP_CHECKS = {
    "its secret encrypted with the data key": "the emulator's CloudFormation ignores MasterUserSecret.KmsKeyId (RDS called directly honours it)",
}


def outputs(region: Region, env_name: str, stack: str) -> Dict[str, str]:
    """A stack's outputs; empty when the stack does not exist or did not complete."""
    try:
        s = region("cloudformation").describe_stacks(StackName=stack_name(env_name, stack))["Stacks"][0]
    except ClientError:
        return {}
    if s["StackStatus"] not in ("CREATE_COMPLETE", "UPDATE_COMPLETE", "UPDATE_ROLLBACK_COMPLETE"):
        return {}
    return {o["OutputKey"]: o.get("OutputValue", "") for o in s.get("Outputs", [])}


def physical(region: Region, env_name: str, stack: str, logical: str) -> str:
    return region("cloudformation").describe_stack_resource(StackName=stack_name(env_name, stack), LogicalResourceId=logical)["StackResourceDetail"]["PhysicalResourceId"]


def verify(env: str) -> Dict[str, Any]:
    """Read the deployed resources back from each service and check them against the design.

    Each check reads the service, not the template: what the emulator reports it created."""
    cfg = load_params(env)
    region = Region(cfg["region"])
    n = cfg["stacks"]["network"]["EnvName"]
    multi_az = cfg["stacks"]["data"]["MultiAz"] == "true"
    checks: List[Dict[str, Any]] = []

    def check(area: str, name: str, fn: Callable[[], Any], expect: Any = True) -> None:
        try:
            got = fn()
            ok = got == expect if not callable(expect) else bool(expect(got))
        except Exception as exc:  # a check that cannot be read is a failure, reported as such
            got, ok = f"{type(exc).__name__}: {exc}"[:200], False
        checks.append({"area": area, "check": name, "ok": ok, "observed": got})
        say(f"  {'PASS' if ok else 'FAIL'} [{area}] {name}: {str(got)[:120]}")

    ec2, rds, es, ec, s3 = region("ec2"), region("rds"), region("opensearch"), region("elasticache"), region("s3")
    ecs, elb, aas, cw, bk = region("ecs"), region("elbv2"), region("application-autoscaling"), region("cloudwatch"), region("backup")
    net = outputs(region, n, "network")
    data = outputs(region, n, "data")
    missing = [s for s in STACKS if not outputs(region, n, s) and s != "app"]
    if missing or not data:
        result = {"env": env, "region": cfg["region"], "passed": 0, "total": 0, "error": f"stacks not deployed: {missing}"}
        save(f"verify_{env}", result)
        say(f"verify {env}: skipped, stacks not deployed: {missing}")
        return result
    sg = {k: net[k] for k in ("LoadBalancerSecurityGroup", "AppSecurityGroup", "DatabaseSecurityGroup", "SearchSecurityGroup", "CacheSecurityGroup")}

    def subnets(key: str) -> List[Dict[str, Any]]:
        return ec2.describe_subnets(SubnetIds=net[key].split(","))["Subnets"]

    def routes_of(subnet_id: str) -> List[Dict[str, Any]]:
        tables = ec2.describe_route_tables(Filters=[{"Name": "association.subnet-id", "Values": [subnet_id]}])["RouteTables"]
        return [r for t in tables for r in t["Routes"]]

    def ingress(group: str) -> List[str]:
        rules = ec2.describe_security_groups(GroupIds=[group])["SecurityGroups"][0]["IpPermissions"]
        out = []
        for r in rules:
            port = r.get("FromPort")
            out += [f"{port} from {p['GroupId']}" for p in r.get("UserIdGroupPairs", [])]
            out += [f"{port} from {p['CidrIp']}" for p in r.get("IpRanges", [])]
        return sorted(out)

    say(f"verify {env} ({cfg['region']})")
    # Network
    check("network", "subnets span two Availability Zones", lambda: len({s["AvailabilityZone"] for s in subnets("AppSubnets")}), 2)
    check("network", "data subnets have no route to the internet",
          lambda: [r.get("DestinationCidrBlock") for s in net["DataSubnets"].split(",") for r in routes_of(s) if r.get("DestinationCidrBlock") == "0.0.0.0/0"], [])
    check("network", "app subnets reach the internet only through NAT",
          lambda: sorted({("nat" if r.get("NatGatewayId") else "igw" if r.get("GatewayId", "").startswith("igw") else "other")
                          for s in net["AppSubnets"].split(",") for r in routes_of(s) if r.get("DestinationCidrBlock") == "0.0.0.0/0"}), ["nat"])
    check("network", "NAT gateways (one per AZ in prod)",
          lambda: len(ec2.describe_nat_gateways(Filters=[{"Name": "vpc-id", "Values": [net["VpcId"]]}])["NatGateways"]),
          2 if cfg["stacks"]["network"]["NatGateways"] == "PerAz" else 1)
    check("network", "only the load balancer is open to the internet",
          lambda: sorted(k for k, g in sg.items() if any(x.endswith("0.0.0.0/0") for x in ingress(g))), ["LoadBalancerSecurityGroup"])
    check("network", "app accepts only the load balancer", lambda: ingress(sg["AppSecurityGroup"]), [f"8090 from {sg['LoadBalancerSecurityGroup']}"])
    for key, port in (("DatabaseSecurityGroup", 5432), ("SearchSecurityGroup", 443), ("CacheSecurityGroup", 6379)):
        check("network", f"{key.replace('SecurityGroup', '').lower()} accepts only the app", lambda key=key: ingress(sg[key]), [f"{port} from {sg['AppSecurityGroup']}"])

    # PostgreSQL
    db = lambda: rds.describe_db_instances(DBInstanceIdentifier=f"beyondrag-{n}")["DBInstances"][0]
    check("postgres", "engine", lambda: (db()["Engine"], db()["EngineVersion"].split(".")[0]), ("postgres", "16"))
    check("postgres", "Multi-AZ standby", lambda: db()["MultiAZ"], multi_az)
    key_id = lambda k: (k or "").split("/")[-1]  # RDS may report a key ID or a key ARN
    check("postgres", "storage encrypted with the data key", lambda: (db()["StorageEncrypted"], key_id(db().get("KmsKeyId")) == key_id(data["DataKeyArn"])), (True, True))
    check("postgres", "its secret encrypted with the data key", lambda: key_id(db()["MasterUserSecret"].get("KmsKeyId")) == key_id(data["DataKeyArn"]), True)
    check("postgres", "not publicly accessible", lambda: db()["PubliclyAccessible"], False)
    check("postgres", "automated backups kept", lambda: db()["BackupRetentionPeriod"], int(cfg["stacks"]["data"]["BackupRetentionDays"]))
    check("postgres", "deletion protection", lambda: db()["DeletionProtection"], cfg["stacks"]["data"].get("DeletionProtection", "true") == "true" or multi_az)
    check("postgres", "password managed in Secrets Manager", lambda: db().get("MasterUserSecret", {}).get("SecretArn", "").startswith("arn:"), True)
    check("postgres", "in the data subnets", lambda: sorted(s["SubnetIdentifier"] for s in db()["DBSubnetGroup"]["Subnets"]), sorted(net["DataSubnets"].split(",")))

    # OpenSearch
    dom = lambda: es.describe_domain(DomainName=f"beyondrag-{n}")["DomainStatus"]
    check("opensearch", "engine", lambda: dom()["EngineVersion"], "OpenSearch_2.17")
    check("opensearch", "zone awareness", lambda: dom()["ClusterConfig"].get("ZoneAwarenessEnabled", False), multi_az)
    check("opensearch", "data nodes", lambda: dom()["ClusterConfig"]["InstanceCount"], 2 if multi_az else 1)
    check("opensearch", "encrypted at rest and between nodes",
          lambda: (dom()["EncryptionAtRestOptions"]["Enabled"], dom()["NodeToNodeEncryptionOptions"]["Enabled"]), (True, True))
    check("opensearch", "HTTPS only", lambda: dom()["DomainEndpointOptions"]["EnforceHTTPS"], True)
    check("opensearch", "inside the VPC, not public", lambda: bool(dom().get("VPCOptions", {}).get("VPCId")), True)

    # Valkey
    rg = lambda: ec.describe_replication_groups(ReplicationGroupId=f"beyondrag-{n}")["ReplicationGroups"][0]
    check("elasticache", "engine", lambda: rg().get("Engine"), "valkey")
    check("elasticache", "automatic failover", lambda: rg()["AutomaticFailover"], "enabled" if multi_az else "disabled")
    check("elasticache", "Multi-AZ", lambda: rg()["MultiAZ"], "enabled" if multi_az else "disabled")
    check("elasticache", "nodes", lambda: len(rg()["MemberClusters"]), 2 if multi_az else 1)
    check("elasticache", "encrypted at rest and in transit", lambda: (rg()["AtRestEncryptionEnabled"], rg()["TransitEncryptionEnabled"]), (True, True))

    # Uploads
    bucket = data["UploadsBucket"]
    check("s3", "versioning", lambda: s3.get_bucket_versioning(Bucket=bucket).get("Status"), "Enabled")
    check("s3", "all public access blocked", lambda: all(s3.get_public_access_block(Bucket=bucket)["PublicAccessBlockConfiguration"].values()), True)
    check("s3", "encrypted with KMS",
          lambda: s3.get_bucket_encryption(Bucket=bucket)["ServerSideEncryptionConfiguration"]["Rules"][0]["ApplyServerSideEncryptionByDefault"]["SSEAlgorithm"], "aws:kms")
    check("s3", "TLS required by policy", lambda: "aws:SecureTransport" in s3.get_bucket_policy(Bucket=bucket)["Policy"], True)

    # Backups
    # ListBackupPlans is not implemented in the emulator; the plan's ID comes from the stack.
    plans = lambda: [bk.get_backup_plan(BackupPlanId=physical(region, n, "data", "BackupPlan"))["BackupPlan"]]
    check("backup", "daily plan, kept 35 days", lambda: [(r["ScheduleExpression"], r["Lifecycle"]["DeleteAfterDays"]) for p in plans() for r in p["Rules"]], [("cron(0 3 * * ? *)", 35)])
    check("backup", "copied to the recovery Region",
          lambda: [a["DestinationBackupVaultArn"].split(":")[3] for p in plans() for r in p["Rules"] for a in r.get("CopyActions", [])],
          ["us-west-2"] if cfg["stacks"]["data"].get("DrBackupVaultArn") else [])

    # App
    cluster, service = f"beyondrag-{n}", "app"
    svc = lambda: ecs.describe_services(cluster=cluster, services=[service])["services"][0]
    td = lambda: ecs.describe_task_definition(taskDefinition=svc()["taskDefinition"])["taskDefinition"]
    check("ecs", "Container Insights on", lambda: [s["value"] for s in ecs.describe_clusters(clusters=[cluster], include=["SETTINGS"])["clusters"][0]["settings"] if s["name"] == "containerInsights"], ["enabled"])
    check("ecs", "tasks in the private app subnets, no public IP",
          lambda: (sorted(svc()["networkConfiguration"]["awsvpcConfiguration"]["subnets"]), svc()["networkConfiguration"]["awsvpcConfiguration"]["assignPublicIp"]),
          (sorted(net["AppSubnets"].split(",")), "DISABLED"))
    check("ecs", "deployment circuit breaker rolls back", lambda: svc()["deploymentConfiguration"]["deploymentCircuitBreaker"], {"enable": True, "rollback": True})
    check("ecs", "desired tasks (Empty = 0)", lambda: svc()["desiredCount"], 0 if cfg["stacks"]["app"]["ServiceState"] == "Empty" else CAPACITY[cfg["stacks"]["app"]["Capacity"]][0])
    check("ecs", "secrets come from Secrets Manager, not plain environment",
          lambda: sorted(s["name"] for s in td()["containerDefinitions"][0].get("secrets", [])), ["POSTGRES_PASSWORD", "POSTGRES_USER", "RAG_AUTH_SECRET", "RAG_ERASURE_KEY"])
    check("ecs", "no secret-looking names in plain environment",
          lambda: [e["name"] for e in td()["containerDefinitions"][0].get("environment", []) if any(w in e["name"] for w in ("PASSWORD", "SECRET", "TOKEN", "KEY"))], [])
    check("ecs", "app wired to the data stores",
          lambda: sorted(e["name"] for e in td()["containerDefinitions"][0]["environment"] if e["name"] in ("ELASTICSEARCH_URL", "REDIS_URL", "POSTGRES_HOST", "RAG_UPLOADS_BUCKET")),
          ["ELASTICSEARCH_URL", "POSTGRES_HOST", "RAG_UPLOADS_BUCKET", "REDIS_URL"])
    lb_name = f"beyondrag-{n}"
    lb = lambda: elb.describe_load_balancers(Names=[lb_name])["LoadBalancers"][0]
    tg = lambda: elb.describe_target_groups(Names=[lb_name])["TargetGroups"][0]
    check("alb", "internet-facing in the public subnets", lambda: (lb()["Scheme"], sorted(a["SubnetId"] for a in lb()["AvailabilityZones"])), ("internet-facing", sorted(net["PublicSubnets"].split(","))))
    check("alb", "health check on /api/health", lambda: (tg()["HealthCheckPath"], tg()["TargetType"]), ("/api/health", "ip"))
    target = lambda: aas.describe_scalable_targets(ServiceNamespace="ecs", ResourceIds=[f"service/{cluster}/{service}"])["ScalableTargets"][0]
    live = cfg["stacks"]["app"]["ServiceState"] == "Live"
    check("scaling", "bounds", lambda: (target()["MinCapacity"], target()["MaxCapacity"]),
          CAPACITY[cfg["stacks"]["app"]["Capacity"]] if live else (0, 0))
    check("scaling", "target tracking on CPU and on requests per task",
          lambda: sorted(p["TargetTrackingScalingPolicyConfiguration"]["PredefinedMetricSpecification"]["PredefinedMetricType"]
                         for p in aas.describe_scaling_policies(ServiceNamespace="ecs", ResourceId=f"service/{cluster}/{service}")["ScalingPolicies"]),
          ["ALBRequestCountPerTarget", "ECSServiceAverageCPUUtilization"])
    topic = outputs(region, n, "observability")["AlarmTopicArn"]
    check("alarms", "app and data alarms notify the alarm topic",
          lambda: sorted(a["AlarmName"] for a in cw.describe_alarms(AlarmNamePrefix=f"beyondrag-{n}-")["MetricAlarms"] if topic in a["AlarmActions"]),
          sorted(f"beyondrag-{n}-{x}" for x in ("5xx", "p95-latency", "unhealthy-tasks", "db-cpu", "search-red")))
    check("grafana", "workspace reads CloudWatch",
          lambda: [w for w in region("grafana").list_workspaces()["workspaces"] if w["name"] == f"beyondrag-{n}"][0].get("status"), lambda s: s in ("ACTIVE", "CREATING"))

    for c in checks:
        if not c["ok"] and c["area"] in EMULATOR_GAPS and "NotFound" in str(c["observed"]):
            c["blocked_by"] = EMULATOR_GAPS[c["area"]]
        if not c["ok"] and c["check"] in EMULATOR_GAP_CHECKS:
            c["blocked_by"] = EMULATOR_GAP_CHECKS[c["check"]]
    blocked = [c for c in checks if c.get("blocked_by")]
    result = {"env": env, "region": cfg["region"], "passed": sum(c["ok"] for c in checks), "total": len(checks),
              "failed": [c["check"] for c in checks if not c["ok"] and not c.get("blocked_by")],
              "blocked_by_emulator": len(blocked), "checks": checks}
    save(f"verify_{env}", result)
    say(f"verify {env}: {result['passed']}/{result['total']} checks pass, {len(blocked)} blocked by a known emulator gap, failed: {result['failed']}")
    return result


# --- drills -----------------------------------------------------------------------------

def drills() -> Dict[str, Any]:
    cfg = load_params("prod")
    region = Region(cfg["region"])
    n = cfg["stacks"]["app"]["EnvName"]
    out: Dict[str, Any] = {}
    ecs, aas, cw, ec2, cfn = region("ecs"), region("application-autoscaling"), region("cloudwatch"), region("ec2"), region("cloudformation")
    cluster, resource = f"beyondrag-{n}", f"service/beyondrag-{n}/app"

    say("drill 1: go live: run the capacity profile (the image tag stays; delivering images is CI/CD's job)")
    live = dict(cfg["stacks"]["app"], ServiceState="Live")
    live.update(emulator_overrides(region, n, []))
    r = apply(region, n, "app", live, label="go-live")
    for f in r.get("failures", []):
        say(f"  FAILED {f}")
    svc = ecs.describe_services(cluster=cluster, services=["app"])["services"][0]
    t = aas.describe_scalable_targets(ServiceNamespace="ecs", ResourceIds=[resource])["ScalableTargets"][0]
    out["go_live"] = {"status": r["status"], "seconds": r["seconds"], "changes": r["changes"],
                      "desired": svc["desiredCount"], "bounds": [t["MinCapacity"], t["MaxCapacity"]]}
    say(f"  {r['status']}: {len(r['changes'])} change(s) {[c['resource'] + ':' + c['action'] for c in r['changes']]}, desired {svc['desiredCount']}, bounds {t['MinCapacity']}-{t['MaxCapacity']}")

    say("drill 2: CPU at 90% against a 60% target, then at 15%")
    # No task runs in the emulator, so CloudWatch has no CPU data to grade. The drill hands the
    # target-tracking alarm the datapoints CloudWatch would have graded (StateReasonData, the
    # field real CloudWatch fills), and watches what Application Auto Scaling does with them.
    alarms = cw.describe_alarms(AlarmNamePrefix=f"TargetTracking-{resource}")["MetricAlarms"]
    cpu = [a for a in alarms if a["MetricName"] == "CPUUtilization"]
    out["scaling"] = {"target_tracking_alarms": sorted(a["AlarmName"].split("-")[-2] + ":" + a["MetricName"] for a in alarms), "simulated_metric": True}

    def breach(alarm: Dict[str, Any], value: float, label: str) -> List[Any]:
        data = json.dumps({"version": "1.0", "queryDate": time.strftime("%Y-%m-%dT%H:%M:%S.000+0000", time.gmtime()),
                           "statistic": "Average", "period": alarm["Period"], "recentDatapoints": [value] * alarm["EvaluationPeriods"],
                           "threshold": alarm["Threshold"]})
        before = ecs.describe_services(cluster=cluster, services=["app"])["services"][0]["desiredCount"]
        cw.set_alarm_state(AlarmName=alarm["AlarmName"], StateValue="ALARM", StateReason=f"drill: {label}", StateReasonData=data)
        seen, t0 = [], time.time()
        while time.time() - t0 < 20:
            time.sleep(1)
            d = ecs.describe_services(cluster=cluster, services=["app"])["services"][0]["desiredCount"]
            seen.append(d)
            if d != before and len(seen) > 2 and seen[-1] == seen[-2]:
                break
        cw.set_alarm_state(AlarmName=alarm["AlarmName"], StateValue="OK", StateReason="drill: back to normal")
        return [before, seen[-1], round(time.time() - t0, 1)]

    high = [a for a in cpu if "AlarmHigh" in a["AlarmName"]]
    low = [a for a in cpu if "AlarmLow" in a["AlarmName"]]
    if high:
        out["scaling"]["cpu_90"] = breach(high[0], 90.0, "CPU 90% against a 60% target")
        say(f"  CPU 90%: desired {out['scaling']['cpu_90'][0]} -> {out['scaling']['cpu_90'][1]} in {out['scaling']['cpu_90'][2]} s")
    if low:
        out["scaling"]["cpu_15"] = breach(low[0], 15.0, "CPU 15% against a 60% target")
        say(f"  CPU 15%: desired {out['scaling']['cpu_15'][0]} -> {out['scaling']['cpu_15'][1]} in {out['scaling']['cpu_15'][2]} s (scale-in cooldown is 300 s)")
    acts = aas.describe_scaling_activities(ServiceNamespace="ecs", ResourceId=resource)["ScalingActivities"]
    out["scaling"]["activities"] = [(a["StatusCode"], a["Description"], a.get("Cause", "")[:160]) for a in acts][:6]
    out["scaling"]["running_tasks"] = ecs.describe_services(cluster=cluster, services=["app"])["services"][0]["runningCount"]
    for a in out["scaling"]["activities"]:
        say(f"  activity: {a[0]} {a[1]}")

    say("drill 3: redeploy with nothing changed")
    r = apply(region, n, "app", live, label="noop")
    out["no_op"] = {"status": r["status"], "reason": r.get("reason"), "changes": r["changes"]}
    say(f"  {r['status']}: {r.get('reason')}")

    say("drill 4: two changes made by hand, outside CloudFormation; drift detection")
    def detect(stack: str) -> Dict[str, Any]:
        det = cfn.detect_stack_drift(StackName=stack)["StackDriftDetectionId"]
        while True:
            st = cfn.describe_stack_drift_detection_status(StackDriftDetectionId=det)
            if st["DetectionStatus"] != "DETECTION_IN_PROGRESS":
                break
            time.sleep(1)
        drifts = cfn.describe_stack_resource_drifts(StackName=stack, StackResourceDriftStatusFilters=["MODIFIED", "DELETED"])["StackResourceDrifts"]
        return {"stack_status": st.get("StackDriftStatus"), "drifted": [
            {"resource": d["LogicalResourceId"], "status": d["StackResourceDriftStatus"],
             "differences": [(p["PropertyPath"], p["DifferenceType"]) for p in d.get("PropertyDifferences", [])][:4]} for d in drifts]}

    db_sg = outputs(region, n, "network")["DatabaseSecurityGroup"]
    bucket = outputs(region, n, "data")["UploadsBucket"]
    s3 = region("s3")
    ec2.authorize_security_group_ingress(GroupId=db_sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432, "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "debugging, remove later"}]}])
    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Suspended"})
    out["drift"] = {"database opened to 0.0.0.0/0": detect(stack_name(n, "network")), "uploads versioning suspended": detect(stack_name(n, "data"))}
    for k, v in out["drift"].items():
        say(f"  {k}: {v['stack_status']} {[(d['resource'], d['differences']) for d in v['drifted']]}")
    ec2.revoke_security_group_ingress(GroupId=db_sg, IpPermissions=[{"IpProtocol": "tcp", "FromPort": 5432, "ToPort": 5432, "IpRanges": [{"CidrIp": "0.0.0.0/0"}]}])
    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})

    say("drill 5: lose us-east-1; rebuild the recovery Region's app tier to production size")
    dr_cfg = load_params("dr")
    dr = Region(dr_cfg["region"])
    t0 = time.time()
    dr_values = dict(dr_cfg["stacks"]["app"], ServiceState="Live")
    dr_values.update(emulator_overrides(dr, n, []))
    r = apply(dr, n, "app", dr_values, label="failover")
    dr_svc = dr("ecs").describe_services(cluster=cluster, services=["app"])["services"][0]
    out["failover"] = {"status": r["status"], "seconds": round(time.time() - t0, 1), "changes": r["changes"], "desired_in_dr": dr_svc["desiredCount"],
                       "dr_backup_vault": dr("backup").describe_backup_vault(BackupVaultName=f"beyondrag-{n}")["BackupVaultArn"]}
    say(f"  {r['status']} in {out['failover']['seconds']} s, desired {dr_svc['desiredCount']} in {dr_cfg['region']}")

    save("drills", out)
    return out


def main() -> None:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    env = sys.argv[2] if len(sys.argv) > 2 else "prod"
    if cmd == "lint":
        lint()
    elif cmd == "deploy":
        deploy(env)
    elif cmd == "verify":
        verify(env)
    elif cmd == "inventory":
        inventory(env)
    elif cmd == "drills":
        drills()
    elif cmd == "all":
        t0 = time.time()
        lint()
        for e in ("dr", "prod"):
            deploy(e)
        for e in ("dr", "prod"):
            verify(e)
            inventory(e)
        drills()
        say(f"all done in {time.time() - t0:.0f} s; results in {RESULTS}")
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
