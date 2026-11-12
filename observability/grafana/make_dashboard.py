"""Writes observability/grafana/dashboards/beyondrag-overview.json. Run from the repository root."""
import json
from pathlib import Path

PROM = {"type": "prometheus", "uid": "prometheus"}
LOKI = {"type": "loki", "uid": "loki"}
TEMPO = {"type": "tempo", "uid": "tempo"}
SEL = 'service_name=~"$service", rag_tenant=~"$tenant"'
panels = []
LINES = '{service_name=~"$service"} |= "rag.request" | json | tenant=~"$tenant"'

def ltarget(expr, legend="", ref="A", instant=False):
    t = {"datasource": LOKI, "expr": expr, "legendFormat": legend, "refId": ref}
    t["queryType"] = "instant" if instant else "range"
    return t

def lstat(title, expr, unit="short", thresholds=None, desc=""):
    panel = stat(title, expr, unit, thresholds, desc)
    panel["datasource"] = LOKI
    panel["targets"] = [ltarget(expr, instant=True)]
    return panel

pid = [0]

def add(panel, x, y, w, h):
    pid[0] += 1
    panel.update({"id": pid[0], "gridPos": {"x": x, "y": y, "w": w, "h": h}})
    panels.append(panel)

def target(expr, legend="", ref="A", instant=False):
    t = {"datasource": PROM, "expr": expr, "legendFormat": legend or "__auto", "refId": ref}
    if instant:
        t.update({"instant": True, "range": False})
    return t

def stat(title, expr, unit="short", thresholds=None, desc=""):
    return {
        "type": "stat", "title": title, "description": desc, "datasource": PROM,
        "targets": [target(expr, instant=True)],
        "fieldConfig": {"defaults": {"unit": unit, "decimals": {"percentunit": 1, "currencyUSD": 3}.get(unit, 0),
            "thresholds": {"mode": "absolute", "steps": thresholds or [{"color": "text", "value": None}]}}, "overrides": []},
        "options": {"reduceOptions": {"calcs": ["lastNotNull"]}, "colorMode": "value", "graphMode": "none", "textMode": "value"},
    }

def ts(title, targets, unit="short", stack=False, desc="", overrides=None):
    bars = stack and any("increase(" in t.get("expr", "") for t in targets)
    if bars:
        desc = (desc + " " if desc else "") + "Prometheus drops the first increase of each new series, so these are a floor."
    return {"interval": "5m" if bars else None,
        "type": "timeseries", "title": title, "description": desc, "datasource": PROM, "targets": targets,
        "fieldConfig": {"defaults": {"unit": unit, "custom": {"drawStyle": "bars" if bars else "line", "lineWidth": 2 if not bars else 1, "fillOpacity": 80 if bars else 0,
            "stacking": {"mode": "normal" if stack else "none"}, "showPoints": "never"}}, "overrides": overrides or []},
        "options": {"legend": {"displayMode": "table", "placement": "right", "calcs": ["sum"] if stack else ["mean", "max"]},
                    "tooltip": {"mode": "multi", "sort": "desc"}},
    }

total = f'sum(count_over_time({LINES} [$__range]))'
def share(group):
    return f'(sum(count_over_time({LINES} | outcome_group="{group}" [$__range])) or vector(0)) / {total}'

red = [{"color": "green", "value": None}, {"color": "orange", "value": 0.05}, {"color": "red", "value": 0.2}]
y = 0
add({"type": "row", "title": "Did people get answers?", "collapsed": False, "panels": []}, 0, y, 24, 1); y += 1
add(lstat("Questions", total, desc="Questions asked in the selected time range."), 0, y, 4, 4)
add(lstat("Answered", share("ok"), "percentunit", [{"color": "red", "value": None}, {"color": "orange", "value": 0.8}, {"color": "green", "value": 0.95}],
         "answered or answered_extractive."), 4, y, 5, 4)
add(lstat("Degraded", share("degraded"), "percentunit", red,
         "HTTP 200, but not the answer that was asked for: the model failed and the fast answer was used, the budget ran out, or the verifier found unsupported claims."), 9, y, 5, 4)
add(lstat("Refused", share("refused"), "percentunit", [{"color": "text", "value": None}],
         "Blocked by a guardrail, no documents, access blocked, or an abstention. Not an error, but a rising share means something changed."), 14, y, 5, 4)
add(lstat("Errors", share("error"), "percentunit", red, "Exceptions: the request failed."), 19, y, 5, 4)
y += 4
q = ts("Questions by outcome", [ltarget(f'sum by (outcome) (count_over_time({LINES} [$__interval]))', "{{outcome}}")],
       stack=True, desc="One label per question, counted from the one log line each question writes. Every degraded label here returned HTTP 200.")
q["datasource"] = LOKI; q["interval"] = "5m"; q["fieldConfig"]["defaults"]["custom"].update({"drawStyle": "bars", "fillOpacity": 80})
add(q, 0, y, 12, 9)
add(ts("Latency per question (p50, p95)", [
        target(f'histogram_quantile(0.5, sum by (le) (rate(rag_request_duration_seconds_bucket{{{SEL}}}[$__rate_interval])))', "p50"),
        target(f'histogram_quantile(0.95, sum by (le) (rate(rag_request_duration_seconds_bucket{{{SEL}}}[$__rate_interval])))', "p95", "B"),
    ], "s"), 12, y, 12, 9)
y += 9
add({"type": "row", "title": "What did it cost?", "collapsed": False, "panels": []}, 0, y, 24, 1); y += 1
add(lstat("Input tokens in questions", f'sum(sum_over_time({LINES} | unwrap input_tokens [$__range]))'), 0, y, 6, 4)
add(lstat("Output tokens in questions", f'sum(sum_over_time({LINES} | unwrap output_tokens [$__range]))'), 6, y, 6, 4)
add(lstat("Model calls in questions", f'sum(sum_over_time({LINES} | unwrap llm_calls [$__range]))'), 12, y, 6, 4)
add(lstat("Tokens per question", f'(sum(sum_over_time({LINES} | unwrap input_tokens [$__range])) + sum(sum_over_time({LINES} | unwrap output_tokens [$__range]))) / {total}'), 18, y, 6, 4)
y += 4
add(ts("Tokens by pipeline stage", [target(f'sum by (rag_stage) (increase(gen_ai_client_token_usage_sum{{{SEL}}}[$__interval]))', "{{rag_stage}}")],
       stack=True, desc="Where the tokens go. Generation is rarely the only stage that calls a model."), 0, y, 12, 9)
add(ts("Tokens by tenant", [target(f'sum by (rag_tenant) (increase(gen_ai_client_token_usage_sum{{{SEL}}}[$__interval]))', "{{rag_tenant}}")],
       stack=True, desc="Tenant is a label here, which is fine for tens of tenants and not for tens of thousands."), 12, y, 12, 9)
y += 9
add(ts("Model call latency p95, by model and operation", [target(
        f'histogram_quantile(0.95, sum by (le, gen_ai_request_model, gen_ai_operation_name) (rate(gen_ai_client_operation_duration_seconds_bucket{{{SEL}}}[$__rate_interval])))',
        "{{gen_ai_operation_name}} {{gen_ai_request_model}}")], "s"), 0, y, 12, 8)
add(ts("Stage latency p95", [target(
        'histogram_quantile(0.95, sum by (le, rag_stage) (rate(rag_stage_duration_seconds_bucket{service_name=~"$service"}[$__rate_interval])))', "{{rag_stage}}")], "s"),
    12, y, 12, 8)
y += 8
add({"type": "row", "title": "What went wrong, and what was stopped?", "collapsed": False, "panels": []}, 0, y, 24, 1); y += 1
add(ts("Failure events by kind", [target(f'sum by (rag_event) (increase(rag_events_total{{{SEL}}}[$__interval]))', "{{rag_event}}")],
       stack=True, desc="Fallbacks, failed steps and refusals, each counted where it happened."), 0, y, 12, 8)
add(ts("Guardrail decisions", [target(f'sum by (rag_guardrail, rag_guardrail_action) (increase(rag_guardrail_decisions_total{{{SEL}}}[$__interval]))',
       "{{rag_guardrail}} {{rag_guardrail_action}}")], stack=True), 12, y, 12, 8)
y += 8
add({
    "type": "table", "title": "Questions that did not end in an answer (traces)", "datasource": TEMPO,
    "description": "Click a trace id to open the waterfall.",
    "targets": [{"datasource": TEMPO, "refId": "A", "queryType": "traceql", "limit": 20, "tableType": "traces",
                 "query": '{ name = "rag.query" && span.rag.outcome_group != "ok" && resource.service.name =~ "$service" }'}],
    "fieldConfig": {"defaults": {}, "overrides": []}, "options": {"showHeader": True},
}, 0, y, 12, 10)
add({
    "type": "logs", "title": "One line per question", "datasource": LOKI,
    "description": "The wide event: outcome, tokens, events and guardrails. Never the question text.",
    "targets": [{"datasource": LOKI, "refId": "A", "expr": '{service_name=~"$service"} |= "rag.request"'}],
    "options": {"showTime": True, "wrapLogMessage": True, "sortOrder": "Descending", "enableLogDetails": True},
}, 12, y, 12, 10)

y += 10  # past the traces table and the log panel
add({"type": "row", "title": "What did caching and routing save?", "collapsed": False, "panels": []}, 0, y, 24, 1); y += 1
add(lstat("Answered from the cache", f'(sum(count_over_time({LINES} | outcome="answered_cached" [$__range])) or vector(0)) / {total}', "percentunit",
          desc="Share of questions served by the answer cache, exact or semantic."), 0, y, 6, 4)
add(lstat("Model time spent", f'sum(sum_over_time({LINES} | unwrap model_seconds [$__range]))', "s",
          desc="Seconds of model generation the questions actually used."), 6, y, 6, 4)
add(lstat("Model time saved", f'sum(sum_over_time({LINES} | unwrap saved_model_seconds [$__range]))', "s",
          desc="Seconds of model time that cache hits avoided, as recorded when each entry was filled."), 12, y, 6, 4)
add(lstat("Estimated cost", f'sum(sum_over_time({LINES} | unwrap estimated_cost_usd [$__range]))', "currencyUSD",
          desc="Illustrative: local tokens priced at a hosted model of a similar tier, from config/model_prices.json."), 18, y, 6, 4)
y += 4
add(ts("Cache lookups by layer", [target(f'sum by (rag_cache_layer, rag_cache_result) (increase(rag_cache_lookups_total{{{SEL}}}[$__interval]))',
       "{{rag_cache_layer}} {{rag_cache_result}}")], stack=True), 0, y, 12, 8)
add(ts("Model calls by model", [target(f'sum by (gen_ai_request_model, gen_ai_operation_name) (increase(gen_ai_client_operation_duration_seconds_count{{{SEL}}}[$__interval]))',
       "{{gen_ai_operation_name}} {{gen_ai_request_model}}")], stack=True, desc="Where routing sends the calls."), 12, y, 12, 8)
y += 8

for panel in panels:
    if panel.get("interval") is None:
        panel.pop("interval", None)

dashboard = {
    "uid": "beyondrag-overview", "title": "BeyondRAG: answers, cost and failures", "tags": ["beyondrag", "genai"],
    "timezone": "browser", "schemaVersion": 41, "version": 1, "refresh": "30s",
    "time": {"from": "now-1h", "to": "now"},
    "templating": {"list": [
        {"name": "service", "label": "Service", "type": "query", "datasource": PROM, "query": "label_values(rag_requests_total, service_name)",
         "definition": "label_values(rag_requests_total, service_name)", "refresh": 2, "includeAll": True, "multi": True,
         "current": {"text": "All", "value": "$__all"}, "allValue": ".*"},
        {"name": "tenant", "label": "Tenant", "type": "query", "datasource": PROM, "query": 'label_values(rag_requests_total{service_name=~"$service"}, rag_tenant)',
         "definition": 'label_values(rag_requests_total{service_name=~"$service"}, rag_tenant)', "refresh": 2, "includeAll": True, "multi": True,
         "current": {"text": "All", "value": "$__all"}, "allValue": ".*"},
    ]},
    "panels": panels,
}
Path("observability/grafana/dashboards/beyondrag-overview.json").write_text(json.dumps(dashboard, indent=2) + "\n")
print(len(panels), "panels")
