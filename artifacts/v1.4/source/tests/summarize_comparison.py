"""Summarize completed compare_versions runs without estimating official scores.

Only compact summary.json files are read. Source changes, incompatible source
hashes and duplicate experiments reject the whole merge. Repeat observations
are aggregated within each sample/version before cases receive equal weight.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import statistics


PROJECT = Path(__file__).resolve().parents[1]
METRICS = ("time_us", "request_s", "tokens")


def local_path(value, project=PROJECT):
    path = (project / value).resolve()
    if not path.is_relative_to(project.resolve()):
        raise ValueError("analysis paths must stay inside the project")
    return path


def version_key(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", value):
        raise ValueError("version labels must be numeric versions such as 1.3")
    return tuple(map(int, value.split(".")))


def number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def read_run(path):
    raw = path.read_bytes()
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema_version") != "agentcc.version-comparison.v1":
        raise ValueError("expected a compare_versions summary, not a legacy or analysis report")
    if not isinstance(data.get("rows"), list) or not data["rows"]:
        raise ValueError("run has no measurement rows")
    if not data.get("complete") or len(data.get("rows", [])) != data.get("expected_requests"):
        raise ValueError("run is incomplete; wait for all requests and the final source check")
    if data.get("source_files_changed_during_run") is not False:
        raise ValueError("source changed during the run, or its stability was not verified")
    before, after = data.get("source_sha256_at_start"), data.get("source_sha256_at_end")
    if not isinstance(before, dict) or not before or before != after:
        raise ValueError("source hashes are missing or differ between the start and end")
    versions = [item["version"] for item in data.get("versions", [])]
    if not versions or len(set(versions)) != len(versions) or set(before) != set(versions):
        raise ValueError("version declarations and source hashes do not match")
    for version, files in before.items():
        version_key(version)
        if not isinstance(files, dict) or not files:
            raise ValueError(f"version {version} has no source hashes")
        if any(not isinstance(key, str) or not isinstance(value, str)
               or not re.fullmatch(r"[0-9a-f]{64}", value) for key, value in files.items()):
            raise ValueError(f"version {version} contains malformed source hashes")
    seen = set()
    for row in data["rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("sample"), str) or not row["sample"]:
            raise ValueError("every observation needs a sample label")
        if row.get("version") not in versions or type(row.get("repeat")) is not int or row["repeat"] < 1:
            raise ValueError("every observation needs a declared version and positive repeat number")
        identity = row["sample"], row["version"], row["repeat"]
        if identity in seen:
            raise ValueError("duplicate sample/version/repeat inside one experiment")
        seen.add(identity)
        if type(row.get("valid")) is not bool or not number(row.get("request_s")):
            raise ValueError("every observation needs a validity flag and finite request duration")
        for metric in ("time_us", "tokens"):
            if row.get(metric) is not None and not number(row[metric]):
                raise ValueError(f"observation has an invalid {metric}")
        if row["valid"] and not number(row.get("time_us")):
            raise ValueError("a Replay-valid observation needs its communication duration")
        if row.get("model_selected") is not None and type(row["model_selected"]) is not bool:
            raise ValueError("model_selected must be true, false or absent")
        if row.get("calls") is not None and (type(row["calls"]) is not int or row["calls"] < 0):
            raise ValueError("model call counts must be nonnegative integers or absent")
    return data, hashlib.sha256(raw).hexdigest()


def valid_request(row):
    return (row["valid"] and row.get("status") == "success" and row.get("http_status") == 200
            and type(row.get("calls")) is int and row["calls"] == 1)


def distribution(values):
    if not values:
        return {"count": 0, "median": None, "mean": None, "min": None, "max": None}
    return {"count": len(values), "median": statistics.median(values),
            "mean": statistics.mean(values), "min": min(values), "max": max(values)}


def aggregate(rows):
    valid = [row for row in rows if valid_request(row)]
    editable = [row for row in valid if type(row.get("model_selected")) is bool]
    selected = sum(row["model_selected"] for row in editable)
    return {"requests": len(rows), "valid_requests": len(valid), "valid_rate": len(valid) / len(rows),
            "metrics": {metric: distribution([row[metric] for row in valid if number(row.get(metric))])
                        for metric in METRICS},
            "model_observations": len(editable), "model_selected": selected,
            "model_selection_rate": selected / len(editable) if editable else None,
            "tokens_total_all_requests": sum(row["tokens"] for row in rows if number(row.get("tokens"))),
            "known_token_observations": sum(number(row.get("tokens")) for row in rows),
            "llm_calls_total": sum(row["calls"] for row in rows if type(row.get("calls")) is int),
            "failed_requests": len(rows) - len(valid)}


def compare_pair(groups, samples, baseline, target):
    rows, regressions, validity_regressions, excluded = [], [], [], []
    for sample in samples:
        old, new = groups.get((sample, baseline)), groups.get((sample, target))
        if old is None or new is None:
            excluded.append({"sample": sample, "reason": "version missing"})
            continue
        if new["valid_rate"] < old["valid_rate"]:
            validity_regressions.append({"sample": sample, "baseline": old["valid_rate"],
                                         "target": new["valid_rate"]})
        comparable = old["valid_rate"] == 1 and new["valid_rate"] == 1
        if not comparable:
            excluded.append({"sample": sample, "reason": "one or both versions have failed requests"})
        row = {"sample": sample, "comparable": comparable,
               "baseline_valid_rate": old["valid_rate"], "target_valid_rate": new["valid_rate"],
               "metrics": {}}
        for metric in METRICS:
            before, after = old["metrics"][metric]["median"], new["metrics"][metric]["median"]
            eligible = comparable and before is not None and after is not None
            change = after - before if eligible else None
            entry = {"baseline_median": before, "target_median": after, "eligible": eligible,
                     "delta": change,
                     "change_pct": 100 * change / before if eligible and before > 0 else
                                   0.0 if eligible and before == after == 0 else None,
                     "speedup": before / after if eligible and before > 0 and after > 0 else None}
            if eligible and change > max(1e-9, abs(before) * 1e-9):
                regressions.append({"sample": sample, "metric": metric, **entry})
            row["metrics"][metric] = entry
        rows.append(row)
    metrics = {}
    for metric in METRICS:
        entries = [row["metrics"][metric] for row in rows if row["metrics"][metric]["eligible"]]
        ratios = [entry["speedup"] for entry in entries if entry["speedup"] is not None]
        faster = sum(entry["delta"] < -max(1e-9, abs(entry["baseline_median"]) * 1e-9) for entry in entries)
        slower = sum(entry["delta"] > max(1e-9, abs(entry["baseline_median"]) * 1e-9) for entry in entries)
        metrics[metric] = {"paired_cases": len(entries), "faster_cases": faster,
                           "slower_cases": slower, "tied_cases": len(entries) - faster - slower,
                           "baseline_case_median_mean": statistics.mean(entry["baseline_median"] for entry in entries) if entries else None,
                           "target_case_median_mean": statistics.mean(entry["target_median"] for entry in entries) if entries else None,
                           "geomean_speedup": math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None,
                           "geomean_pairs": len(ratios)}
    return {"baseline": baseline, "target": target, "metrics": metrics, "rows": rows,
            "regressions": regressions, "validity_regressions": validity_regressions,
            "excluded_cases": excluded}


def summarize(paths, *, target=None, project=PROJECT):
    """Reject incompatible experiments instead of silently dropping evidence."""
    report = {"schema_version": "agentcc.comparison-analysis.v1", "status": "rejected",
              "official_score_estimate": False, "scope": "Local public samples, not official scores",
              "inputs": [], "rejections": [], "source_sha256": {},
              "notes": [
                  "Communication, request and Token distributions use valid requests only; failures remain in valid_rate and total costs.",
                  "Each sample has equal weight after repeats are aggregated by sample/version median.",
                  "Speedups compare group medians, not individual paired-request ratios. Ratios with zero durations are omitted from the geometric mean.",
                  "Source hashes must match exactly for every shared version. Samples are matched by labels; the summary does not contain input fingerprints.",
                  "The analysis reads only summary files. It makes no model/network calls and does not predict platform scores.",
              ]}
    runs, seen_paths, seen_content, sources = [], set(), set(), {}
    conditions = None
    for value in paths:
        path = local_path(value, project)
        relative = str(path.relative_to(project.resolve()))
        try:
            if path in seen_paths:
                raise ValueError("same summary was supplied more than once")
            seen_paths.add(path)
            data, digest = read_run(path)
            if digest in seen_content:
                raise ValueError("duplicate summary content is not an independent repeat")
            seen_content.add(digest)
            current_conditions = {key: data.get(key) for key in ("limits", "timeout_ms", "timing")}
            if conditions is not None and current_conditions != conditions:
                raise ValueError("resource limits, timeout or timing definitions differ across experiments")
            conditions = current_conditions
            for version, hashes in data["source_sha256_at_start"].items():
                if version in sources and sources[version] != hashes:
                    raise ValueError(f"version {version} source hashes differ across experiments")
                sources[version] = hashes
            report["inputs"].append({"path": relative, "summary_sha256": digest,
                                     "requests": len(data["rows"]), "complete": True,
                                     "source_unchanged": True, "conditions": current_conditions})
            runs.append(data)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            report["rejections"].append({"path": relative, "reason": str(exc)})
    if report["rejections"]:
        report["rejection_policy"] = "No measurements merged because at least one input was incompatible or incomplete."
        return report
    if not runs:
        raise ValueError("at least one completed summary is required")
    observations = defaultdict(list)
    samples = []
    for run in runs:
        for row in run["rows"]:
            if row["sample"] not in samples:
                samples.append(row["sample"])
            observations[row["sample"], row["version"]].append(row)
    versions = sorted(sources, key=version_key)
    if target is not None and target not in versions:
        raise ValueError("target version is not present in the measurements")
    groups = {key: aggregate(rows) for key, rows in observations.items()}
    comparisons = {version: {baseline: compare_pair(groups, samples, baseline, version)
                             for baseline in versions if baseline != version} for version in versions}
    report.update(status="complete", source_sha256=sources, versions=versions, samples=samples,
                  default_target=target or versions[-1],
                  requests=sum(len(run["rows"]) for run in runs),
                  valid_requests=sum(group["valid_requests"] for group in groups.values()),
                  groups=[{"sample": sample, "version": version, **value}
                          for (sample, version), value in groups.items()], comparisons=comparisons)
    return report


def render_html(report, output):
    """Self-contained HTML; no external scripts, data requests or dependencies."""
    back = html.escape(os.path.relpath(PROJECT / "artifacts/index.html", output), quote=True)
    links = " · ".join(f'<a href="{html.escape(os.path.relpath(PROJECT / item["path"], output), quote=True)}">实验 {index + 1}</a>'
                       for index, item in enumerate(report["inputs"]))
    if report["status"] != "complete":
        errors = "".join(f'<li><code>{html.escape(item["path"])}</code>：{html.escape(item["reason"])}</li>'
                         for item in report["rejections"])
        content = f'<section class="panel"><h2>拒绝合并</h2><p>来源不完整或不兼容，没有生成任何汇总统计。</p><ul>{errors}</ul></section>'
        script = ""
    else:
        content = '''<section class="panel"><div class="controls"><label>基线版本<select id="baseline"></select></label><label>目标版本<select id="target"></select></label><label>观察指标<select id="metric"><option value="time_us">通信耗时 · μs</option><option value="request_s">求解时间 · s</option><option value="tokens">Token 用量</option></select></label></div><div class="cards"><div><span>等权平均 · 各例中位数</span><strong id="means">—</strong></div><div><span>几何平均加速比 · 大于 1 更好</span><strong id="speedup">—</strong></div><div><span>改善 / 持平 / 退化</span><strong id="counts">—</strong></div></div><p id="scope" class="muted"></p><div class="table-wrap"><table><thead><tr><th>样例</th><th id="before-label">基线</th><th id="after-label">目标</th><th>中位数变化</th><th>基线 / 目标有效请求</th><th>模型入选率 · 基线 / 目标</th></tr></thead><tbody id="rows"></tbody></table></div><p class="muted">每个指标主值为组内中位数，小字为均值。绿色表示目标更低，红色表示目标更高。任一版本出现失败时，该样例不进入加速比与性能胜负统计；失败仍计入有效率。</p></section><section class="panel"><h2>来源与解释边界</h2><ul id="notes"></ul><p class="muted">只有相同源码的独立重复可以合并。模型入选率按有效请求中具有具体编辑记录的观察计算；旧版本没有该字段时显示“未记录”。</p></section><noscript><section class="panel">交互表需要 JavaScript。原始统计仍可通过 report.json 查看。</section></noscript>'''
        data = json.dumps({key: report[key] for key in ("versions", "default_target", "groups", "comparisons", "notes")},
                          ensure_ascii=False, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
        script = '<script type="application/json" id="report-data">' + data + '</script>' + SCRIPT
    return HTML.replace("__BACK__", back).replace("__LINKS__", links).replace("__CONTENT__", content).replace("__SCRIPT__", script)


HTML = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>AgentCC · 本地版本对照</title>
<style>
:root{font-family:system-ui,-apple-system,"Segoe UI","Noto Sans SC",sans-serif;color:#1e2d43;background:#f3f6fa;line-height:1.6;color-scheme:light}*{box-sizing:border-box}body{margin:0}main{max-width:1220px;margin:auto;padding:32px 22px}h1{font-size:28px;margin:0 0 10px}h2{font-size:19px;margin:0 0 10px}p{margin-top:0}a{color:#24599e;text-underline-offset:3px}.badge{display:inline-block;background:#e8eef8;border-radius:5px;color:#436184;font-size:12px;padding:3px 8px;margin-bottom:12px}.muted{color:#6a788b;font-size:13px}.nav{display:flex;gap:20px;flex-wrap:wrap;margin:16px 0 24px;font-size:13px}.panel{background:white;border:1px solid #dee5ef;border-radius:12px;padding:22px;margin:18px 0}.controls{display:flex;gap:20px;flex-wrap:wrap;margin-bottom:20px}label{font-size:13px;color:#596c83}select{display:block;margin-top:5px;padding:7px 12px;font:inherit;color:#1e2d43;background:white;border:1px solid #cbd7e5;border-radius:6px;min-width:130px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:18px}.cards>div{background:#f5f8fc;padding:14px;border-radius:8px}.cards span{font-size:12px;color:#6c7b90}.cards strong{display:block;font-size:23px;font-variant-numeric:tabular-nums;margin-top:6px}.table-wrap{overflow:auto}table{border-collapse:collapse;width:100%;font-size:13px;white-space:nowrap}th,td{padding:11px 12px;text-align:right;border-bottom:1px solid #e8edf5;font-variant-numeric:tabular-nums}th{font-weight:600;color:#5d7089;background:#f7f9fc;font-size:12px}th:first-child,td:first-child{text-align:left}td small{display:block;color:#7d899c;font-size:11px}.good{color:#137765}.bad{color:#b24836}.neutral{color:#7b8796}ul{padding-left:21px;font-size:13px}li{margin-bottom:7px}footer{font-size:12px;color:#7c889b;margin:20px 0}code{overflow-wrap:anywhere}@media(max-width:700px){main{padding:20px 12px}.panel{padding:16px}.cards{grid-template-columns:1fr}.cards strong{font-size:21px}.controls{gap:12px}}
</style></head><body><main><span class="badge">本地公开样例，非官方得分</span><h1>版本对照：通信、求解时间与 Token</h1><p class="muted">按相同样例和源码版本聚合重复请求。报告只展示测量值与相对变化，不估算平台分数。</p><nav class="nav"><a href="__BACK__">返回版本看板</a><a href="report.json">完整分析 JSON</a><span>__LINKS__</span></nav>__CONTENT__<footer>离线报告，无外部依赖、无联网请求。有效请求要求 SDK 成功、一次模型调用及独立 Replay 有效。源记录中的失败不会被静默删掉。</footer></main>__SCRIPT__</body></html>'''


SCRIPT = '''<script>
(() => {
 const data=JSON.parse(document.getElementById('report-data').textContent);
 const baseline=document.getElementById('baseline'), target=document.getElementById('target'), metric=document.getElementById('metric');
 const groups=new Map(data.groups.map(group=>[JSON.stringify([group.sample,group.version]),group]));
 const display=(value,digits=3)=>value==null?'—':Number(value).toLocaleString('en-US',{maximumFractionDigits:digits});
 for(const version of data.versions){for(const select of [baseline,target]){const option=document.createElement('option');option.value=version;option.textContent=version;select.appendChild(option);}}
 target.value=data.default_target;baseline.value=data.versions.filter(version=>version!==target.value).slice(-1)[0]||target.value;
 const rate=group=>group.model_selection_rate==null?'未记录':display(group.model_selection_rate*100,1)+'% ('+group.model_selected+'/'+group.model_observations+')';
 function draw(){
  if(baseline.value===target.value){baseline.value=data.versions.find(version=>version!==target.value)||target.value;}
  const comparison=data.comparisons[target.value][baseline.value], body=document.getElementById('rows');body.replaceChildren();
  if(!comparison){document.getElementById('scope').textContent='只有一个版本，分组统计保存在 report.json；选择至少两个版本才能比较。';return;}
  const summary=comparison.metrics[metric.value], digits=metric.value==='tokens'?1:3;
  document.getElementById('means').textContent=display(summary.baseline_case_median_mean,digits)+' → '+display(summary.target_case_median_mean,digits);
  document.getElementById('speedup').textContent=summary.geomean_speedup==null?'—':display(summary.geomean_speedup,3)+'×';
  document.getElementById('counts').textContent=summary.faster_cases+' / '+summary.tied_cases+' / '+summary.slower_cases;
  document.getElementById('before-label').textContent=baseline.value+' · 中位 / 均值';document.getElementById('after-label').textContent=target.value+' · 中位 / 均值';
  document.getElementById('scope').textContent='纳入 '+summary.paired_cases+' 个双方均有效的样例；几何平均使用 '+summary.geomean_pairs+' 个非零比值。未配齐或含失败的样例 '+comparison.excluded_cases.length+' 个。';
  const rows=comparison.rows.slice().sort((a,b)=>(b.metrics[metric.value].change_pct??-Infinity)-(a.metrics[metric.value].change_pct??-Infinity));
  for(const row of rows){
   const old=groups.get(JSON.stringify([row.sample,baseline.value])), next=groups.get(JSON.stringify([row.sample,target.value])), value=row.metrics[metric.value], tr=document.createElement('tr');
   const cell=(text,cls)=>{const td=document.createElement('td');td.textContent=text;if(cls)td.className=cls;tr.appendChild(td);return td;};
   cell(row.sample);
   for(const group of [old,next]){const stats=group.metrics[metric.value], td=cell(display(stats.median,digits)), small=document.createElement('small');small.textContent='均值 '+display(stats.mean,digits)+' · n='+stats.count;td.appendChild(small);}
   const change=value.change_pct==null?'—':(value.change_pct>0?'+':'')+display(value.change_pct,2)+'%';
   cell(value.eligible?change:'不纳入比较',!value.eligible?'neutral':value.delta>1e-9?'bad':value.delta< -1e-9?'good':'neutral');
   cell(old.valid_requests+'/'+old.requests+' / '+next.valid_requests+'/'+next.requests);
   cell(rate(old)+' / '+rate(next));body.appendChild(tr);
  }
 }
 const notes=['指标分布仅使用有效请求；所有失败仍计入有效率，全部已知 Token 消耗另外保留。','先对每个样例、每个版本取重复观察的中位数，再对各样例等权汇总。','加速比是组内中位数之比；不等于逐轮成对加速比的平均值。','同版本源码 SHA 已核对一致。样例按标签匹配，summary 不含输入指纹；实际输入一致性仍需实验设计保证。'];
 const list=document.getElementById('notes');for(const note of notes){const li=document.createElement('li');li.textContent=note;list.appendChild(li);}
 for(const select of [baseline,target,metric])select.addEventListener('change',draw);draw();
})();
</script>'''


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", help="one or more completed compare_versions summary.json files")
    parser.add_argument("--output-dir", help="default: the first input directory; writes report.json and report.html")
    parser.add_argument("--target", help="default comparison target; default: highest measured version")
    args = parser.parse_args(argv)
    try:
        paths = [local_path(value) for value in args.summaries]
        output = local_path(args.output_dir) if args.output_dir else paths[0].parent
        if output / "report.json" in paths or output / "report.html" in paths:
            raise ValueError("report output must not overwrite an input summary")
        report = summarize(paths, target=args.target)
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        (output / "report.html").write_text(render_html(report, output))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"status": report["status"], "report": str(output.relative_to(PROJECT)),
                      "requests": report.get("requests", 0), "rejections": len(report["rejections"])}, ensure_ascii=False))
    return 0 if report["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
