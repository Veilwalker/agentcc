"""One model-generated search program, with a bounded feedback repair call."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
import time

from .llm import LLMError, _chat_completion


SYSTEM = (
    'Write only compact Python def solve(c,limit), no comments/imports. '
    'c[i][w-1] is current capacity-derived cost for transition i->i+w. '
    'Rows are ragged; enumerate each row from width 1. Find lowest SUM cost path '
    '0->len(c)-1, at most limit steps. c includes an EMPTY final row for the terminal. '
    'Return widths only, summing to len(c)-1. Initial path is []. '
    'Generate forward DP storing (cost,path), no backtracking. If step_bound_free is '
    'true, one state per position suffices; otherwise retain step count too. '
    'Initialize ONLY position 0 to cost 0 and an empty path; all other states cost infinity. '
    'Use for/if, lists/dicts, indexing, len/range/enumerate/min/float, append/get. '
    'All positions are reachable; every row contains only in-bounds transitions. '
    'No helpers/lambda/keywords. Actual tables arrive at execution; never hardcode a path. '
    'Use short variable names, under 120 tokens if possible. '
    'Independent Replay checks the result.'
)

# Only state initialization and result extraction are supplied by the host.
# The model must generate the recurrence that actually computes the path.
BODY_SYSTEM = '''Generate only the Python loop body for a collective stage solver.
Runtime c[i][w-1] is the capacity cost of transition FROM i-w TO i. c[0]=[]; every later row has widths 1..min(3,i). Infinite cost means forbidden transition.
Host initializes d=[(infinity,[])]*len(c), d[0]=(0,[]), and returns d[-1][1].
Compute each d[i] as the minimum (total_cost,width_list) from its predecessors using current costs. Visit i=1..len(c)-1 in order. The round bound is nonbinding here.
Only emit loops updating d. Allowed for,if, indexing,assignments,len,range,enumerate,min,list comprehensions. No def,return,import,helpers,comments. Use short names, about 70 tokens. Independent Replay checks the schedule.'''


def assemble_body(raw):
    """Apply the advertised wrapper, never repair the model's recurrence."""
    body = raw.strip()
    if body.startswith('```') and body.endswith('```'):
        body = '\n'.join(body.splitlines()[1:-1])
    return ('def solve(c,limit):\n'
            ' d=[(float("inf"),[])]*len(c)\n'
            ' d[0]=(0,[])\n'
            + '\n'.join(' ' + line for line in body.splitlines())
            + '\n return d[-1][1]\n')


def propose(problem, settings, feedback=None):
    body_mode = problem['step_bound_free']
    content = {'instance': problem}
    if feedback is not None:
        content['repair'] = feedback
    try:
        from agenthack import llm_call
        trace = llm_call(model=settings.model, provider='official-platform')
    except ImportError:
        trace = nullcontext(None)
    with trace as call:
        started = time.monotonic()
        try:
            raw, evidence = _chat_completion(
                BODY_SYSTEM if body_mode else SYSTEM,
                json.dumps(content, separators=(',', ':')),
                replace(settings, max_tokens=256 if body_mode else 512),
                temperature=0, allow_truncated=True)
        except LLMError as error:
            error.duration_ms = (time.monotonic() - started) * 1000
            raise
        redacted = bool(settings.api_key and settings.api_key in raw)
        patch = raw.replace(settings.api_key, '[REDACTED]') if redacted else raw
        source = assemble_body(patch) if body_mode else patch
        evidence.update(mode='generated_stage_solver',
            response_evidence={'format': 'python_loop_body' if body_mode else 'python_source', 'content': patch,
                               'credential_redacted': redacted},
            program_protocol='incoming_v1' if body_mode else 'outgoing_v1',
            generation_scope='search_recurrence' if body_mode else 'search_function',
            code_sha256=hashlib.sha256(source.encode()).hexdigest())
        if call is not None:
            usage = evidence.get('usage', {})
            call.set(input_tokens=usage.get('prompt_tokens', 0), output_tokens=usage.get('completion_tokens', 0))
            call.result({'source': source, 'sha256': evidence['code_sha256']})
        if redacted or evidence.get('finish_reason') == 'length':
            source = ''
        return source, evidence
