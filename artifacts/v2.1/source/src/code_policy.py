"""Generate a single request-specific search program."""
import ast
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


FRONTIER_SYSTEM = (
    "Write one Python lambda only: lambda v,t: EXPRESSION. v is predecessor completion "
    "time and t is the FULL cost of the candidate communication stage. The host ranks "
    "alternatives by the return value (smaller first). Determine the end-to-end time "
    "after executing the two stages in sequence. No calls, Markdown, or explanations."
)


def compile_reduction(raw):
    """Turn a model-written reduction over predecessors into a bounded VM program."""
    fragment = raw.strip()
    if fragment.startswith('```') and fragment.endswith('```'):
        fragment = '\n'.join(fragment.splitlines()[1:-1]).strip()
    try:
        node = ast.parse(fragment, mode='eval').body
    except (SyntaxError, ValueError, RecursionError):
        return ''
    if isinstance(node, ast.Lambda):
        args = node.args
        if (args.posonlyargs or args.vararg or args.kwonlyargs or args.kwarg or args.defaults
                or [arg.arg for arg in args.args] != ['v', 't']):
            return ''
        reducer, score = 'min', node.body
    elif (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
          and node.func.id in ('min', 'max') and len(node.args) == 1 and not node.keywords):
        # Permit replay of code generated under either earlier expression protocol.
        reducer, score = node.func.id, node.args[0]
        if isinstance(score, ast.GeneratorExp):
            return assemble_frontier(fragment)
    else:
        return ''
    syntax = (ast.Name, ast.Load, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
              ast.Div, ast.UnaryOp, ast.UAdd, ast.USub, ast.Constant)
    nodes = list(ast.walk(score))
    names = {item.id for item in nodes if isinstance(item, ast.Name)}
    allowed_names = {'v', 't'} if isinstance(node, ast.Lambda) else {'v', 't', 'w'}
    if (len(nodes) > 24 or not {'v', 't'} <= names <= allowed_names
            or any(not isinstance(item, syntax) for item in nodes)
            or any(type(item.value) not in (int, float) for item in nodes if isinstance(item, ast.Constant))):
        return ''
    expression = ast.unparse(score)
    return assemble_frontier(f'{reducer}((({expression}),p+[w]) for v,p,w,t in a)')


def assemble_frontier(raw):
    """Wrap a model expression without changing its decision or arithmetic.

    The host owns state initialization and legal predecessor enumeration. The
    expression owns their combination and selection. Unsupported code is passed
    to the normal rejection/feedback path, never replaced with host search code.
    """
    text = raw.strip()
    if text.startswith('```'):
        text = '\n'.join(text.splitlines()[1:]).removesuffix('```').strip()
    try:
        tree = ast.parse(text, mode='eval')
        if not any(isinstance(n, ast.Name) and n.id == 'a' and isinstance(n.ctx, ast.Load)
                   for n in ast.walk(tree)):
            return ''
        expression = ast.unparse(tree.body)
    except (SyntaxError, ValueError, RecursionError):
        return ''
    return ('def solve(c,limit):\n'
            ' d=[(float("inf"),[])]*len(c)\n'
            ' d[0]=(0,[])\n'
            ' for i in range(1,len(c)):\n'
            '  a=[(d[i-w][0],d[i-w][1],w,t) for w,t in enumerate(c[i],1)]\n'
            '  d[i]=' + expression + '\n return d[-1][1]\n')


def propose(problem, settings, feedback=None):
    body_mode = problem['step_bound_free']
    content = dict(problem)
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
            user_input = json.dumps(content, separators=(',', ':'))
            if body_mode:
                user_input += '\nWrite the one-line recurrence now. Return code only.'
            raw, evidence = _chat_completion(
                FRONTIER_SYSTEM if body_mode else SYSTEM,
                user_input,
                replace(settings, max_tokens=32 if body_mode else 512),
                temperature=0, allow_truncated=True)
        except LLMError as error:
            error.duration_ms = (time.monotonic() - started) * 1000
            raise
        redacted = bool(settings.api_key and settings.api_key in raw)
        patch = raw.replace(settings.api_key, '[REDACTED]') if redacted else raw
        source = compile_reduction(patch) if body_mode else patch
        evidence.update(mode='generated_stage_solver',
            response_evidence={'format': 'stage_reduction_expression' if body_mode else 'python_source', 'content': patch,
                               'credential_redacted': redacted},
            program_protocol='incoming_v1' if body_mode else 'outgoing_v1',
            generation_scope='frontier_combination_and_selection' if body_mode else 'search_function',
            code_sha256=hashlib.sha256(source.encode()).hexdigest())
        if call is not None:
            usage = evidence.get('usage', {})
            call.set(input_tokens=usage.get('prompt_tokens', 0), output_tokens=usage.get('completion_tokens', 0))
            call.result({'source': source, 'sha256': evidence['code_sha256']})
        if redacted or evidence.get('finish_reason') == 'length':
            source = ''
        return source, evidence
