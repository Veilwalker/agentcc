"""A bounded interpreter for model-written numerical Python functions.

No Python eval/exec, imports, object introspection, files, network, or host
functions are exposed. AST steps and all collection sizes are bounded. This
is a small language, not a claim to sandbox arbitrary Python programs.
"""
import ast
import math
import operator


class ProgramError(ValueError):
    pass


class _Returned(Exception):
    def __init__(self, value):
        self.value = value


class Program:
    MAX_ITEMS = 4096
    OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
    CMP = {ast.Lt: operator.lt, ast.LtE: operator.le, ast.Gt: operator.gt,
           ast.GtE: operator.ge, ast.Eq: operator.eq, ast.NotEq: operator.ne,
           ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
           ast.Is: operator.is_, ast.IsNot: operator.is_not}

    def __init__(self, source):
        if not isinstance(source, str) or len(source) > 8000:
            raise ProgramError('source must be at most 8000 characters')
        text = source.strip()
        if text.startswith('```') and text.endswith('```'):
            text = '\n'.join(text.splitlines()[1:-1])
        self.source = text
        try:
            tree = ast.parse(text)
        except (SyntaxError, RecursionError, ValueError):
            raise ProgramError('invalid Python syntax') from None
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ProgramError('write only def solve(c, limit)')
        self.function = f = tree.body[0]
        if (f.name != 'solve' or [a.arg for a in f.args.args] != ['c', 'limit']
                or f.decorator_list or f.args.defaults or f.args.kw_defaults
                or f.args.kwarg or f.args.vararg or f.args.posonlyargs or f.args.kwonlyargs
                or f.returns or any(a.annotation for a in f.args.args)):
            raise ProgramError('signature must be solve(c, limit) without annotations')
        allowed = (ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Load, ast.Store,
            ast.Assign, ast.AugAssign, ast.Expr, ast.Return, ast.If, ast.For,
            ast.Continue, ast.Break, ast.While, ast.Pass, ast.Name, ast.Constant, ast.List, ast.Tuple, ast.Dict,
            ast.Subscript, ast.Slice, ast.Call, ast.Attribute, ast.BinOp, ast.UnaryOp,
            ast.USub, ast.UAdd, ast.Not, ast.Compare, ast.BoolOp, ast.And, ast.Or,
            ast.IfExp, ast.ListComp, ast.GeneratorExp, ast.comprehension, ast.keyword,
            *self.OPS, *self.CMP)
        nodes = list(ast.walk(tree))
        if any(isinstance(n, ast.Lambda) for n in nodes):
            raise ProgramError('lambda is not allowed; min compares (cost,path) tuples directly without key=')
        if len(nodes) > 1500 or any(not isinstance(n, allowed) for n in nodes):
            raise ProgramError('unsupported syntax; use assignments, for, if, return and arithmetic')
        for n in nodes:
            if isinstance(n, ast.Attribute) and n.attr not in ('append', 'get'):
                raise ProgramError('only list.append and dict.get methods are allowed')
            if isinstance(n, ast.Name) and n.id.startswith('__'):
                raise ProgramError('private names are not allowed')
            if isinstance(n, ast.comprehension) and n.is_async:
                raise ProgramError('async syntax is not allowed')
        loaded = {n.id for n in nodes if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if 'c' not in loaded:
            raise ProgramError('solver must read runtime costs c')

    def run(self, costs, limit):
        def freeze(value):
            return tuple(freeze(x) for x in value) if isinstance(value, (tuple, list)) else value
        self.fuel, self.steps, self.env = 300_000, 0, {'c': freeze(costs), 'limit': limit}
        try:
            self.block(self.function.body)
        except _Returned as result:
            return result.value
        except ProgramError:
            raise
        except (TypeError, ValueError, KeyError, IndexError, ZeroDivisionError,
                OverflowError, RecursionError, StopIteration):
            raise ProgramError('numerical program failed; check indexing, types and empty states') from None
        raise ProgramError('solver did not return a result')

    def tick(self):
        self.fuel -= 1
        self.steps += 1
        if self.fuel < 0:
            raise ProgramError('program exceeded instruction budget')

    def bound(self, value):
        if isinstance(value, (list, tuple, dict, range, str)) and len(value) > self.MAX_ITEMS:
            raise ProgramError('collection exceeded size limit')
        if type(value) is int and abs(value) > 10**12:
            raise ProgramError('integer exceeded size limit')
        if type(value) is float and math.isnan(value):
            raise ProgramError('NaN is not permitted')
        # Reject cycles and exponentially nested aliases before Python's
        # comparison/min/sum primitives can traverse them outside the VM.
        stack = [(value, 0, frozenset())]
        count = 0
        while stack:
            item, depth, parents = stack.pop()
            count += 1
            if count > 16384 or depth > 16:
                raise ProgramError('nested value exceeded size limit')
            if isinstance(item, (list, tuple, dict)):
                if id(item) in parents:
                    raise ProgramError('cyclic value is not allowed')
                ancestors = parents | {id(item)}
                values = item.values() if isinstance(item, dict) else item
                stack.extend((x, depth+1, ancestors) for x in values)
        return value

    def assign(self, target, value):
        self.tick()
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, (ast.List, ast.Tuple)):
            if len(target.elts) != len(value):
                raise ProgramError('unpacking size differs')
            for t, v in zip(target.elts, value):
                self.assign(t, v)
        elif isinstance(target, ast.Subscript):
            container, key = self.expr(target.value), self.expr(target.slice)
            if not isinstance(container, (dict, list)):
                raise ProgramError('invalid assignment container')
            container[key] = value
            self.bound(container)
        else:
            raise ProgramError('unsupported assignment')

    def block(self, statements):
        for node in statements:
            self.tick()
            if isinstance(node, ast.Assign):
                value = self.expr(node.value)
                for target in node.targets:
                    self.assign(target, value)
            elif isinstance(node, ast.AugAssign):
                self.assign(node.target, self.binary(node.op, self.expr(node.target), self.expr(node.value)))
            elif isinstance(node, ast.Return):
                raise _Returned(self.expr(node.value))
            elif isinstance(node, ast.Expr):
                self.expr(node.value)
            elif isinstance(node, ast.If):
                signal = self.block(node.body if self.expr(node.test) else node.orelse)
                if signal:
                    return signal
            elif isinstance(node, ast.For):
                values = self.bound(self.expr(node.iter))
                for value in values:
                    self.assign(node.target, value)
                    if self.block(node.body) == 'break':
                        break
                else:
                    self.block(node.orelse)
            elif isinstance(node, ast.While):
                while self.expr(node.test):
                    if self.block(node.body) == 'break':
                        break
                else:
                    self.block(node.orelse)
            elif isinstance(node, ast.Continue):
                return 'continue'
            elif isinstance(node, ast.Break):
                return 'break'
            elif not isinstance(node, ast.Pass):
                raise ProgramError('unsupported statement')

    def binary(self, op, a, b):
        if isinstance(op, ast.Mult):
            if isinstance(a, (list, tuple, str)) and type(b) is int and len(a) * b > self.MAX_ITEMS:
                raise ProgramError('repetition exceeded size limit')
            if isinstance(b, (list, tuple, str)) and type(a) is int and len(b) * a > self.MAX_ITEMS:
                raise ProgramError('repetition exceeded size limit')
        return self.bound(self.OPS[type(op)](a, b))

    def expr(self, node):
        self.tick()
        if node is None:
            return None
        if isinstance(node, ast.Constant):
            if type(node.value) not in (int, float, str, bool, type(None)):
                raise ProgramError('unsupported literal')
            return self.bound(node.value)
        if isinstance(node, ast.Name):
            return self.env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            result = [self.expr(v) for v in node.elts]
            return self.bound(tuple(result) if isinstance(node, ast.Tuple) else result)
        if isinstance(node, ast.Dict):
            return self.bound({self.expr(k): self.expr(v) for k, v in zip(node.keys, node.values)})
        if isinstance(node, ast.BinOp):
            return self.binary(node.op, self.expr(node.left), self.expr(node.right))
        if isinstance(node, ast.UnaryOp):
            return self.bound({ast.USub: operator.neg, ast.UAdd: operator.pos, ast.Not: operator.not_}[type(node.op)](self.expr(node.operand)))
        if isinstance(node, ast.Slice):
            return slice(self.expr(node.lower), self.expr(node.upper), self.expr(node.step))
        if isinstance(node, ast.Subscript):
            return self.bound(self.expr(node.value)[self.expr(node.slice)])
        if isinstance(node, ast.Compare):
            a = self.expr(node.left)
            for op, other in zip(node.ops, node.comparators):
                b = self.expr(other)
                if not self.CMP[type(op)](a, b):
                    return False
                a = b
            return True
        if isinstance(node, ast.BoolOp):
            for child in node.values:
                value = self.expr(child)
                if (isinstance(node.op, ast.And) and not value) or (isinstance(node.op, ast.Or) and value):
                    return value
            return value
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test) else node.orelse)
        if isinstance(node, (ast.ListComp, ast.GeneratorExp)):
            saved, out = self.env.copy(), []
            def walk(index):
                if index == len(node.generators):
                    out.append(self.expr(node.elt))
                    self.bound(out)
                    return
                gen = node.generators[index]
                for value in self.bound(self.expr(gen.iter)):
                    self.assign(gen.target, value)
                    if all(self.expr(condition) for condition in gen.ifs):
                        walk(index + 1)
            walk(0)
            self.env = saved
            return out
        if isinstance(node, ast.Call):
            args = [self.expr(arg) for arg in node.args]
            if node.keywords:
                if (not isinstance(node.func, ast.Name) or node.func.id != 'enumerate'
                        or len(node.keywords) != 1 or node.keywords[0].arg != 'start'
                        or len(args) != 1):
                    raise ProgramError('only enumerate(start=integer) is supported')
                args.append(self.expr(node.keywords[0].value))
            if isinstance(node.func, ast.Attribute):
                obj = self.expr(node.func.value)
                if node.func.attr == 'append' and isinstance(obj, list) and len(args) == 1:
                    if args[0] is obj:
                        raise ProgramError('self-reference is not allowed')
                    obj.append(args[0])
                    self.bound(obj)
                    return None
                if node.func.attr == 'get' and isinstance(obj, dict) and 1 <= len(args) <= 2:
                    return obj.get(*args)
                raise ProgramError('unsupported method')
            if not isinstance(node.func, ast.Name):
                raise ProgramError('unsupported call')
            functions = {'len': len, 'range': range, 'min': min, 'max': max,
                         'sum': sum, 'abs': abs, 'int': int, 'float': float,
                         'list': list, 'tuple': tuple, 'enumerate': lambda x, start=0: list(enumerate(x,start)),
                         'reversed': lambda x: list(reversed(x))}
            if node.func.id not in functions:
                raise ProgramError('unsupported function')
            return self.bound(functions[node.func.id](*args))
        raise ProgramError('unsupported expression')
