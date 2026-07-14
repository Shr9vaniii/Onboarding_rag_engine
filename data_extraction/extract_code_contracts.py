import ast


HTTP_METHODS = {
    "get", "post", "put", "delete",
    "patch", "options", "head"
}


# =====================================================
# Helpers
# =====================================================

def extract_name(node):
    if node is None:
        return None

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = extract_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr

    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            pass

    return None


def get_annotation(node):
    if node is None:
        return None

    if hasattr(ast, "unparse"):
        return ast.unparse(node)

    return extract_name(node)


def literal(node):
    if node is None:
        return None

    if isinstance(node, ast.Constant):
        return node.value

    if hasattr(ast, "unparse"):
        try:
            return ast.unparse(node)
        except Exception:
            pass

    return None

def get_default_value(node) -> str | None:
    """Converts an AST default node to a readable string."""
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return repr(node.value)
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{node.value.id}.{node.attr}"
    # For complex defaults just mark as present
    return "..."

import re


def extract_annotated_doc(type_str: str) -> tuple[str, str]:
    """
    Splits "Annotated[bool, Doc('...description...')]"
    into ("bool", "description")
    Returns (clean_type, doc_text) — doc_text is "" if no Doc present.
    """
    if type_str is None:
        return "", ""
    if not type_str.startswith("Annotated["):
        return type_str, ""

    # Extract the Doc content
    doc_match = re.search(r"Doc\(['\"]+(.*?)['\"]+\)", type_str, re.DOTALL)
    doc_text = ""
    if doc_match:
        doc_text = doc_match.group(1).strip().replace("\\n", " ").strip()
        # collapse whitespace
        doc_text = re.sub(r"\s+", " ", doc_text)

    # Extract the clean type (first arg of Annotated)
    inner = type_str[len("Annotated["):-1]  # strip Annotated[ and ]
    clean_type = inner.split(",")[0].strip()

    return clean_type, doc_text


# =====================================================
# Function Body Analyzer
# =====================================================

class FunctionAnalyzer(ast.NodeVisitor):

    def __init__(self):
        self.calls = set()
        self.raises = set()
        self.dependencies = set()

    def visit_Call(self, node):

        name = extract_name(node.func)

        if name:
            self.calls.add(name)

            if name.endswith("Depends"):
                if node.args:
                    dep = extract_name(node.args[0])
                    if dep:
                        self.dependencies.add(dep)

        self.generic_visit(node)

    def visit_Raise(self, node):

        if node.exc:

            if isinstance(node.exc, ast.Call):
                exc = extract_name(node.exc.func)
            else:
                exc = extract_name(node.exc)

            if exc:
                self.raises.add(exc)

        self.generic_visit(node)


# =====================================================
# Main Extractor
# =====================================================

class CodeContractExtractor(ast.NodeVisitor):

    def __init__(self):

        self.contract = {

            "imports": [],
            "symbols": {},

            "classes": [],
            "functions": []
        }

    # -------------------------
    # Imports
    # -------------------------

    def visit_Import(self, node):

        for alias in node.names:

            self.contract["imports"].append({

                "module": alias.name,

                "symbols": []
            })

    def visit_ImportFrom(self, node):

        self.contract["imports"].append({

            "module": node.module,

            "symbols": [
                alias.name
                for alias in node.names
            ]
        })

    # -------------------------
    # Classes
    # -------------------------

    def visit_ClassDef(self, node):

        self.contract["symbols"][node.name] = "class"

        cls = {

            "name": node.name,

            "bases": [
                extract_name(base)
                for base in node.bases
            ],

            "attributes": [],

            "methods": [],

            "line": node.lineno
        }

        doc = ast.get_docstring(node)

        if doc:
            cls["docstring"] = doc

        for item in node.body:

            if isinstance(item, ast.AnnAssign):

                cls["attributes"].append({

                    "name": extract_name(item.target),

                    "type": get_annotation(item.annotation)
                })

            elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):

                cls["methods"].append(item.name)

        self.contract["classes"].append(cls)

    # -------------------------
    # Functions
    # -------------------------

    def visit_FunctionDef(self, node):
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node, is_async):

        self.contract["symbols"][node.name] = "function"

        analyzer = FunctionAnalyzer()
        analyzer.visit(node)

        func = {

            "name": node.name,

            "async": is_async,

            "args": [],

            "decorators": self._parse_decorators(node),

            "calls": sorted(
                c for c in analyzer.calls
                if c not in {
                    "str",
                    "int",
                    "dict",
                    "list",
                    "set",
                    "tuple",
                    "len",
                    "print"
                }
            ),

            "raises": sorted(analyzer.raises),

            "dependencies": sorted(analyzer.dependencies),

            "line": node.lineno
        }

        doc = ast.get_docstring(node)

        if doc:
            func["docstring"] = doc

        defaults=node.args.defaults
        args_list=node.args.args
        offset=len(args_list)-len(defaults)

        for arg in node.args.args:
            raw_types = get_annotation(arg.annotation)
            cleaned,doc=extract_annotated_doc(raw_types)
            arg_index=args_list.index(arg)
            default_node=defaults[arg_index-offset] if arg_index>=offset else None
            default_value=get_default_value(default_node)
            

            func["args"].append({

                "name": arg.arg,
                "type": cleaned,
                "doc": doc,
                "default": default_value
            })

        if node.returns:
            func["return_type"] = get_annotation(node.returns)

        self.contract["functions"].append(func)

    # -------------------------
    # Decorators
    # -------------------------

    def _parse_decorators(self, node):

        decorators = []

        for dec in node.decorator_list:

            if isinstance(dec, ast.Call):

                name = extract_name(dec.func)

                info = {
                    "name": name
                }

                # FastAPI routes
                if isinstance(dec.func, ast.Attribute):

                    method = dec.func.attr.lower()

                    if method in HTTP_METHODS:

                        info["http_method"] = method.upper()

                        if dec.args:
                            route = literal(dec.args[0])

                            if route:
                                info["route"] = route

                        for kw in dec.keywords:

                            if kw.arg == "response_model":
                                info["response_model"] = extract_name(
                                    kw.value
                                )

                            elif kw.arg == "tags":
                                info["tags"] = literal(kw.value)

                            elif kw.arg == "summary":
                                info["summary"] = literal(kw.value)

                decorators.append(info)

            else:

                decorators.append({
                    "name": extract_name(dec)
                })

        return decorators


# =====================================================
# Public API
# =====================================================

def extract_contract_from_source(source_code: str,store_element_source: bool = False):

    tree = ast.parse(source_code)

    extractor = CodeContractExtractor()

    extractor.visit(tree)
    if store_element_source:
        source_lines = source_code.splitlines()
        for func in extractor.contract["functions"]:
            start=func["line"]-1
            end=func.get("end_line",start+20)
            func["source_code"]="\n".join(source_lines[start:end])

    return extractor.contract

