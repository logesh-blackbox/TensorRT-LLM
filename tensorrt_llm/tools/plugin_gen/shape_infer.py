from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple, Union

from lark import Lark, Token, Tree

if TYPE_CHECKING:
    from tensorrt_llm.tools.plugin_gen.core import Argument

parser = Lark(r"""
value: SIGNED_NUMBER
      | name
      | expr
      | "(" expr ")"

expr: value "+" value -> add
    | value "-" value -> sub
    | value "*" value -> mul
    | value "/" value -> div
    | value

shaped_tensor: name "[" value ("," value)* ("," "*")? "]" -> tensor
      | name "[" "*" "]" -> wildcard_tensor

tensors: shaped_tensor ("," shaped_tensor)*

deduce_shape: tensors "->" tensors

deduce_dim_size_arg: tensors ":" expr "->" name

name: CNAME
?start: deduce_shape | deduce_dim_size_arg

%import common.SIGNED_NUMBER
%import common.WS
%import common.CNAME
%ignore WS
""".strip())


# Here we introduce a set of ASTs to represent the target's expression.
# The Ast nodes from lark is not convenient to use.
class _AST:
    pass


@dataclass
class NumberAST(_AST):
    value: int


@dataclass
class BinaryAST(_AST):
    op: str
    left: _AST
    right: _AST


@dataclass
class ShapeAST:
    dims: List[_AST]


@dataclass
class DimAST(_AST):
    name: str


@dataclass
class ShapedTensorAST(_AST):
    arg_name: str
    shape: ShapeAST


@dataclass
class DeduceShapeRule(_AST):
    left: List[ShapedTensorAST]
    right: List[ShapedTensorAST]


@dataclass
class DeduceDimSizeArgRule(_AST):
    left: List[ShapedTensorAST]
    expr: _AST
    right: str


class ToAst:

    def __call__(self,
                 tree: Tree) -> Union[DeduceShapeRule, DeduceDimSizeArgRule]:
        if tree.data == "deduce_shape":
            assert len(tree.children) == 2
            return self.visit_DeduceShape(tree.children[0], tree.children[1])
        elif tree.data == "deduce_dim_size_arg":
            assert len(tree.children) == 3
            return self.visit_DeduceDimSizeArg(tree.children[0],
                                               tree.children[1],
                                               tree.children[2])
        raise NotImplementedError()

    def visit_DeduceShape(self, left: Tree, right: Tree) -> DeduceShapeRule:
        assert left.data == "tensors"
        assert right.data == "tensors"

        lefts = self.visit_tensors(left)
        rights = self.visit_tensors(right)
        return DeduceShapeRule(lefts, rights)

    def visit_tensors(self, tree: Tree) -> List[ShapedTensorAST]:
        assert tree.data == "tensors", repr(tree)
        return [self.visit_tensor(child) for child in tree.children]

    def visit_tensor(self, tree: Tree) -> ShapedTensorAST:
        if tree.data == "tensor":
            arg_name = self.visit_name(tree.children[0])
            dims = [self.visit_expr(child) for child in tree.children[1:]]
            return ShapedTensorAST(arg_name, ShapeAST(dims))

        assert tree.data == "wildcard_tensor", repr(tree)
        arg_name = self.visit_name(tree.children[0])
        return ShapedTensorAST(arg_name, ShapeAST([DimAST("*")]))

    def visit_number(self, v: str) -> _AST:
        return NumberAST(int(v))

    def visit_expr(self, tree: Tree) -> _AST:
        '''
        for expression of dims, like `m * 2 + 1`
        '''

        def visit(tree: Union[Tree, Token]) -> _AST:
            if isinstance(tree, Token):
                if tree.type == "SIGNED_NUMBER":
                    return NumberAST(int(tree.value))
                elif tree.type == "CNAME":
                    return DimAST(tree.value)
                raise ValueError("Unexpected token: %s" % tree)

            elif isinstance(tree.data, Token):  # RULE; CNAME
                tree_type = tree.data.value
                if tree_type == 'name':
                    return DimAST(tree.children[0].value)
                elif tree_type == 'value':
                    return visit(tree.children[0])
                elif tree_type == 'expr':
                    return visit(tree.children[0])
                elif tree.data == "SIGNED_NUMBER":
                    return NumberAST(int(tree.children[0].data))
                else:
                    raise ValueError(f"Unexpected tree: {repr(tree)}")

            elif tree.data == "add":
                assert len(tree.children) == 2
                return BinaryAST("+", visit(tree.children[0]),
                                 visit(tree.children[1]))
            elif tree.data == "sub":
                assert len(tree.children) == 2
                return BinaryAST("-", visit(tree.children[0]),
                                 visit(tree.children[1]))
            elif tree.data == "mul":
                assert len(tree.children) == 2
                return BinaryAST("*", visit(tree.children[0]),
                                 visit(tree.children[1]))
            elif tree.data == "div":
                assert len(tree.children) == 2
               
