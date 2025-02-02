import pytest

from tensorrt_llm.tools.plugin_gen.core import InputArg, Type
from tensorrt_llm.tools.plugin_gen.shape_infer import *


@pytest.mark.parametrize('expr', [
    "a[m,n,k]:m*2+k+(n+1) -> b",
    "a[m,n,k] -> b[m*n+k, 2*k, (3+1) * k]",
    "a[*] -> b[*]",
    "Q[m,n,k,*] : m -> batch_size",
])
def test_ToAst(expr: str):
    ast = parser.parse(expr)
    ast = ToAst()(ast)
    assert ast

    if isinstance(ast, DeduceDimSizeArgRule):
        assert ast.left
        assert ast.expr
        assert ast.right


@pytest.mark.parametrize('expr, target', [
    ("a[m,n,k]:m*2+k+(n+1) -> b",
     "((inputDesc[0].dims.d[0] * 2) + (inputDesc[0].dims.d[2] + (inputDesc[0].dims.d[1] + 1)))"
     ),
    ("a[m,n,k]:m*(2+k)+n+1 -> b",
     "((inputDesc[0].dims.d[0] * (2 + inputDesc[0].dims.d[2])) + (inputDesc[0].dims.d[1] + 1))"
     ),
    ("a[m,n,k] -> b[m*((((n+1))))]", """
if (outputIndex == 0) {
  outputDims.nbDims = 1;
  outputDims.d[0] = (inputDims[0].d[0] * (inputDims[0].dims.d[1] + 1));
}
     """),
    ("a[m,n,k] -> b[m*(n+k), 2*n, k+3]", """
nvinfer1::DimsExprs outputDims;
if (outputIndex == 0) {
  outputDims.nbDims = 3;
  outputDims.d[0] = (inputDims[0].d[0] * (inputDims[0].dims.d[1] + inputDims[0].dims.d[2]));
  outputDims.d[1] = (2 * inputDims[0].dims.d[1]);
  outputDims.d[2] = (inputDims[0].dims.d[2] + 3);
}""")
])
def test_CppCodeTranspiler(expr: str, target: str):
    args = dict(
        a=InputArg('a', Type('fp16')),
        b=InputArg('b', Type('fp16')),
   
