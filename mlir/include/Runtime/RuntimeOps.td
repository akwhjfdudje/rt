include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinOps.td"
include "./RuntimeTypes.td"

class RT_Op<string mnemonic, list<Trait> traits = []>
    : Op<RT_Dialect, mnemonic, traits>;

///   %A = rt.alloc() : tensor<128x128xf64>
def AllocOp : RT_Op<"alloc", [Pure]> {
  let summary = "Allocate a runtime tensor";
  let arguments = (ins);
  let results   = (outs TensorType:$result);
}

def MatrixMulOp : RT_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication op";
  let arguments = (ins TensorType:$lhs, TensorType:$rhs);
  let results   = (outs TensorType:$out);
}


