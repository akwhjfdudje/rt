include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/AttrTypeBase.td"

def RT_Dialect : Dialect {
  let name = "rt";
  let cppNamespace = "::mlir::rt";
}

class RT_Op<string mnemonic, list<Trait> traits = []>
    : Op<RT_Dialect, mnemonic, traits>;

// %A = rt.alloc : tensor<...>
def AllocOp : RT_Op<"alloc", [Pure]> {
  let summary = "Allocate a tensor";
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}


// %C = rt.matmul %A, %B : tensor<...>
def MatMulOp : RT_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

