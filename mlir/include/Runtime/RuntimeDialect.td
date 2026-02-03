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
def MatMulOp : RT_Op<"matmul", [Pure, SameOperandsAndResultType]> {
  let summary = "Matrix multiplication between two square matrices";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// %A1 = rt.noise %A : tensor<...>
def NoiseOp : RT_Op<"noise", [Pure, SameOperandsAndResultType]> {
  let summary = "Noise a tensor";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

// %C = rt.add %A, %B : tensor<...>
def AddOp : RT_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Adds two tensors";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

// %A1 = rt.relu %A : tensor<...>
def ReLUOp : RT_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "Elementwise ReLU a tensor";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}

// %A1 = rt.softmax %A : tensor<...>
def SoftmaxOp: RT_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Row-wise softmax a tensor";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$result);
  let assemblyFormat = "$input attr-dict `:` type($result)";
}
