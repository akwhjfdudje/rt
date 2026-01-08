include "mlir/IR/AttrTypeBase.td"

def RT_Dialect : Dialect {
  let name = "rt";
  let cppNamespace = "::mlir::rt";
}

class RT_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<RT_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TensorType : RT_Type<"Tensor", "tensor"> {
  let summary = "runtime tensor type";
  let assemblyFormat = "tensor<[$dims+]x$elementType>";
}
