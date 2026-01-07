include "mlir/IR/OpBase.td"

def NoiseOp : RT_Op<"noise", []> {
    let summary = "Generates noise into a tensor";
    let arguments = (ins TensorType:$output, I32:$seed);
    let results = (outs TensorType:$output);
}

