include "mlir/IR/OpBase.td"

def MatrixMulOp : Runtime_Op<"matrix.mul", [
    DeclareOpInterfaceMethods<RuntimeInvokeInterface>
]> {
    let summary = "Matrix multiplication runtime op";
    let description = [{
        Calls into the gpu_runtime adapter: rt_matrixMul.
        Assumes 2D tensors and square shape NxN.
    }];

    let arguments = (ins
        Runtime_Tensor:$A,
        Runtime_Tensor:$B,
        Runtime_Tensor:$C
    );

    let results = (outs);
}

