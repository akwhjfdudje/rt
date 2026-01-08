module {
  %A = rt.alloc() : tensor<128x128xf32>
  %B = rt.alloc() : tensor<128x128xf32>
  %C = rt.alloc() : tensor<128x128xf32>

  %C1 = rt.matmul %A, %B : tensor<128x128xf32>
}

