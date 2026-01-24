module {
  %A = rt.alloc : tensor<128x128xf32>
  %B = rt.alloc : tensor<128x128xf32>
  %C = rt.alloc : tensor<128x128xf32>

  %A1 = rt.noise %A : tensor<128x128xf32>

  %C1 = rt.matmul %A1, %B : tensor<128x128xf32>
}
