module {
  %A = rt.alloc : tensor<128x128xf32>

  %A1 = rt.noise %A : tensor<128x128xf32>

  %A2 = rt.softmax %A1 : tensor<128x128xf32>
}
