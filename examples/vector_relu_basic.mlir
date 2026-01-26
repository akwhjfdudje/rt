module {
  %A = rt.alloc : tensor<128x128xf32>
  %B = rt.alloc : tensor<128x128xf32>
  %C = rt.alloc : tensor<128x128xf32>

  %A1 = rt.noise %A : tensor<128x128xf32>
  %A2 = rt.relu %A1 : tensor<128x128xf32>

  %B1 = rt.noise %B : tensor<128x128xf32>
  %B2 = rt.relu %B1 : tensor<128x128xf32>

  %C1 = rt.add %A2, %B2 : tensor<128x128xf32>
}
