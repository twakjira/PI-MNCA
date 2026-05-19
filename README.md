<div align="center">

# PI-MNCA

---

### Physics-Informed Modern Neighborhood Component Attention for Bond Strength Prediction of FRP Bars Embedded in Ultra-High-Performance Concrete

<a href="https://twakjira.github.io/PI-MNCA/" target="_blank" rel="noopener noreferrer"><img alt="Project page" src="https://img.shields.io/badge/PROJECT-PAGE-blue?style=for-the-badge"></a>
<a href="#" target="_blank" rel="noopener noreferrer"><img alt="Paper" src="https://img.shields.io/badge/PAPER-UNDER%20REVIEW-red?style=for-the-badge"></a>

<a href="https://ai4riselab.com" target="_blank" rel="noopener noreferrer">Tadesse G. Wakjira</a>

*Under Review*

</div>

---

## Overview

PI-MNCA is a physics-informed Modern Neighborhood Component Attention
regressor for the ultimate pullout bond strength <em>&tau;<sub>u</sub></em>
of fiber-reinforced polymer (FRP) bars embedded in ultra-high-performance
concrete (UHPC). The framework couples a 128-dimensional learned
embedding with soft kernel attention over a fixed memory bank of
training specimens, an ACI-style target normalization
<em>y</em> = ln(<em>&tau;<sub>u</sub></em>/&radic;<em>f<sub>c</sub></em>)
that is invariant under proportional concrete-strength scaling, a soft
monotonicity penalty that suppresses local violations of four
mechanically known directions
(<em>f<sub>c</sub></em>&uarr;, <em>l/d</em>&darr;, <em>&rho;<sub>SF</sub></em>&uarr;, <em>c/d</em>&uarr;),
dimension-symmetry feature augmentation, and split-conformal
calibration on real out-of-fold residuals.

On a held-out 20% test partition of the curated 475-specimen FRP-UHPC
pullout database (15 independent source programs), the model achieves
R&sup2; = 0.914, RMSE = 2.95 MPa, MAE = 2.05 MPa, MAPE = 6.9%, and a
conformal 90% prediction band of &plusmn;9.16 MPa. Every prediction is
traceable to the specific training specimens that contributed to it
through the attention weights, enabling case-level interpretability for
engineering use.

An interactive in-browser deployment of the model is available at the
<a href="https://twakjira.github.io/PI-MNCA/" target="_blank" rel="noopener noreferrer">project page</a>.

## Status

The manuscript is currently under review. Full source code, training scripts, and trained weights will be released upon paper acceptance.

## Architectural contributions

| | Element |
|---|---|
| R1 | Modern NCA soft kernel attention over a fixed memory bank of training specimens in a learned 128-d embedding space |
| R2 | ACI-style target normalization <em>y</em> = ln(<em>&tau;<sub>u</sub></em>/&radic;<em>f<sub>c</sub></em>) invariant under proportional concrete-strength scaling |
| R3 | Soft physics-informed monotonicity penalty along four mechanically known directions (<em>f<sub>c</sub></em>&uarr;, <em>l/d</em>&darr;, <em>&rho;<sub>SF</sub></em>&uarr;, <em>c/d</em>&uarr;) |
| R4 | Dimension-symmetry feature augmentation that exploits the geometric invariances of the pullout problem |
| R5 | Split-conformal calibration on real out-of-fold residuals delivering a distribution-free 90% prediction band |
| R6 | Case-level interpretability through attention weights that trace every prediction back to the supporting training specimens |

## Citation

```
@article{wakjira2026pimnca,
    title   = {Physics-Informed Modern Neighborhood Component Attention
               for Bond Strength Prediction of FRP Bars Embedded in
               Ultra-High-Performance Concrete},
    author  = {Wakjira, Tadesse G.},
    journal = {Under Review},
    year    = {2026}
}
```

## Authors

<strong><a href="https://ai4riselab.com" target="_blank" rel="noopener noreferrer">Tadesse G. Wakjira</a></strong>

## Development

Developed by <a href="https://ai4riselab.com" target="_blank" rel="noopener noreferrer">AI4RISE Lab</a>
