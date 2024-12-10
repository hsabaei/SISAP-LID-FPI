<h1><b>Local Intrinsic Dimensionality and the Convergence Order of Fixed-Point Iteration</b></h1>

<p>This repository contains the implementation of the methods and experiments described in the paper <i>"Local Intrinsic Dimensionality and the Convergence Order of Fixed-Point Iteration"</i>, presented at <b>SISAP 2024</b>.</p>

<hr>

<h2><b>Overview</b></h2>

<p>The paper explores the relationship between the <b>convergence order of fixed-point iteration (FPI)</b> and the <b>local intrinsic dimensionality (LID)</b> of its update function. It introduces <b>novel LID-based estimators</b> for the convergence order of FPI and demonstrates their effectiveness compared to traditional methods.</p>

<h3><b>Key Contributions</b></h3>
<ul>
  <li><b>Theoretical Equivalence:</b> Establishing a connection between LID and FPI convergence order.</li>
  <li><b>Novel Estimators:</b> Adapting LID estimation methods, particularly the <i>MLE (Hill) estimator</i>, for convergence order estimation.</li>
  <li><b>Experimental Analysis:</b> Comparing the proposed methods with traditional techniques on various tasks.</li>
</ul>

<hr>

<h2><b>Repository Contents</b></h2>
<p>This repository contains:</p>
<ul>
  <li><b>Core implementation of algorithms</b> (<code>estimators.py</code>):
    <ul>
      <li>LID-based estimators: <i>FIE (Fixed-point Iteration Estimator)</i>, <i>GIE (General Iteration Estimator)</i>, and <i>Bayesian-COE</i>.</li>
      <li>Traditional estimators: <i>Log-Log</i>, <i>Iterative Ratio (IR)</i>.</li>
    </ul>
  </li>
  <li><b>Experiment code</b> to reproduce the results in the paper:
    <ul>
      <li><b>Root-finding tasks:</b> <code>NR_Exp_CO2.py</code>, <code>NR_Poly_CO2.py</code>, <code>NR_Poly_CO1.py</code>, <code>SM_Exp.py</code>, and <code>SM_Poly.py</code>.</li>
      <li><b>General iteration tasks:</b> <code>Monomial.py</code>, <code>Polynomial.py</code>, and <code>Trigonometric.py</code>.</li>
      <li><b>Machine Learning tasks:</b> <code>Quadratic_Regression.py</code>.</li>
    </ul>
  </li>
</ul>

<hr>

<h2><b>Requirements</b></h2>
<p>To run the code, you need:</p>
<ul>
  <li><b>Python >= 3.8</b></li>
</ul>
  <hr>
  
<h2><b>Citation</b></h2>
<p>If you use this repository, please cite our paper:</p>
<pre><code>
@inproceedings{houle2024local,
  title={Local Intrinsic Dimensionality and the Convergence Order of Fixed-Point Iteration},
  author={Houle, Michael E and Oria, Vincent and Sabaei, Hamideh},
  booktitle={International Conference on Similarity Search and Applications},
  pages={193--206},
  year={2024},
  organization={Springer}
}
</code></pre>

<hr>

<h2><b>Contact</b></h2>
<p>For any questions or issues, please open a GitHub issue or contact the authors.</p>
