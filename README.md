# CausalDA
Code to reproduce the numerical experiments in the paper "Domain adaptation under structural causal models"
by Yuansi Chen and Peter B&uuml;hlmann.

Code written with Python 3.7.1 and Pytorch version as follows

torch==1.2.0
torchvision==0.4.0



## User Guide

- Linear SCM simulations: first seven experiments
  - sim_linearSCM_mean_shift_exp1-7.ipynb can run on a single core and plot
- Linear SCM simulations: last two experiments
  - Run the following simulations in a computer cluster
    - sim_linearSCM_var_shift_exp8_box_submit.py
    - sim_linearSCM_var_shift_exp8_scat_submit.py
    - sim_linearSCM_var_shift_exp9_scat_submit.py
  - Read the results and plot with sim_linearSCM_variance_shift_exp8-9.ipynb
- Amazon review dataset experiments
  - Run the simulations with submit_amazon_review_data_2018_subset_regression.py
  - Plot with amazon_read_and_plot.ipynb



## License and Citation
Code is released under MIT License.
Please cite our paper if the code helps your research.