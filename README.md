# CausalDA
Code to reproduce the numerical experiments in the paper "Domain adaptation under structural causal models"
by Yuansi Chen and Peter B&uuml;hlmann.

Code written with Python 3.7.1 and PyTorch version as follows

torch==1.2.0
torchvision==0.4.0



## User Guide

- semiclass.py implements the main DA methods
  - semitorchclass,  semitorchstocclass implement the same functions with PyTorch
  - semitorchMNISTclass is tailored to convolutional neural nets
- Linear SCM simulations: first seven experiments
  - sim_linearSCM_mean_shift_exp1-7.ipynb can run on a single core and plot
- Linear SCM simulations: last two experiments
  - Run the following simulations in a computer cluster
    - sim_linearSCM_var_shift_exp8_box_submit.py
    - sim_linearSCM_var_shift_exp8_scat_submit.py
    - sim_linearSCM_var_shift_exp9_scat_submit.py
  - Read the results and plot with sim_linearSCM_variance_shift_exp8-9.ipynb
- MNIST experiments:
  - Need to set the MNIST data folder!
  - Single source exp: run submit_simu_MNIST_patches_2M.py
  - Mutiple source exp: run submit_simu_MNIST_patches.py
  - Read the results and plot with MNIST_read_and_plot_whitepatch2M.ipynb and MNIST_read_and_plot_rotation5M.ipynb
- Amazon review dataset experiments
  - Need to set the Amazon review data folder!
  - Preprocess the data with read_and_preprocess_amazon_review_data_2018_subset.ipynb
  - Run the simulations with submit_amazon_review_data_2018_subset_regression.py
  - Plot with amazon_read_and_plot.ipynb



## License and Citation
Code is released under MIT License.
Please cite our paper if the code helps your research.