% This scripts assumes that python and the follownig packages are
% installed (available for each platform): argparse numpy
% If you have python and package manager pip installed, you can simply
% install the packages using: pip install argparse numpy
%
% Also, check that the relative paths are correct.

system('python ../external/tum-evaluation/evaluate_ate.py camera_poses.txt output_poses.txt --plot example_ate.pdf')
system('python ../external/tum-evaluation/evaluate_rpe.py camera_poses.txt output_poses.txt --plot example_rpe.pdf --fixed_delta')