**Python Implementation of the paper Likelihood-free Out-of-Distribution Detection with Invertible Generative Models**
Amirhossein Ahmadian and Fredrik Lindsten
International Joint Conference on Artificial Intelligence (IJCAI) 2021

# Requirements

..*PyTorch (1.1)

..*Torchvision (0.3)

..*Scipy (1.3)

..*Scikit-learn (0.22)

..*Matplotlib

..*FLIF image compressor (‘flif’ command line tool, see https://github.com/FLIF-hub/FLIF)

# Running OOD Detection Experiments
To run experiments and obtain performance results like the ones presented in the paper, you need to follow these three main steps:
1. Feeding a dataset to a pretrained invertible generative model to obtain the basic necessary values, that are the latent codes, derivatives, and likelihood
2. Using the values from the previous step and data to compute a set of ‘statistics’ for each data point
3. Running an OOD detection algorithm that uses the statistics as input to obtain the performance measures (AUROC, etc.) and error curves

##Step 1
To work with the models GLOW, IresNet, and ResFlow, use the scripts in the directories *GlowProcs*, *Iresnet_Procs* and *Resflow_Procs* respectively. You need to have a pretrained PyTorch model file for each case (it could be helpful to visit the original GitHub repository of each of these models). 
Run the *Run_model.py* script from the corresponding directory on command line. e.g., for GLOW:
python Run_model.py -modelpath … -trainedon … -teston … -n … -outfile … 
You could see the source file for more details about the arguments and supported datasets. A single output file is generated after running the script. It is recommended that you name the output file (*-outfile* argument) as ‘model_dataset1_dataset2.dat’ where ‘model’ is the model name, ‘dataset1’ is the data that the model has been trained on, and ‘dataset2’ is the data that the model is tested on (i.e. the output values are computed on).
##Step2
You need to compute the statistics required in OOD detection for each of the files obtained in step 1. To do so, use the *Compute_stats.py* script in *OOD_Detection*. There are several statistics that this script can compute, some of which are not used for the paper results (see the header comments). First, look at the *to_compute* set defined in *Compute_stats.py* to ensure that it contains the statistics that you are interested to obtain. Then, run the script using the command line.
python Compute_stats.py input output
where *input* is the path to a file (.dat) that you have obtained in the previous step. It is recommended that you save the statistics to a file (*output* argument) with the same name as the input file but with ‘.sts’ extension. 
The ‘complexity based statistic’ (T_2 in the paper) is an exception, and another script should be used to compute it. If you need this statistic in your experiments, first compute all other statistics using *Compute_stats.py* as above, then run the *Complexity.py* script on the file that contains the other statistics:
python Complexity.py input
where *input* is a statistics (.sts) file, which will be modified. The script computes the complexity measure and appends it to the current set of statistics. Note that this operation is not possible without having the FLIF tool on your system.
 
#Step 3
You can choose among the four different OOD detection methods implemented in the following source files:
..* _OOD_osvm.py_: based on One Class SVM and selectable statistics (the method proposed by the paper and Morningstar et al,2020)
..* _OOD__typical.py_: 1-sample typicality test (Nalisnick et al,2019; Morningstar et al,2020)
..* _OOD__compll.py_: based on sum of likelihood and image complexity score (Serra et al, 2020)
..* _OOD__ll.py_: traditional likelihood threshold method
Use the _RunOOD_ function to run the algorithm and see the results. As the input to this function, you need the path to a statistics file (.sts), obtained in step 2, that is used as the in-distribution data. Also, a list of paths to statistics files corresponding to OOD data should be provided (OOD detection will run on each pair of in-distribution/OOD). In *OOD_osvm*, you can also choose the statistics that are used as the input features to the model (which should be present in the input .sts files).
