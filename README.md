# stanford-sleep-and-AD-research
Summer 2023 Internship research project with Joe Winer in the Mormino Lab. 
This project explores Actigraphy as a method of better understanding connections between sleep and neurodegenerative disease. <br>
Various directions are explored, including: 
1) Sleep metric comparisons across various packages (pyActigraphy (python), and GGIR and nPARACT (R))
2) Locomotor Inactivity During Sleep (LIDS) analysis, as put forward by Winnebeck et al. 2018
3) Day-to-day Modeling of objective and subjective sleep, and differences between healthy, Alzheimer’s, and Lewy Body Subjects.

## File Guide:
### Setting Virtual Environment
1) common.py - read in raw actigraphy data from a directory that should be specified (!) by the user (set environment variables) <br>
   
### Sleep Metric Comparisons:
1) compute_and_export_metrics.py - for each unique file (corresponding to a unique subject's data), compute and export relevant sleep metrics as computed by pyActigraphy
2) CRAN_vs_pyActigraphy_analysis.py - *note that CRAN represents the nPARACT package* - prepares the data for both nPARACT and pyActigraphy packages. Plots either (a) a common metric across the two packages or (b) two different metrics from the packages against each other
3) GGIR_vs_CRAN.py - similarly to above, graphs a sleep metric output by GGIR and nPARACT packages against each other
4) GGIR_vs_pyActigraphy.py - graphs a sleep metric output by GGIR and nPARACT packages against each other
5) intra_package_metric_analysis.py - plots different metrics against each other from the same packge. <br>

### Locomotor Inactivity During Sleep (LIDS) Analysis: 
1) LIDS_analysis_new.py - a script that contains ALL of the methods for LIDS analysis. Notably, performs analysis on single files, multiple files, a mean of all files, a normalized mean of all files, and binned normalized means of all files (as outlined in Winnebeck et al. 2018)

### Day to Day Objective vs. Subjective Analysis: 
1) day_to_day_modeling.py - matches the objective and subjective sleep metrics for each unique subject and date, for analysis in R. There is lots of file manipulation to ultimately achieve the goal of a single csv file with both objective and subjective sleep metrics, for multiple days/nights, for each unique subject

### Others: 
The rest of the files are either to get aquainted with pyActigraphy, for visualizations, or works in progress.



