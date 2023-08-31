# stanford-sleep-and-AD-research
Summer 2023 Internship research project with Joe Winer in the Mormino Lab. 
This project explores Actigraphy as a method of better understanding connections between sleep and neurodegenerative disease. <br>
Various directions are explored, including: 
1) Sleep metric comparisons across various packages (pyActigraphy (python), and GGIR and nPARACT (R))
2) Locomotor Inactivity During Sleep (LIDS) analysis, as put forward by Winnebeck et al. 2018
3) Day-to-day Modeling of objective and subjective sleep, and differences between healthy, Alzheimer’s, and Lewy Body Subjects.

## File Guide:
### Setting Virtual Environment
1) common.py - read in raw actigraphy data from a directory that should be specified (!) by the user <br>
   
### Sleep metric comparisons:
1) compute_and_export_metrics.py - for each unique file (corresponding to a unique subject's data), compute and export relevant sleep metrics as computed by pyActigraphy
2) CRAN_vs_pyActigraphy_analysis.py - *note that CRAN represents the nPARACT package* - prepares the data for both 

