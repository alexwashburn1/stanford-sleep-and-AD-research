library(GGIR)

# Tests
# see parameters in each parameter category and their default value
print(load_params())

# sleep only parameters
print(load_params()$params_sleep)

##### LOADING THE DATA #####
source("/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/Bowdoin-GGIR-plot-comparisons.R")

# analyze the actigraphy record and output it to the output directory
GGIR(datadir="/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/sample_public_data/accsamp.cwa",
     outputdir="/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/GGIR_bowdoin_output", studyname = "my_study") # <- what file extension here??

setwd("/Users/awashburn/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing")

data <- read.csv("sample_public_data/LIDS-sleep-bouts_database.csv")
view(data)

