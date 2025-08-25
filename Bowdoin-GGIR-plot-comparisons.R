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
View(data)

# try sorting the data in ascending order
sorted_data <- data[order(data$ID), ]
View(sorted_data)



########## CFA model setup ##########
# import data
setwd("/Users/awashburn/Library/CloudStorage/OneDrive-BowdoinCollege/Documents/Mormino-Lab-Internship/Python-Projects/Actigraphy-Testing/sample_public_data/")
dummy_data <- read.csv("Bowdoin_obj_v_subj_dummy_data.csv")

# set up model
CFA.model <- 'subjective sleep quality  =~ Mentally.Alert + Deep.Sleep + Overall.quality + Well.rested'

# fit model to our data
fit <- cfa(CFA.model, data = dummy_data)

# view the summary of the model
summary(fit, fit.measures = TRUE, standardized = T)

# visualize the output
semPaths(fit, "std", edge.label.cex = 0.5, curvePivot = TRUE)

# add model output as a column to our data
dummy_data$latent_variable <- predict(fit)
